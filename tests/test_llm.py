"""Тесты реакции на ошибки прокси: конец бюджета, лимит частоты, недоступная модель."""
import pytest

from harness import config, llm
from harness.llm import LLM, BudgetExhausted, Ledger

MSGS = [{"role": "user", "content": "hi"}]


class FakeProxy:
    """Подменяет вызов модели: каждое значение из script - ответ или ошибка очередного запроса."""

    def __init__(self, script, then=None):
        self.script, self.then, self.calls, self.models = list(script), then, 0, []

    def __call__(self, messages, model, max_tokens):
        self.calls += 1
        self.models.append(model)
        item = self.script.pop(0) if self.script else self.then
        if isinstance(item, Exception):
            raise item
        return item


OK = ("4", 100, 10)


@pytest.fixture
def make_llm(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def make(script, then=None, budget=100000):
        client = LLM(Ledger(budget))
        fake = FakeProxy(script, then)
        monkeypatch.setattr(client, "_call_oj", fake)
        monkeypatch.setattr(client, "_call_httpx", fake)
        return client, fake
    return make


def test_cost_is_output_plus_tenth_of_input(make_llm):
    client, _ = make_llm([OK])
    _, cost, meta = client.chat(MSGS)
    assert cost == pytest.approx(20)
    assert client.ledger.spent == pytest.approx(20)
    assert (meta["prompt_tokens"], meta["completion_tokens"]) == (100, 10)


@pytest.mark.parametrize("text", [
    "Error code: 402 - payment required",
    "token budget exhausted",
    "Бюджет исчерпан",
    "insufficient funds",
    "лимит токенов исчерпан",
])
def test_explicit_budget_error_stops_after_one_request(make_llm, text):
    client, fake = make_llm([], then=RuntimeError(text))
    with pytest.raises(BudgetExhausted):
        client.chat(MSGS)
    assert fake.calls == 1
    assert client.ledger.exhausted


@pytest.mark.parametrize("text", [
    "Error code: 500 - internal server error",
    "Connection error.",
    "Request timed out.",
    "Error code: 503",
    "unexpected response",
])
def test_silent_proxy_is_exhausted_within_six_failed_requests(make_llm, text):
    client, fake = make_llm([], then=RuntimeError(text))
    for _ in range(10):
        try:
            client.chat(MSGS)
        except BudgetExhausted:
            break
        except RuntimeError:
            continue
    assert client.ledger.exhausted
    assert fake.calls <= config.FAIL_STREAK_EXHAUSTED


def test_no_calls_after_exhaustion(make_llm):
    client, fake = make_llm([], then=RuntimeError("budget"))
    with pytest.raises(BudgetExhausted):
        client.chat(MSGS)
    with pytest.raises(BudgetExhausted):
        client.chat(MSGS)
    assert fake.calls == 1


def test_success_resets_failure_streak(make_llm):
    errors = [RuntimeError("Connection error.")] * (config.FAIL_STREAK_EXHAUSTED - 1)
    client, _ = make_llm(errors + [OK])
    for _ in range(3):  # один вызов делает до 3 попыток при сетевой ошибке
        try:
            assert client.chat(MSGS)[0] == "4"
            break
        except RuntimeError:
            continue
    assert client.ledger.fail_streak == 0
    assert not client.ledger.exhausted


def test_rate_limit_is_retried_not_exhausted(make_llm):
    client, fake = make_llm([RuntimeError("Error code: 429 - 3 уже в работе при потолке тарифа 3")] * 5 + [OK])
    assert client.chat(MSGS)[0] == "4"
    assert fake.calls == 6
    assert not client.ledger.exhausted


def test_endless_rate_limit_means_exhausted(make_llm):
    client, _ = make_llm([], then=RuntimeError("Error code: 429 - слишком много запросов"))
    client.ledger.last_ok -= config.RATE_GIVE_UP_S + 1
    with pytest.raises(BudgetExhausted):
        client.chat(MSGS)
    assert client.ledger.exhausted


def test_unknown_429_gives_up_sooner_than_known_rate_limit(make_llm):
    # RateLimitError - имя класса в тексте ошибки клиента, а не признак лимита частоты
    client, _ = make_llm([], then=RuntimeError("RateLimitError: Error code: 429 - limit reached"))
    client.ledger.last_ok -= config.UNKNOWN_429_GIVE_UP_S + 1
    assert config.UNKNOWN_429_GIVE_UP_S < config.RATE_GIVE_UP_S
    with pytest.raises(BudgetExhausted):
        client.chat(MSGS)


def test_recent_unknown_429_is_retried(make_llm):
    client, fake = make_llm([RuntimeError("Error code: 429 - limit reached")] * 3 + [OK])
    assert client.chat(MSGS)[0] == "4"
    assert fake.calls == 4
    assert not client.ledger.exhausted


@pytest.mark.parametrize("text", [
    "failed to generate a response",  # слово generate не должно считаться лимитом частоты (rate)
    "model call failed, moderated",
])
def test_words_containing_rate_are_not_rate_limit(make_llm, text):
    client, fake = make_llm([], then=RuntimeError(text))
    with pytest.raises(RuntimeError):
        client.chat(MSGS)
    assert fake.calls == 3  # обычные повторы, а не 12 попыток лимита частоты


FREE_TIER = ("PermissionDeniedError: Error code: 403 - Модель qwen3.8-27b-noreason не входит в бесплатный "
             "тариф. Она доступна на платных тарифах и с кошелька по токенам: https://neuraldeep.ru/pricing.")


def test_unavailable_model_falls_back_without_counting_failure(make_llm):
    client, fake = make_llm([RuntimeError(FREE_TIER), OK])
    text, _, meta = client.chat(MSGS, model=config.MODEL_FAST)
    assert text == "4"
    assert fake.models == [config.MODEL_FAST, config.MODEL_ALT]
    assert meta["model"] == config.MODEL_ALT
    assert client.ledger.fail_streak == 0
    assert not client.ledger.exhausted


def test_unavailable_model_is_not_tried_again(make_llm):
    client, fake = make_llm([RuntimeError(FREE_TIER), OK, OK])
    client.chat(MSGS, model=config.MODEL_FAST)
    client.chat(MSGS, model=config.MODEL_FAST)
    assert fake.models == [config.MODEL_FAST, config.MODEL_ALT, config.MODEL_ALT]


def test_strong_model_falls_back_to_strong_alternative(make_llm):
    client, fake = make_llm([RuntimeError(FREE_TIER), OK])
    client.chat(MSGS, model=config.MODEL_STRONG)
    assert fake.models == [config.MODEL_STRONG, config.MODEL_STRONG_ALT]


def test_all_models_unavailable_stops_llm_calls(make_llm):
    client, fake = make_llm([], then=RuntimeError(FREE_TIER))
    with pytest.raises(BudgetExhausted):
        client.chat(MSGS)
    assert fake.calls == 4
    assert client.ledger.exhausted


def test_missing_usage_is_estimated(make_llm):
    client, _ = make_llm([("abcdef", 0, 0)])
    _, cost, meta = client.chat([{"role": "user", "content": "x" * 300}])
    assert meta["prompt_tokens"] == 100
    assert meta["completion_tokens"] == 2
    assert cost == pytest.approx(12)
