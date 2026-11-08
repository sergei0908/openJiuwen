"""Вызовы LLM через openJiuwen + учёт бюджета в выходных токенах.

cost = completion_tokens + prompt_tokens / 10 (регламент: 10 входных = 1 выходной,
рассуждения входят в completion, кеш не учитывается).
"""
import asyncio
import logging
import math
import os
import re
import threading
import time

from . import config

_tls = threading.local()
_SLOTS = threading.BoundedSemaphore(config.MAX_CONCURRENT)

BUDGET_RE = re.compile(r"budget|бюджет|exhaust|исчерпан|\b402\b|insufficient|quota exceeded|"
                       r"лимит токенов")
RATE_RE = re.compile(r"rate[ _-]limit|too many|слишком много|одновремен|concurrent|limit resets|"
                     r"уже в работе|потолк")
HTTP429_RE = re.compile(r"\b429\b")
# Модель недоступна на этом ключе или не существует: это не конец бюджета, нужна другая модель.
MODEL_RE = re.compile(r"(\b403\b|\b404\b).*(модел|model)|(модел|model).*(\b403\b|\b404\b)|"
                      r"не входит в .*тариф|model.*(not found|does not exist)|permission.?denied")


class BudgetExhausted(Exception):
    pass


class Ledger:
    """Потокобезопасный счётчик расхода."""

    def __init__(self, budget=None):
        self.budget = budget
        self.spent = 0.0
        self.calls = 0
        self.exhausted = False
        self.fail_streak = 0
        self.last_ok = time.time()
        self._lock = threading.Lock()

    def add(self, prompt_tokens, completion_tokens):
        c = completion_tokens + prompt_tokens * config.INPUT_WEIGHT
        with self._lock:
            self.spent += c
            self.calls += 1
        return c

    def remaining(self):
        if self.budget is None:
            return math.inf
        return self.budget * (1 - config.RESERVE_FRAC) - self.spent

    def can_afford(self, est):
        return not self.exhausted and self.remaining() >= est

    def ok(self):
        with self._lock:
            self.fail_streak = 0
            self.last_ok = time.time()

    def failed(self):
        """Неудачный запрос. True, если подряд их столько, что прокси, видимо, закрыл бюджет."""
        with self._lock:
            self.fail_streak += 1
            if self.fail_streak >= config.FAIL_STREAK_EXHAUSTED:
                self.exhausted = True
            return self.exhausted


def _quiet_openjiuwen_logs():
    """openJiuwen по умолчанию пишет INFO по каждому вызову модели - ошибки харнесс логирует сам."""
    try:
        from openjiuwen.core.common.logging.manager import LogManager
        for name in ("common", "llm", "agent", "tool", "runner", "prompt", "interface",
                     "performance", "session", "sys_operation"):
            LogManager.get_logger(name).set_level(logging.CRITICAL)
    except Exception:  # noqa: BLE001, S110 - логирование openJiuwen необязательно
        pass


class LLM:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("LLM_API_KEY", "")
        self._models = {}
        self._lock = threading.Lock()
        self._backend = None
        self.unavailable = set()
        try:
            from openjiuwen.core.foundation.llm.model import Model  # noqa: F401
            self._backend = "openjiuwen"
            _quiet_openjiuwen_logs()
        except Exception:  # noqa: BLE001 - без openJiuwen работаем через httpx
            self._backend = "httpx"

    def _oj_model(self, name):
        with self._lock:
            if name not in self._models:
                from openjiuwen.core.foundation.llm.model import Model
                from openjiuwen.core.foundation.llm.schema.config import (
                    ModelClientConfig,
                    ModelRequestConfig,
                )
                self._models[name] = Model(
                    ModelClientConfig(client_provider="OpenAI", api_key=self.key or "none",
                                      api_base=self.base, timeout=config.LLM_TIMEOUT_S,
                                      use_shared_llm_http_client=False),
                    ModelRequestConfig(model=name, temperature=0),
                )
                _quiet_openjiuwen_logs()
            return self._models[name]

    def _fallback(self, model):
        """Ближайшая доступная модель: задача не должна пропасть из-за недоступной модели."""
        chain = (config.MODEL_STRONG, config.MODEL_STRONG_ALT, config.MODEL_FAST, config.MODEL_ALT) \
            if model in (config.MODEL_STRONG, config.MODEL_STRONG_ALT) \
            else (config.MODEL_FAST, config.MODEL_ALT, config.MODEL_STRONG_ALT, config.MODEL_STRONG)
        return next((m for m in (model, *chain) if m not in self.unavailable), None)

    def handle_error(self, e, current, attempt):
        """Решение по ошибке прокси - общее для chat() и агента openJiuwen (oj_agent.BudgetModel).

        Возвращает ("switch", 0) - модель недоступна, взять следующую; ("retry", пауза) - повторить;
        ("fail", 0) - этот запрос не удался. Бросает BudgetExhausted, если бюджет закончился.
        """
        msg = str(e).lower()
        if BUDGET_RE.search(msg):
            self.ledger.exhausted = True
            raise BudgetExhausted(str(e)) from e
        if MODEL_RE.search(msg):
            self.unavailable.add(current)
            return "switch", 0
        known = RATE_RE.search(msg)
        if known or HTTP429_RE.search(msg):
            # лимит частоты - это не конец бюджета, просто ждём
            limit = config.RATE_GIVE_UP_S if known else config.UNKNOWN_429_GIVE_UP_S
            if time.time() - self.ledger.last_ok > limit:
                self.ledger.exhausted = True
                raise BudgetExhausted(str(e)) from e
            return "retry", min(20, 1.5 * (attempt + 1))
        if self.ledger.failed():
            raise BudgetExhausted(str(e)) from e
        if attempt < 2:
            return "retry", 1 + attempt
        return "fail", 0

    def chat(self, messages, model=None, max_tokens=None):
        """Возвращает (text, cost, meta). Бросает BudgetExhausted, если прокси перестал отвечать."""
        if self.ledger.exhausted:
            raise BudgetExhausted("ledger exhausted")
        model = model or config.MODEL_FAST
        max_tokens = max_tokens or config.MAX_OUT_FAST
        t0 = time.time()
        last_err = None
        for attempt in range(12):
            if self.ledger.exhausted:
                raise BudgetExhausted(str(last_err))
            current = self._fallback(model)
            if current is None:  # ни одна модель не отвечает: звать больше некого
                self.ledger.exhausted = True
                raise BudgetExhausted(f"no available model: {last_err}")
            try:
                with _SLOTS:  # не больше MAX_CONCURRENT запросов одновременно
                    if self._backend == "openjiuwen":
                        text, pt, ct = self._call_oj(messages, current, max_tokens)
                    else:
                        text, pt, ct = self._call_httpx(messages, current, max_tokens)
                if pt == 0:  # прокси не вернул usage - оцениваем сами, с запасом
                    pt = sum(len(str(m.get("content", ""))) for m in messages) // 3
                    ct = max(ct, len(text) // 3)
                cost = self.ledger.add(pt, ct)
                self.ledger.ok()
                return text, cost, {"model": current, "prompt_tokens": pt,
                                    "completion_tokens": ct, "latency": time.time() - t0}
            except Exception as e:  # noqa: BLE001 - любая ошибка прокси разбирается в handle_error
                last_err = e
                action, delay = self.handle_error(e, current, attempt)
                if action == "switch":
                    continue
                if action == "retry":
                    time.sleep(delay)
                    continue
                break
        raise RuntimeError(f"LLM call failed: {last_err}")

    def _call_oj(self, messages, model, max_tokens):
        m = self._oj_model(model)

        loop = getattr(_tls, "loop", None)
        if loop is None or loop.is_closed():
            loop = _tls.loop = asyncio.new_event_loop()
        r = loop.run_until_complete(m.invoke(messages, max_tokens=max_tokens, temperature=0))
        text = r.content if isinstance(r.content, str) else str(r.content)
        u = r.usage_metadata
        pt = (u.input_tokens if u else 0) or 0
        ct = (u.output_tokens if u else 0) or 0
        # reasoning_tokens обычно уже внутри completion_tokens; если провайдер отдаёт
        # их отдельно и completion меньше - берём максимум, чтобы не занижать расход.
        if u and u.reasoning_tokens and u.reasoning_tokens > ct:
            ct = u.reasoning_tokens + len(text) // 4
        return text, pt, ct

    def _call_httpx(self, messages, model, max_tokens):
        import httpx
        r = httpx.post(f"{self.base}/chat/completions",
                       headers={"Authorization": f"Bearer {self.key}"},
                       json={"model": model, "messages": messages, "temperature": 0,
                             "max_tokens": max_tokens},
                       timeout=config.LLM_TIMEOUT_S)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        j = r.json()
        text = j["choices"][0]["message"].get("content") or ""
        u = j.get("usage") or {}
        return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
