"""Вызовы LLM через openJiuwen + учёт бюджета в выходных токенах.

cost = completion_tokens + prompt_tokens / 10 (регламент: 10 входных = 1 выходной,
рассуждения входят в completion, кеш не учитывается).
"""
import asyncio
import logging
import math
import os
import threading
import time

from . import config

_tls = threading.local()


class BudgetExhausted(Exception):
    pass


class Ledger:
    """Потокобезопасный счётчик расхода."""

    def __init__(self, budget=None):
        self.budget = budget
        self.spent = 0.0
        self.calls = 0
        self.exhausted = False
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


def _quiet_openjiuwen_logs():
    for name in list(logging.root.manager.loggerDict):
        if "jiuwen" in name or name in ("llm", "common", "agent", "workflow"):
            logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from loguru import logger as _lg
        _lg.remove()
        _lg.add(lambda m: None)
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

    def chat(self, messages, model=None, max_tokens=None):
        """Возвращает (text, cost). Бросает BudgetExhausted, если прокси перестал отвечать."""
        if self.ledger.exhausted:
            raise BudgetExhausted("ledger exhausted")
        model = model or config.MODEL_FAST
        max_tokens = max_tokens or config.MAX_OUT_FAST
        t0 = time.time()
        last_err = None
        for attempt in range(3):
            try:
                if self._backend == "openjiuwen":
                    text, pt, ct = self._call_oj(messages, model, max_tokens)
                else:
                    text, pt, ct = self._call_httpx(messages, model, max_tokens)
                if pt == 0:  # прокси не вернул usage - оцениваем сами, с запасом
                    pt = sum(len(str(m.get("content", ""))) for m in messages) // 3
                    ct = max(ct, len(text) // 3)
                cost = self.ledger.add(pt, ct)
                return text, cost, {"model": model, "prompt_tokens": pt,
                                    "completion_tokens": ct, "latency": time.time() - t0}
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                if any(s in msg for s in ("budget", "quota", "exhaust", "402", "insufficient")):
                    self.ledger.exhausted = True
                    raise BudgetExhausted(str(e))
                if "429" in msg or "rate" in msg:
                    time.sleep(2 + 3 * attempt)
                    continue
                if any(s in msg for s in ("connect", "timeout", "timed out", "503", "502")):
                    time.sleep(1 + attempt)
                    continue
                break
        # Прокси молчит после нескольких попыток - считаем бюджет исчерпанным.
        if last_err is not None and any(s in str(last_err).lower()
                                        for s in ("connect", "429", "refused")):
            self.ledger.exhausted = True
            raise BudgetExhausted(str(last_err))
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
