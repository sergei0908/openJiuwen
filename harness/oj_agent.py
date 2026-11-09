"""LLM-путь харнесса: агент openJiuwen (DeepAgent) с нашими лёгкими инструментами.

Почему так, а не готовый code agent openJiuwen: стандартный набор (14 инструментов и большой
системный промпт) весит около 38 000 символов на каждый запрос, а бюджет всего прогона -
100 000 токенов. Здесь агент openJiuwen получает:
  - короткий системный промпт под тип задачи;
  - 1-3 своих инструмента (@tool openJiuwen) с короткими описаниями;
  - заранее собранный харнессом контекст (нужные файлы, а не вся папка);
  - лимит шагов max_iterations.
Один запрос в таком режиме - около 500-1500 символов.

Учёт бюджета: агент сам вызывает модель, поэтому каждый вызов проходит через BudgetModel -
обёртку над openJiuwen Model, которая списывает расход в общий Ledger и в бюджет текущей
задачи и ограничивает число одновременных запросов. Ошибки прокси разбирает тот же
LLM.handle_error, что и в обычном вызове: конец бюджета, лимит частоты (429), недоступная
модель (переход на следующую по цепочке LLM._fallback).

Все агенты работают в ОДНОМ цикле событий в отдельном потоке. В openJiuwen есть общие на процесс
асинхронные блокировки (например, файловые в sys_operation): если агенты живут в разных циклах,
ожидание блокировки из чужого цикла зависает до таймаута (~60 с). Поэтому потоки-воркеры
планировщика только отправляют задачу в общий цикл и ждут результат; инструменты асинхронные
и тяжёлую работу (подпроцессы) выносят в пул потоков, чтобы не блокировать остальных агентов.
"""
import asyncio
import contextvars
import re
import shutil
import tempfile
import threading
from pathlib import Path

from . import config
from .llm import BudgetExhausted, _quiet_openjiuwen_logs

# Текущая задача (TaskLLM) - нужна BudgetModel, чтобы списать расход именно с неё.
CURRENT_TASK = contextvars.ContextVar("burnrate_current_task", default=None)
# Условие досрочной остановки агента (например, тесты уже прошли): проверяется перед каждым
# вызовом модели, чтобы не платить за лишний финальный ответ.
STOP = contextvars.ContextVar("burnrate_agent_stop", default=None)
FINAL_RE = re.compile(r"FINAL\s*:\s*(.+)", re.IGNORECASE)
TOOL_OUTPUT_LIMIT = 1500

_loop_lock = threading.Lock()
_LOOP = None
_SLOTS = None  # asyncio.Semaphore общего цикла: не больше MAX_CONCURRENT запросов к модели


class TaskBudgetExceeded(Exception):
    pass


class AgentStopped(Exception):
    """Цель задачи уже достигнута - следующий вызов модели не нужен. args[0] - финальный текст."""


def _shared_loop():
    """Общий цикл событий для всех агентов (создаётся при первом вызове)."""
    global _LOOP, _SLOTS
    with _loop_lock:
        if _LOOP is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="oj-agents", daemon=True).start()

            async def make_sem():
                return asyncio.Semaphore(config.MAX_CONCURRENT)
            _SLOTS = asyncio.run_coroutine_threadsafe(make_sem(), loop).result()
            _LOOP = loop
        return _LOOP


def _usage(msg):
    u = getattr(msg, "usage_metadata", None)
    if not u:
        return 0, 0
    pt = u.input_tokens or 0
    ct = u.output_tokens or 0
    # рассуждения считаются как выходные, даже если провайдер отдаёт их отдельно
    if u.reasoning_tokens and u.reasoning_tokens > ct:
        ct = u.reasoning_tokens + ct
    return pt, ct


# Шаблон qwen добавляет к запросу с инструментами ~250 токенов описания (замер 9.11: вход math
# 470 токенов при ~220 по тексту запроса) - это ~25 в единицах стоимости.
TOOL_TEMPLATE_COST = 25
MIN_ANSWER_TOKENS = 30
TYPICAL_ANSWER_TOKENS = 60  # типичный ответ агента (замер 9.11: 57-89 токенов выхода на вызов)


def _call_floor(messages):
    """Минимальная стоимость вызова: вход (по тексту + шаблон) и короткий ответ."""
    return _approx_prompt_tokens(messages) * config.INPUT_WEIGHT + TOOL_TEMPLATE_COST + MIN_ANSWER_TOKENS


def _approx_prompt_tokens(messages):
    return sum(len(str(getattr(m, "content", m))) for m in (messages or [])) // 3


def make_budget_model(llm, model_name, max_tokens):
    """openJiuwen Model, у которого каждый вызов списывается с бюджета."""
    from openjiuwen.core.foundation.llm.model import Model
    from openjiuwen.core.foundation.llm.schema.config import (
        ModelClientConfig,
        ModelRequestConfig,
    )

    class BudgetModel(Model):
        def supports_kv_cache_affinity(self):
            return False  # KV-кеш Ascend не нужен: кеш прокси всё равно не учитывается

        def _charge(self, pt, ct, messages, text_len, used_model):
            if pt == 0:  # прокси не вернул usage - оцениваем сами, с запасом
                pt = _approx_prompt_tokens(messages)
                ct = max(ct, text_len // 3)
            cost = llm.ledger.add(pt, ct)
            llm.ledger.ok()
            task = CURRENT_TASK.get()
            if task is not None:
                task.record(used_model, pt, ct, cost)
            return cost

        def _check(self, messages=None):
            stop = STOP.get()
            res = stop() if stop is not None else None
            if res:
                raise AgentStopped(res if isinstance(res, str) else "FINAL: DONE")
            if llm.ledger.exhausted:
                raise BudgetExhausted("ledger exhausted")
            task = CURRENT_TASK.get()
            if task is not None and task.spent >= task.cap:
                raise TaskBudgetExceeded(f"task cap {task.cap:.0f}")

        def _reserve_call(self, messages):
            """Резервирует ожидаемую стоимость вызова сверх резерва задачи. Параллельные агенты иначе
            видят один и тот же остаток и вместе выходят за бюджет. Возвращает, сколько освободить
            после вызова; вызов, который не помещается в бюджет, не делаем."""
            if llm.ledger.budget is None:
                return 0.0
            task = CURRENT_TASK.get()
            if task is not None and not task.calls and task.res_left > 0:
                # первый вызов задачи покрывает её собственный резерв (прогноз оценщика): иначе дешёвая
                # задача вроде search (~27) не стартует в конце бюджета из-за грубой оценки вызова (~130)
                return 0.0
            need = _call_floor(messages) + TYPICAL_ANSWER_TOKENS
            extra = max(0.0, need - (task.res_left if task is not None else 0.0))
            if extra and not llm.ledger.try_reserve(extra):
                raise TaskBudgetExceeded(f"budget left {llm.ledger.free():.0f}")
            return extra

        def _limit_max_tokens(self, kwargs, messages=None):
            task = CURRENT_TASK.get()
            rem = task.available() if task is not None else llm.ledger.free()
            if rem != float("inf"):
                rem -= _approx_prompt_tokens(messages) * config.INPUT_WEIGHT + TOOL_TEMPLATE_COST
            mt = kwargs.get("max_tokens") or max_tokens
            if rem != float("inf"):
                mt = int(max(MIN_ANSWER_TOKENS, min(mt, rem)))
            kwargs["max_tokens"] = mt
            kwargs.setdefault("temperature", 0)

        async def invoke(self, messages, **kwargs):
            self._limit_max_tokens(kwargs, messages)
            kwargs.pop("model", None)
            last = None
            for attempt in range(12):
                self._check(messages)
                current = llm._fallback(model_name)
                if current is None:  # ни одна модель не отвечает: звать больше некого
                    llm.ledger.exhausted = True
                    raise BudgetExhausted(f"no available model: {last}")
                held = self._reserve_call(messages)
                try:
                    async with _SLOTS:
                        r = await super().invoke(messages, model=current, **kwargs)
                    llm.ledger.release(held)
                    held = 0.0
                    pt, ct = _usage(r)
                    self._charge(pt, ct, messages, len(str(r.content or "")), current)
                    return r
                except (BudgetExhausted, TaskBudgetExceeded):
                    llm.ledger.release(held)
                    raise
                except Exception as e:  # noqa: BLE001 - разбирает LLM.handle_error
                    llm.ledger.release(held)
                    last = e
                    action, delay = llm.handle_error(e, current, attempt)
                    if action == "switch":
                        continue
                    if action == "retry":
                        await asyncio.sleep(delay)
                        continue
                    break
                except BaseException:  # отмена по таймауту: резерв вызова тоже возвращаем
                    llm.ledger.release(held)
                    raise
            raise RuntimeError(f"LLM call failed: {last}")

        async def stream(self, messages, **kwargs):
            # агенту отдаём один готовый ответ: так проще считать расход и повторять при 429
            yield await self.invoke(messages, **kwargs)

    _quiet_openjiuwen_logs()
    return BudgetModel(
        ModelClientConfig(client_provider="OpenAI", api_key=llm.key or "none", api_base=llm.base,
                          timeout=config.LLM_TIMEOUT_S, use_shared_llm_http_client=False),
        ModelRequestConfig(model=model_name, temperature=0, max_tokens=max_tokens),
    )


# ---------------- инструменты (openJiuwen @tool) ----------------

def _clip(s):
    s = s or ""
    return s if len(s) <= TOOL_OUTPUT_LIMIT else "...\n" + s[-TOOL_OUTPUT_LIMIT:]


def make_python_tool(ws: Path, state=None):
    """Инструмент python. Если передан state, строка 'FINAL: ...' в выводе успешного скрипта
    запоминается в state["final"] - по ней агент останавливается (см. final_stop), и модели
    не нужен ещё один вызов, чтобы переписать готовый ответ."""
    from openjiuwen.core.foundation.tool.tool import tool

    @tool(name="python", description="Run a Python 3 script in the task directory; returns stdout and stderr.",
          input_params={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]})
    async def python(code: str) -> str:
        from . import sandbox
        rc, out, err = await asyncio.to_thread(sandbox.run_python, code, ws, 20)
        if state is not None and rc == 0:
            found = FINAL_RE.findall(out)
            if found and found[-1].strip():
                state["final"] = found[-1].strip()
        return _clip(out + (("\n[stderr]\n" + err) if err.strip() else "") + f"\n[exit {rc}]")

    return python


def final_stop(state):
    """Условие остановки для run_agent: ответ уже напечатан скриптом."""
    return lambda: f"FINAL: {state['final']}" if state.get("final") else None


def make_coding_tools(ws: Path, editable, test_cmd):
    """Инструмент coding: update_code вклеивает исправленные функции (coding_solver.merge_reply)
    и сразу прогоняет тесты. Выход модели в 10 раз дороже входа, поэтому целый файл она не
    переписывает, а отдельный вызов run_tests не нужен.

    Возвращает (tools, state); state["passed"] = True, когда тесты прошли - по нему агент
    останавливается без лишнего вызова модели (см. STOP)."""
    from openjiuwen.core.foundation.tool.tool import tool

    from . import sandbox
    from .solvers.coding_solver import NO_TESTS, compact_test_output, merge_reply
    from .tasks import write_file

    editable = sorted(editable)
    state = {"passed": False, "updates": 0}

    @tool(name="update_code",
          description="Replace functions in a source file with corrected versions and run the tests. "
                      "Pass ONLY the corrected functions, each complete; do not repeat unchanged ones. "
                      "Returns the test result.",
          input_params={"type": "object", "properties": {
              "code": {"type": "string", "description": "corrected functions"},
              "file": {"type": "string", "description": "file name; may be omitted if there is one file"}},
              "required": ["code"]})
    async def update_code(code: str, file: str = "") -> str:
        name = file.strip().lstrip("./") if file else (editable[0] if len(editable) == 1 else "")
        if name not in editable:  # модель назвала файл без папки: «mylib.py» вместо «src/mylib.py»
            same = [e for e in editable if Path(e).name == Path(name).name]
            name = same[0] if len(same) == 1 else name
        if name not in editable:
            return f"error: choose file from: {', '.join(editable)}"
        path = ws / name
        merged = merge_reply(path.read_text(encoding="utf-8"), code)
        if merged is None:
            return "error: code is not valid Python or has no top-level functions"
        await asyncio.to_thread(write_file, path, merged)
        state["updates"] += 1
        rc, out, err = await asyncio.to_thread(sandbox.run, test_cmd, ws, 60)
        if rc == 0 and not NO_TESTS.search(out + "\n" + err):
            state["passed"] = True
            return f"updated {name}; all tests passed"
        return f"updated {name}; tests failed:\n" + compact_test_output(out + "\n" + err)

    return [update_code], state


# ---------------- запуск агента ----------------

async def _run_in_loop(llm, task_llm, system_prompt, query, tools, model_name, max_iterations,
                       max_tokens, timeout_s, stop=None):
    from openjiuwen.core.kv_cache.kv_cache_config import KVCacheAffinityConfig
    from openjiuwen.harness.factory import create_deep_agent

    CURRENT_TASK.set(task_llm)  # контекст этой задачи asyncio; вложенные задачи его наследуют
    STOP.set(stop)
    model = make_budget_model(llm, model_name, max_tokens)
    # служебные файлы DeepAgent (AGENT.md, memory/ ...) - в /tmp, а не в папку задачи
    home = Path(tempfile.mkdtemp(prefix="oj_agent_", dir=tempfile.gettempdir()))
    try:
        agent = create_deep_agent(
            model, system_prompt=system_prompt, tools=tools, max_iterations=max_iterations,
            workspace=str(home), language="en", restrict_to_work_dir=True,
            enable_read_image_multimodal=False, enable_security_rail=False,
            enable_model_anomaly_detection_rail=False,
            kv_cache_affinity_config=KVCacheAffinityConfig(enable_kv_cache_affinity=False),
        )
        try:
            return await asyncio.wait_for(agent.invoke({"query": query}), timeout_s)
        except AgentStopped as e:
            return {"output": e.args[0]}
    finally:
        await asyncio.to_thread(shutil.rmtree, home, True)


def run_agent(llm, task_llm, *, system_prompt, query, tools, model_name, max_iterations=4,
              max_tokens=800, timeout_s=150, stop=None):
    """Решает задачу агентом openJiuwen в общем цикле. Возвращает финальный текст агента."""
    loop = _shared_loop()
    fut = asyncio.run_coroutine_threadsafe(
        _run_in_loop(llm, task_llm, system_prompt, query, tools, model_name, max_iterations,
                     max_tokens, timeout_s, stop),
        loop)
    try:
        result = fut.result(timeout_s + 30)
    except BaseException:
        fut.cancel()
        raise
    if isinstance(result, dict):
        return str(result.get("output") or "")
    return str(result or "")


def shutdown():
    """Гасит фоновые задачи openJiuwen и останавливает общий цикл (в конце прогона)."""
    global _LOOP
    loop = _LOOP
    if loop is None:
        return

    async def _cancel_all():
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    try:
        asyncio.run_coroutine_threadsafe(_cancel_all(), loop).result(10)
    except Exception:  # noqa: BLE001, S110 - завершение прогона, ошибки здесь не важны
        pass
    loop.call_soon_threadsafe(loop.stop)
    _LOOP = None


def final_answer(text):
    """Достаёт ответ из 'FINAL: ...'; без маркера - последнюю непустую строку."""
    m = FINAL_RE.findall(text or "")
    if m:
        return m[-1].strip().splitlines()[0].strip()
    lines = [x.strip() for x in (text or "").strip().splitlines() if x.strip()]
    return lines[-1] if lines else ""
