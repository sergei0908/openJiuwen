"""Общий интерфейс решателей.

analyze(task)  -> Plan: бесплатный (без LLM) анализ: выполнима ли задача, есть ли
                  готовый ответ без LLM, сколько токенов потребует LLM-путь.
solve_llm(...) -> ответ или None (с учётом бюджета).
"""
import re
from dataclasses import dataclass, field

UNSOLVABLE_MARKERS = ("IMPOSSIBLE", "UNKNOWN", "CANNOT", "NOT FOUND", "N/A", "INSUFFICIENT")

ANSWER_RULES = ("The answer must be a single line with ONLY the final value: no units, no words, "
                "no quotes. If the task is impossible (missing data, contradictory), output "
                "exactly IMPOSSIBLE.")


@dataclass
class Plan:
    feasible: bool | None = None       # False -> не тратим токены
    fast_answer: str | None = None     # ответ без LLM
    fast_done: bool = False               # задача уже решена без LLM (например, код починен)
    est_cost: float = 300.0               # ожидаемая стоимость LLM-пути (выходные токены)
    p_success: float = 0.8                # вероятность успеха LLM-пути
    notes: dict = field(default_factory=dict)
    context: str = ""                     # заранее собранный контекст для промпта


def extract_question(instr: str) -> str:
    qs = re.findall(r"[^.?!]*\?", instr)
    return qs[-1].strip() if qs else instr


def strip_boilerplate(instr: str) -> str:
    """Убирает общий хвост инструкций - экономия входных токенов."""
    s = re.sub(r"\s*Work only inside the current directory\.", "", instr)
    s = re.sub(r"\s*When you are done, write .*$", "", s, flags=re.DOTALL)
    return s.strip()


def is_unsolvable(ans: str | None) -> bool:
    if ans is None:
        return True
    a = ans.strip().upper()
    return a == "" or any(a.startswith(m) for m in UNSOLVABLE_MARKERS)


def clean_answer(ans: str) -> str:
    a = (ans or "").strip().strip("`").strip()
    a = re.sub(r"^(final answer|answer)\s*[:=]\s*", "", a, flags=re.IGNORECASE)
    a = a.strip().strip('"').strip("'").strip()
    if re.fullmatch(r"-?\d+\.0+", a):
        a = a.split(".")[0]
    return a


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


FINAL_RULES = ("When you know the answer, reply with exactly one line 'FINAL: <answer>' - only the value, "
               "no units or words. If the task cannot be answered exactly from the given data, "
               "reply 'FINAL: IMPOSSIBLE'.")
# Для решателей с инструментом python: ответ печатает сам скрипт, и агент останавливается сразу.
PRINT_FINAL = ("The script must end with print('FINAL:', answer) - that line is taken as your answer, "
               "so print it only when the computation is complete. ")


def agent_answer(llm, *, system_prompt, query, tools, strong=False, max_tokens=600, max_iterations=3,
                 stop=None):
    """Решение через агента openJiuwen; возвращает очищенный ответ или None."""
    from .. import config
    from ..oj_agent import final_answer
    text = llm.agent(system_prompt=system_prompt, query=query, tools=tools,
                     model=config.MODEL_STRONG if strong else config.MODEL_FAST,
                     max_iterations=max_iterations,
                     max_tokens=config.MAX_OUT_STRONG if strong else max_tokens, stop=stop)
    ans = clean_answer(final_answer(text))
    return None if is_unsolvable(ans) else ans
