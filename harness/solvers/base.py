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
    """Последнее предложение-вопрос. Предложения делим по «. », а не по любой точке: иначе
    «… orders in sales.csv?» превращается в «csv?»."""
    sents = re.split(r"(?<=[.!?])\s+", instr)
    qs = [s.strip() for s in sents if "?" in s]
    if not qs:
        return instr
    q = qs[-1]
    return q[:q.rindex("?") + 1]


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
    m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", a)
    if m and int(m.group(2)):
        # дробь «36/5» -> «7.2», если она записывается конечной десятичной
        from fractions import Fraction
        f = Fraction(int(m.group(1)), int(m.group(2)))
        d = f.denominator
        while d % 2 == 0:
            d //= 2
        while d % 5 == 0:
            d //= 5
        if d == 1:
            a = format(f.numerator / f.denominator, ".10f").rstrip("0").rstrip(".")
    return a


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


FINAL_RULES = ("When you know the answer, reply with exactly one line 'FINAL: <answer>' - only the value, "
               "no units or words. If the task cannot be answered exactly from the given data, "
               "reply 'FINAL: IMPOSSIBLE'.")
# Для решателей с инструментом python: ответ печатает сам скрипт, и агент останавливается сразу.
PRINT_FINAL = ("Call the python tool right away, with no text around the call. Keep the script minimal, "
               "no comments. It must end with print('FINAL:', answer) - that line is taken as your answer "
               "(a plain number - a decimal, not a fraction - or a short value), so print it only when the "
               "computation is complete. ")


def agent_answer(llm, *, system_prompt, query, tools, strong=False, max_tokens=600, max_iterations=3,
                 stop=None):
    """Решение через агента openJiuwen; возвращает очищенный ответ или None."""
    from .. import config
    from ..oj_agent import FINAL_RE, final_answer
    model = config.MODEL_STRONG if strong else config.MODEL_FAST
    text = llm.agent(system_prompt=system_prompt, query=query, tools=tools, model=model,
                     max_iterations=max_iterations,
                     max_tokens=config.MAX_OUT_STRONG if strong else max_tokens, stop=stop)
    ans = clean_answer(final_answer(text))
    if not FINAL_RE.search(text or "") and not plausible_answer(ans):
        # модель не дописала «FINAL:», и последняя строка - обрывок рассуждений («2. **Identify the
        # employer...»). Такой мусор в answer.txt не пишем. Без инструментов (search) повтор дешёвый:
        # просим только строку ответа; с инструментами - задачу считаем нерешённой.
        if tools:
            return None
        text = llm.agent(system_prompt=system_prompt, query=query + RETRY_FINAL, tools=[], model=model,
                         max_iterations=1, max_tokens=min(max_tokens, 40))
        if not FINAL_RE.search(text or ""):
            return None
        ans = clean_answer(final_answer(text))
    return None if is_unsolvable(ans) or not ans else ans


RETRY_FINAL = "\n\nReply with exactly one line 'FINAL: <answer>' and nothing else. No reasoning."
LIST_MARK = re.compile(r"^(?:\d+[.)]|[-*•#>])\s")


def plausible_answer(a):
    """Похожа ли строка на ответ, а не на обрывок рассуждений: короткая, одна строка, без разметки."""
    return bool(a) and len(a) <= 80 and "\n" not in a and "**" not in a and not a.endswith(":") \
        and not LIST_MARK.match(a) and len(a.split()) <= 8
