"""Арифметические задачи: строгий разбор шаблона (0 токенов) или LLM пишет программу."""
import re

from .base import (
    FINAL_RULES,
    PRINT_FINAL,
    Plan,
    agent_answer,
    approx_tokens,
    strip_boilerplate,
)

N = r"(\d+)"
PCT_DOWN = r"(?:damaged|lost|spoiled|stolen|sold|removed|given away|thrown away|discarded|destroyed)"
PCT_UP = r"(?:grows|increases|rises|goes up)"
ROUND = r"(?: and thrown away)?(?:,)?(?: \((?P<rd>rounded down)(?: to (?:whole|full) \w+)?\))?"


def _pct(sign):
    """Процент от запаса. Без «rounded down» берём, только если делится нацело: иначе неясно,
    как округлять, и решение остаётся агенту («уверен или молчит»)."""
    def f(s, p, rd=None):
        part, rem = divmod(s * p, 100)
        if rem and not rd:
            raise ValueError("ambiguous rounding")
        return s + sign * part
    return f


# Каждое правило: регэксп предложения целиком -> функция (stock, *ints) -> stock
RULES = [
    (rf"(?:A|The) \w+ starts with {N} (?P<unit>\w+)", lambda s, a: a),
    (rf"It ships {N} (?P<g>\w+) per day for {N} days?", lambda s, a, b: s - a * b),
    (rf"It ships {N} (?P<g>\w+)", lambda s, a: s - a),
    (rf"It (?:receives|gets) {N} deliveries of {N} (?P<g>\w+) each", lambda s, a, b: s + a * b),
    (rf"It (?:receives|gets) {N} (?P<g>\w+)", lambda s, a: s + a),
    (rf"It (?:loses|sells|gives away) {N} (?P<g>\w+)", lambda s, a: s - a),
    (rf"(?:Then )?the stock is split equally into {N} \w+ and it keeps one share \(rounded down\)",
     lambda s, k: s // k),
    (r"(?:Then )?the stock is doubled(?: by merging with a sister \w+)?", lambda s: s * 2),
    (r"(?:Then )?the stock is tripled(?: by merging with two sister \w+)?", lambda s: s * 3),
    (rf"(?:Then )?a customer buys {N} (?P<g>\w+)(?: from it)?", lambda s, a: s - a),
    (r"(?:Then )?its stock grows threefold(?: after an acquisition)?", lambda s: s * 3),
    (r"(?:Then )?its stock grows twofold(?: after an acquisition)?", lambda s: s * 2),
    (rf"(?:Then )?{N}% of (?:the|its) (?:stock|(?P<g>\w+)) (?:is|are|gets|get) {PCT_DOWN}{ROUND}", _pct(-1)),
    (rf"(?:Then )?(?:the|its) stock {PCT_UP} by {N}%{ROUND}", _pct(+1)),
]
# Лишние сведения без действия над запасом: «The warehouse has 12 employees», «It was built 7 years ago».
# Пропускаем предложение, только если глагол из короткого списка безопасных (has, employs, was built…),
# в нём нет ни слова о запасе (единицы, stock) и ни одного действия. «is reduced by 40», «was halved» -
# это действие, просто незнакомое: такое предложение отдаём агенту, а не пропускаем.
FILLER = re.compile(r"(?:The|Its|It)\b[\w ]* (?:has|have|had|employs|employ|opened|was built|were built|"
                    r"was founded|worked)\b[\w ]*", re.IGNORECASE)
ACTION = re.compile(r"\b(?:ship|send|sent|sell|sold|buy|bought|receiv|get|got|deliver|lose|lost|los|give|gave|add|remov|"
                    r"take|took|damag|spoil|steal|stole|split|doubl|tripl|grow|increas|decreas|drop|keep|kept|"
                    r"return|order|move|transfer|donat|throw|thrown|discard|destroy|use|left|remain|stock|"
                    r"share|half|twice|times|fold|per|each|%)", re.IGNORECASE)


def same_unit(a, b):
    """«box» и «boxes», «crate» и «crates» - одна единица."""
    a, b = a.lower(), b.lower()
    return a == b or a in (b + "s", b + "es") or b in (a + "s", a + "es")


def is_filler(sent, unit):
    if not FILLER.fullmatch(sent) or ACTION.search(sent):
        return False
    words = {w.lower().rstrip("s") for w in re.findall(r"[A-Za-z]+", sent)}
    return unit.lower().rstrip("s") not in words
FINAL = re.compile(r"How many (\w+) does it have at the end\?", re.IGNORECASE)


def _group_names(m):
    """Имя каждой группы по порядку (None - безымянная, это числа)."""
    names = [None] * len(m.groups())
    for name, idx in m.re.groupindex.items():
        names[idx - 1] = name
    return names


def fast_solve(instr):
    body = strip_boilerplate(instr)
    body = re.sub(r"\s*Compute it with .*$", "", body, flags=re.DOTALL)
    m = FINAL.search(body)
    if not m:
        return None
    asked = m.group(1)
    body = body[:m.start()].strip()
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", body) if s.strip()]
    if not sentences:
        return None
    start = re.fullmatch(RULES[0][0], sentences[0].rstrip("."), flags=re.IGNORECASE)
    if not start:
        return None
    stock, unit = int(start.group(1)), start.group("unit")
    for sent in sentences[1:]:
        sent = sent.rstrip(".")
        for pat, fn in RULES[1:]:
            mm = re.fullmatch(pat, sent, flags=re.IGNORECASE)
            if mm:
                kw = {k: v for k, v in mm.groupdict().items() if v is not None}
                goods = kw.pop("g", None)
                if goods and not same_unit(goods, unit):
                    return None  # «It loses 2 employees», «It receives 3 complaints» - не о товаре
                nums = [int(v) for k, v in zip(_group_names(mm), mm.groups(), strict=True) if k is None]
                try:
                    stock = fn(stock, *nums, **kw)
                except ZeroDivisionError:
                    return "IMPOSSIBLE"
                except ValueError:
                    return None  # неоднозначно (округление) - отдаём агенту
                break
        else:
            # лишнее предложение пропускаем, только если спрашивают о той же единице, что в запасе:
            # тогда сведения о другом (сотрудники, годы, ящики другого вида) на ответ не влияют
            if asked.lower().rstrip("s") == unit.lower().rstrip("s") and is_filler(sent, unit):
                continue
            return None  # незнакомое предложение - не рискуем, отдаём LLM
    return str(stock)


def analyze(task, fast_path=True):
    p = Plan()
    if fast_path:
        a = fast_solve(task.instruction)
        if a == "IMPOSSIBLE":
            p.feasible = False
        elif a is not None:
            p.fast_answer = a
    p.est_cost = approx_tokens(task.instruction) * 0.1 + 180
    p.p_success = 0.9
    return p


SYSTEM = ("You solve an arithmetic word problem. Compute it with the python tool using exact integer "
          "arithmetic, applying every step in the given order. " + PRINT_FINAL + FINAL_RULES)


def solve_llm(task, llm, plan, strong=False):
    from ..oj_agent import final_stop, make_python_tool
    state = {}
    task_text = re.sub(r"\s*Compute it with .*$", "", strip_boilerplate(task.instruction), flags=re.DOTALL)
    return agent_answer(llm, system_prompt=SYSTEM, query=task_text,
                        tools=[make_python_tool(task.workspace, state)], strong=strong,
                        stop=final_stop(state))
