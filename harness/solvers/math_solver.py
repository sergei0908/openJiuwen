"""Арифметические задачи: строгий разбор шаблона (0 токенов) или LLM пишет программу."""
import re

from .base import (
    FINAL_RULES,
    Plan,
    agent_answer,
    approx_tokens,
    strip_boilerplate,
)

N = r"(\d+)"
# Каждое правило: регэксп предложения целиком -> функция (stock, *ints) -> stock
RULES = [
    (rf"(?:A|The) \w+ starts with {N} \w+", lambda s, a: a),
    (rf"It ships {N} \w+ per day for {N} days?", lambda s, a, b: s - a * b),
    (rf"It ships {N} \w+", lambda s, a: s - a),
    (rf"It (?:receives|gets) {N} deliveries of {N} \w+ each", lambda s, a, b: s + a * b),
    (rf"It (?:receives|gets) {N} \w+", lambda s, a: s + a),
    (rf"It (?:loses|sells|gives away) {N} \w+", lambda s, a: s - a),
    (rf"(?:Then )?the stock is split equally into {N} \w+ and it keeps one share \(rounded down\)",
     lambda s, k: s // k),
    (r"(?:Then )?the stock is doubled(?: by merging with a sister \w+)?", lambda s: s * 2),
    (r"(?:Then )?the stock is tripled(?: by merging with two sister \w+)?", lambda s: s * 3),
    (rf"(?:Then )?a customer buys {N} \w+(?: from it)?", lambda s, a: s - a),
    (r"(?:Then )?its stock grows threefold(?: after an acquisition)?", lambda s: s * 3),
    (r"(?:Then )?its stock grows twofold(?: after an acquisition)?", lambda s: s * 2),
]
FINAL = re.compile(r"How many \w+ does it have at the end\?", re.IGNORECASE)


def fast_solve(instr):
    body = strip_boilerplate(instr)
    body = re.sub(r"\s*Compute it with .*$", "", body, flags=re.DOTALL)
    m = FINAL.search(body)
    if not m:
        return None
    body = body[:m.start()].strip()
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", body) if s.strip()]
    if not sentences:
        return None
    start = re.fullmatch(RULES[0][0], sentences[0].rstrip("."), flags=re.IGNORECASE)
    if not start:
        return None
    stock = int(start.group(1))
    for sent in sentences[1:]:
        sent = sent.rstrip(".")
        for pat, fn in RULES[1:]:
            mm = re.fullmatch(pat, sent, flags=re.IGNORECASE)
            if mm:
                try:
                    stock = fn(stock, *[int(x) for x in mm.groups()])
                except ZeroDivisionError:
                    return "IMPOSSIBLE"
                break
        else:
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
          "arithmetic, applying every step in the given order. " + FINAL_RULES)


def solve_llm(task, llm, plan, strong=False):
    from ..oj_agent import make_python_tool
    task_text = re.sub(r"\s*Compute it with .*$", "", strip_boilerplate(task.instruction), flags=re.DOTALL)
    return agent_answer(llm, system_prompt=SYSTEM, query=task_text,
                        tools=[make_python_tool(task.workspace)], strong=strong)
