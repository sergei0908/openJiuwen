"""Аналитика по CSV: строгий разбор вопроса (0 токенов) или LLM пишет программу.

В промпт идёт только заголовок и 3 строки файла, не весь CSV - так стоимость не растёт
с размером данных.
"""
import csv
import re

from .. import sandbox
from .base import (
    Plan,
    approx_tokens,
    clean_answer,
    extract_question,
    is_unsolvable,
    strip_boilerplate,
)

Q = r"'([^']+)'"
METRICS = [
    (r"How many units of", "units"),
    (r"How many units in total of", "units"),
    (r"What is the total revenue \(units ?\* ?price\) (?:of|for)", "revenue"),
    (r"What is the total revenue (?:of|for)", "revenue"),
    (r"How many orders (?:of|for)", "orders"),
]
FILTERS = [
    (rf"product {Q}", "product"),
    (rf"in(?: the)? region {Q}", "region"),
    (r"in months (\d+) through (\d+) inclusive", "months"),
    (r"in months (\d+) to (\d+) inclusive", "months"),
    (r"in month (\d+)", "month"),
]


def parse_question(q):
    q = q.strip()
    for mpat, metric in METRICS:
        m = re.match(mpat + r"\s+", q, flags=re.IGNORECASE)
        if not m:
            continue
        rest = q[m.end():]
        rest = re.sub(r"\s+(?:were|was) sold", "", rest, flags=re.IGNORECASE)
        rest = re.sub(r"\s+were placed", "", rest, flags=re.IGNORECASE)
        rest = rest.rstrip("?").strip()
        filt = {}
        while rest:
            for fpat, key in FILTERS:
                fm = re.match(fpat + r"\s*(?:,\s*|and\s+)?", rest, flags=re.IGNORECASE)
                if fm:
                    if key in filt or (key == "month" and "months" in filt):
                        return None
                    filt[key] = fm.groups()
                    rest = rest[fm.end():].strip()
                    break
            else:
                return None  # остался непонятый кусок
        return metric, filt
    return None


def compute(rows, metric, filt):
    total = 0
    for r in rows:
        try:
            if "product" in filt and r["product"].strip() != filt["product"][0]:
                continue
            if "region" in filt and r["region"].strip() != filt["region"][0]:
                continue
            mo = int(float(r["month"]))
            if "months" in filt:
                a, b = map(int, filt["months"])
                if not (a <= mo <= b):
                    continue
            if "month" in filt and mo != int(filt["month"][0]):
                continue
            u = float(r["units"])
            if metric == "units":
                total += u
            elif metric == "revenue":
                total += u * float(r["price"])
            elif metric == "orders":
                total += 1
        except (KeyError, ValueError):
            return None
    return str(int(total)) if float(total).is_integer() else f"{total:.2f}"


def csv_files(ws):
    return sorted(p for p in ws.glob("*.csv") if p.is_file())


def analyze(task, fast_path=True):
    p = Plan()
    files = csv_files(task.workspace)
    q = extract_question(task.instruction)
    p.notes["question"] = q
    if fast_path and len(files) == 1 and re.search(r"sales\.csv", task.instruction):
        parsed = parse_question(q)
        if parsed:
            with files[0].open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            a = compute(rows, *parsed)
            if a is not None:
                p.fast_answer = a
                return p
    if not files:
        p.notes["generic"] = True
        p.est_cost = 1500
        p.p_success = 0.5
        return p
    preview = []
    for f in files[:3]:
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        preview.append(f"--- {f.name} ({len(lines) - 1} data rows), first lines:\n" + "\n".join(lines[:4]))
    p.context = "\n".join(preview)
    p.est_cost = (approx_tokens(p.context + task.instruction) + 100) * 0.1 + 160
    p.p_success = 0.85
    return p


PROMPT = ("Task: {task}\n\nFiles in the current directory:\n{ctx}\n\n"
          "Write a short Python 3 program (stdlib csv module, no pandas) that reads the file(s) "
          "from the current directory, computes the answer and prints ONLY the final value "
          "(an integer if the result is whole). Reply with one ```python block only.")


def solve_llm(task, llm, plan, strong=False):
    from .. import config
    if plan.notes.get("generic"):
        from .generic import solve_generic
        return solve_generic(task, llm, plan)
    task_text = re.sub(r"\s*Use bash with python3 to compute it\.", "", strip_boilerplate(task.instruction))
    msgs = [{"role": "user", "content": PROMPT.format(task=task_text, ctx=plan.context)}]
    text, _cost, _meta = llm.chat(msgs, model=config.MODEL_STRONG if strong else config.MODEL_FAST,
                                max_tokens=config.MAX_OUT_STRONG if strong else 600)
    rc, out, _err = sandbox.run_python(sandbox.extract_code(text), task.workspace, timeout=20)
    ans = clean_answer(sandbox.last_line(out)) if rc == 0 else None
    return None if is_unsolvable(ans) else ans
