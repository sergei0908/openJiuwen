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
REVENUE = r"(?:\(units ?(?:\*|x|times|multiplied by) ?price\) )?"
METRICS = [
    (r"How many units of", "units"),
    (r"How many units in total of", "units"),
    (r"What is the total number of units of", "units"),
    (r"What is the total number of units", "units"),
    (r"How many units", "units"),
    (rf"What is the total revenue {REVENUE}(?:of|for)", "revenue"),
    (rf"What is the total revenue {REVENUE}", "revenue"),
    (r"How many orders (?:of|for)", "orders"),
    (r"How many orders", "orders"),
    (r"Which product sold the most units", "top_product_units"),
    (r"Which region sold the most units", "top_region_units"),
    (r"Which product (?:generated|had) the (?:most|highest) (?:total )?revenue", "top_product_revenue"),
    (r"Which region (?:generated|had) the (?:most|highest) (?:total )?revenue", "top_region_revenue"),
]
# Необязательные слова вокруг условий: на результат они не влияют.
NOISE = r"\s+(?:(?:were|was) (?:sold|placed)|sold|placed)\b"
MONTHS = r"(?:in|for|during) months? (\d+) (?:through|to|-) (?:month )?(\d+)(?: inclusive)?"
FILTERS = [
    (rf"(?:for |of )?product {Q}", "product"),
    (rf"(?:in|for)(?: the)? region {Q}", "region"),
    (rf"(?:in|for) the {Q} region", "region"),
    (MONTHS, "months"),
    (r"from month (\d+) to month (\d+)(?: inclusive)?", "months"),
    (r"between months (\d+) and (\d+)(?: inclusive)?", "months"),
    (r"(?:in|for|during) month (\d+)", "month"),
]


def parse_question(q):
    q = q.strip()
    for mpat, metric in METRICS:
        m = re.match(mpat + r"(?:\s+|(?=\?))", q, flags=re.IGNORECASE)
        if not m:
            continue
        rest = " " + q[m.end():]
        rest = re.sub(NOISE, "", rest, flags=re.IGNORECASE)
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


def _same(a, b):
    return a.strip().casefold() == b.strip().casefold()


def data_exists(rows, filt):
    """False, если на вопрос нельзя точно ответить по этим данным (задача невыполнима).

    Товара или региона нет в файле, месяцы вне 1-12, начало диапазона больше конца.
    Товар есть, но продаж в этих месяцах нет - это не сюда: верный ответ 0.
    """
    for col in ("product", "region"):
        if col in filt and not any(_same(r.get(col) or "", filt[col][0]) for r in rows):
            return False
    if "months" in filt:
        a, b = map(int, filt["months"])
        if not 1 <= a <= b <= 12:
            return False
    return "month" not in filt or 1 <= int(filt["month"][0]) <= 12


def named_values(q):
    """Товар и регион, названные в вопросе, даже если весь вопрос разобрать не удалось."""
    filt = {}
    for key, pat in (("product", rf"product {Q}"), ("region", rf"region {Q}")):
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            filt[key] = m.groups()
    return filt


def compute(rows, metric, filt):
    if metric.startswith("top_"):
        return compute_top(rows, metric, filt)
    total = 0
    for r in rows:
        try:
            if "product" in filt and not _same(r["product"], filt["product"][0]):
                continue
            if "region" in filt and not _same(r["region"], filt["region"][0]):
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


def compute_top(rows, metric, filt):
    """Какой товар или регион лидирует. При ничьей точного ответа нет: решает LLM, а не угадываем."""
    _, col, measure = metric.split("_")
    groups = {}
    for r in rows:
        key = (r.get(col) or "").strip()
        value = compute([r], measure, filt)
        if value is None or not key:
            return None
        groups[key] = groups.get(key, 0) + float(value)
    if not groups:
        return None
    best = max(groups.values())
    leaders = [k for k, v in groups.items() if v == best]
    return leaders[0] if len(leaders) == 1 and best > 0 else None


def csv_files(ws):
    return sorted(p for p in ws.glob("*.csv") if p.is_file())


def read_rows(path):
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def analyze(task, fast_path=True):
    p = Plan()
    files = csv_files(task.workspace)
    q = extract_question(task.instruction)
    p.notes["question"] = q
    # Файл, о котором спрашивают, отсутствует - данных для ответа нет.
    have = {f.name for f in files}
    if any(name not in have for name in re.findall(r"[\w.-]+\.csv", task.instruction)):
        p.feasible = False
        return p
    if len(files) == 1:
        rows = read_rows(files[0])
        parsed = parse_question(q)
        if rows is not None and not data_exists(rows, parsed[1] if parsed else named_values(q)):
            p.feasible = False
            return p
        if fast_path and rows is not None and parsed and re.search(r"sales\.csv", task.instruction):
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
          "(an integer if the result is whole). If the answer cannot be determined exactly from the "
          "data, print IMPOSSIBLE. Reply with one ```python block only.")


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
