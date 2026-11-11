"""Аналитика по CSV: строгий разбор вопроса (0 токенов) или LLM пишет программу.

В промпт идёт только заголовок и 3 строки файла, не весь CSV - так стоимость не растёт
с размером данных.
"""
import csv
import re

from .base import (
    FINAL_RULES,
    PRINT_FINAL,
    Plan,
    agent_answer,
    approx_tokens,
    extract_question,
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
    cols = {c.strip().lower() for c in (rows[0].keys() if rows else ()) if c}
    for col in ("product", "region"):
        # столбца нет совсем - скорее мы не так прочли файл (другой разделитель, другая таблица),
        # чем данных нет: решать «невыполнимо» по этому нельзя
        if col in filt and col in cols and not any(_same(r.get(col) or "", filt[col][0]) for r in rows):
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
    if metric == "orders":
        ids = [(r.get("order_id") or "").strip() for r in rows]
        if any(ids) and len(set(ids)) != len(ids):
            # один заказ в нескольких строках: «сколько заказов» - это не число строк, решает агент
            return None
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
    """CSV задачи, в том числе в подпапках (data/sales.csv); скрытые и служебные папки пропускаем."""
    return sorted(p for p in ws.rglob("*.csv") if p.is_file()
                  and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(ws).parts))


def read_rows(path):
    """Строки CSV; разделитель определяется по первой строке («,», «;», табуляция, «|»)."""
    try:
        with path.open(newline="", encoding="utf-8") as f:
            head = f.readline()
            f.seek(0)
            counts = {d: head.count(d) for d in ",;\t|"}
            delim = max(counts, key=counts.get) if any(counts.values()) else ","
            # «Product», «PRODUCT», « product» (пробел после запятой) - тот же столбец product; без этого
            # фильтр по товару не находил столбца и задача объявлялась невыполнимой
            return [{(k or "").strip().lower(): v.strip() if isinstance(v, str) else v for k, v in r.items()}
                    for r in csv.DictReader(f, delimiter=delim)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def analyze(task, fast_path=True):
    p = Plan()
    files = csv_files(task.workspace)
    q = extract_question(task.instruction)
    p.notes["question"] = q
    # Файл, о котором спрашивают, отсутствует - данных для ответа нет (имя или путь от папки задачи).
    have = {f.name for f in files} | {f.relative_to(task.workspace).as_posix() for f in files}
    named = re.findall(r"[\w./-]+\.csv", task.instruction)
    if any(n.lstrip("./") not in have and n.rsplit("/", 1)[-1] not in have for n in named):
        p.feasible = False
        return p
    if len(files) == 1:
        rows = read_rows(files[0])
        parsed = parse_question(q)
        if rows is not None and not data_exists(rows, parsed[1] if parsed else named_values(q)):
            p.feasible = False
            return p
        if fast_path and rows is not None and parsed and files[0].name == "sales.csv":
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
        name = f.relative_to(task.workspace).as_posix()
        preview.append(f"--- {name} ({len(lines) - 1} data rows), first lines:\n" + "\n".join(lines[:4]))
    p.context = "\n".join(preview)
    p.est_cost = (approx_tokens(p.context + task.instruction) + 100) * 0.1 + 160
    p.p_success = 0.85
    return p


SYSTEM = ("You answer a question about CSV data in the task directory. Read and compute with the "
          "python tool (csv module); do not print whole files. An integer result must be printed "
          "without decimals. " + PRINT_FINAL + FINAL_RULES)


def solve_llm(task, llm, plan, strong=False):
    from ..oj_agent import final_stop, make_python_tool
    if plan.notes.get("generic"):
        from .generic import solve_generic
        return solve_generic(task, llm, plan)
    task_text = re.sub(r"\s*Use bash with python3 to compute it\.", "", strip_boilerplate(task.instruction))
    query = f"Task: {task_text}\n\nFiles in the task directory:\n{plan.context}"
    state = {}
    return agent_answer(llm, system_prompt=SYSTEM, query=query,
                        tools=[make_python_tool(task.workspace, state)], strong=strong,
                        stop=final_stop(state))
