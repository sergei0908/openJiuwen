"""Признаки задачи для оценщика расхода (harness/estimator.py, tools/fit_cost_model.py).

Считаются после фазы A только по файлам задачи и плану решателя, без вызова модели, и пишутся в
трассу. Все значения - числа. Если признак посчитать нельзя, его просто нет в словаре.

Общие: instr_chars, context_chars, generic (задача пошла универсальным путём), heur_cost
(эвристический прогноз решателя - тоже полезный признак).
"""
import re
from pathlib import Path

NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
FAIL_RE = re.compile(r"^(?:FAIL|ERROR):", re.MULTILINE)
DEF_RE = re.compile(r"^\s*def \w+", re.MULTILINE)
TEST_RE = re.compile(r"^\s*def test\w*", re.MULTILINE)
AGG_WORDS = (
    (0, ("total", "sum", "how many units", "revenue")),
    (1, ("how many orders", "how many rows", "number of", "count")),
    (2, ("average", "mean")),
    (3, ("maximum", "minimum", "highest", "lowest", "largest", "smallest", "most", "least")),
)


def _read(p: Path, limit=2_000_000):
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _common(task, plan):
    f = {"instr_chars": len(task.instruction or ""), "context_chars": len(plan.context or ""),
         "generic": int(bool(plan.notes.get("generic"))), "heur_cost": round(float(plan.est_cost), 1)}
    return f


def _math(task, plan):
    from .solvers.base import strip_boilerplate
    text = strip_boilerplate(task.instruction or "")
    return {"math_numbers": len(NUM_RE.findall(text)),
            "math_sentences": len(re.findall(r"[.!?](?:\s|$)", text)),
            "math_percent": int("%" in text or "percent" in text.lower())}


def _search(task, plan):
    kb = task.workspace / "kb"
    files = sorted(kb.rglob("*.md")) if kb.is_dir() else []
    q = (plan.notes.get("question") or task.instruction or "")
    # длина цепочки: «employer of the mentor of X» / «X's mentor's employer»
    hops = len(re.findall(r"'s\b", q)) + len(re.findall(r"\bof (?:the |his |her |their )?\w+", q))
    return {"kb_files": len(files), "kb_chars": sum(len(_read(p)) for p in files),
            "search_hops": max(1, hops)}


def _data(task, plan):
    files = sorted(p for p in task.workspace.glob("*.csv") if p.is_file())
    rows, cols, size = 0, 0, 0
    for p in files:
        text = _read(p)
        size += p.stat().st_size if p.exists() else 0
        lines = [x for x in text.splitlines() if x.strip()]
        if lines:
            rows += len(lines) - 1
            cols = max(cols, lines[0].count(",") + 1)
    q = (plan.notes.get("question") or task.instruction or "").lower()
    agg = 4
    for code, words in AGG_WORDS:
        if any(w in q for w in words):
            agg = code
            break
    conditions = len(re.findall(r"'[^']+'|\"[^\"]+\"", q)) + len(re.findall(r"\bmonths?\b|\bbetween\b", q))
    return {"data_files": len(files), "data_rows": rows, "data_cols": cols, "data_bytes": size,
            "data_agg": agg, "data_conditions": conditions}


def _coding(task, plan):
    from .solvers.coding_solver import find_files
    targets, tests = find_files(task.workspace)
    src = [_read(p) for p in targets]
    test_src = [_read(p) for p in tests]
    fail_out = plan.notes.get("fail_out") or ""
    return {"coding_targets": len(targets),
            "coding_src_lines": sum(s.count("\n") + 1 for s in src if s),
            "coding_funcs": sum(len(DEF_RE.findall(s)) for s in src),
            "coding_tests": sum(len(TEST_RE.findall(s)) for s in test_src),
            "coding_failing": len(FAIL_RE.findall(fail_out)),
            "coding_fail_chars": len(fail_out),
            "coding_apr_partial": int(bool(plan.notes.get("partial")))}


BY_TYPE = {"math": _math, "search": _search, "data_analysis": _data, "coding": _coding}


def extract(task, plan):
    """Словарь признаков задачи. Никогда не бросает исключений: оценщик должен работать всегда."""
    out = {}
    for fn in (_common, BY_TYPE.get(task.task_type)):
        if fn is None:
            continue
        try:
            out.update(fn(task, plan))
        except Exception:  # noqa: BLE001, S112 - признак не посчитался, обходимся без него
            continue
    return out
