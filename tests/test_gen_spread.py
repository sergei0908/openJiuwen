"""Генератор: --types и --spread (разброс сложности для обучения оценщика)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def gen(out, *args):
    subprocess.run([sys.executable, str(ROOT / "tools" / "gen_tasks.py"), "--out", str(out), *args],
                   check=True, capture_output=True)
    return [json.loads((d / "task.json").read_text()) for d in sorted(out.iterdir())]


def test_types_filter(tmp_path):
    tasks = gen(tmp_path / "t", "--n", "20", "--types", "math,search")
    assert {t["task_type"] for t in tasks} == {"math", "search"}


def test_spread_zero_keeps_old_sets(tmp_path):
    a = gen(tmp_path / "a", "--n", "30", "--seed", "3")
    b = gen(tmp_path / "b", "--n", "30", "--seed", "3", "--spread", "0")
    assert a == b


def test_spread_widens_sizes(tmp_path):
    out = tmp_path / "s"
    gen(out, "--n", "120", "--seed", "1", "--spread", "1", "--types", "data_analysis,coding", "--infeasible", "0")
    rows, funcs = [], []
    for d in out.iterdir():
        ws = d / "workspace"
        for f in ws.glob("*.csv"):
            rows.append(sum(1 for _ in f.open()) - 1)
        if (ws / "mylib.py").exists():
            funcs.append((ws / "mylib.py").read_text().count("def "))
    assert max(rows) >= 1000 and min(rows) <= 50
    assert max(funcs) >= 9 and min(funcs) <= 4


def test_new_tables_have_correct_answers(tmp_path):
    out = tmp_path / "e"
    gen(out, "--n", "150", "--seed", "4", "--spread", "1", "--types", "data_analysis", "--infeasible", "0")
    checked = 0
    for d in out.iterdir():
        j = json.loads((d / "task.json").read_text())
        emp = d / "workspace" / "employees.csv"
        if not emp.exists() or "total salary" not in j["instruction"]:
            continue
        dep = j["instruction"].split("'")[1]
        with emp.open() as f:
            exp = sum(int(r["salary"]) for r in csv.DictReader(f) if r["department"] == dep)
        assert j["expected"] == str(exp)
        checked += 1
    assert checked > 0
