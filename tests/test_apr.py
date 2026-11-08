"""Ремонт кода без LLM: каждый вид бага из генератора должен чиниться мутациями."""
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from gen_tasks import FUNCS, HIDDEN

from harness.solvers import apr, coding_solver
from harness.tasks import Task

CASES = [(name, i) for name, (_, bads, _) in sorted(FUNCS.items()) for i in range(len(bads))]
# Видимый тест не различает верное исправление и подгонку, правка объективно неоднозначна:
# average([2, 4, 6]) с целым средним не показывает // вместо /, median из 3 чисел - ys[1] вместо ys[n // 2].
AMBIGUOUS = {("average", 1), ("median", 2)}


def write_task(ws, src, checks):
    (ws / "mylib.py").write_text(src, encoding="utf-8")
    body = "\n".join(f"        {c}" for c in checks)
    (ws / "test_mylib.py").write_text(
        "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
        f"    def test_it(self):\n{body}\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8")


@pytest.mark.parametrize(("name", "bug"), CASES, ids=[f"{n}-{i}" for n, i in CASES])
def test_every_generated_bug_is_repaired(tmp_path, name, bug):
    _, bads, test = FUNCS[name]
    write_task(tmp_path, bads[bug], [test])
    tgt = tmp_path / "mylib.py"
    fixed, _, info = apr.repair(tmp_path, tgt, [tmp_path / "test_mylib.py"])
    assert fixed is not None, info
    tgt.write_text(fixed, encoding="utf-8")
    r = subprocess.run([sys.executable, "test_mylib.py"], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    # починка должна быть настоящей, а не подгонкой под видимые тесты
    check = "; ".join(f"assert {c}" for c in HIDDEN[name])
    r = subprocess.run([sys.executable, "-c", f"import mylib as m; {check}"], cwd=tmp_path, capture_output=True,
                       text=True, check=False)
    if (name, bug) in AMBIGUOUS:
        pytest.xfail("видимый тест неоднозначен")
    assert r.returncode == 0, f"overfit:\n{fixed}\n{r.stderr[-300:]}"


def test_every_function_has_hidden_checks():
    assert set(HIDDEN) == set(FUNCS)


def test_syntax_error_is_reported_not_raised(tmp_path):
    write_task(tmp_path, "def f(:\n", ["self.assertEqual(m.f(), 1)"])
    fixed, _, info = apr.repair(tmp_path, tmp_path / "mylib.py", [tmp_path / "test_mylib.py"])
    assert fixed is None
    assert info["reason"] == "syntax"


def test_repair_respects_time_budget(tmp_path):
    # нечинимая функция с большим числом вариантов не должна занимать больше лимита времени
    src = "".join(f"def f{i}(a, b):\n    return a + b * {i} - (a // {i + 1}) % 7\n\n" for i in range(30))
    write_task(tmp_path, src, ["self.assertEqual(m.f3(1, 2), 'unreachable')"])
    start = time.time()
    fixed, _, _ = apr.repair(tmp_path, tmp_path / "mylib.py", [tmp_path / "test_mylib.py"], time_budget=3)
    assert fixed is None
    assert time.time() - start < 15


def test_contradictory_tests_are_detected_before_repair(tmp_path):
    (tmp_path / "workspace").mkdir()
    ws = tmp_path / "workspace"
    write_task(ws, "def total(xs):\n    return sum(xs)\n",
               ["self.assertEqual(m.total([1, 2, 3]), 6)", "self.assertEqual(m.total([1, 2, 3]), 7)"])
    task = Task(task_id="t", dir=tmp_path, task_type="coding",
                instruction="Run `python3 test_mylib.py` in the current directory.", reward=1)
    plan = coding_solver.analyze(task)
    assert plan.feasible is False
    assert plan.fast_answer is None
    assert "apr" not in plan.notes
