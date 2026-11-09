"""Устойчивость к другому оформлению задач (tools/make_variants.py): pytest-тесты, подпапки,
CSV с «;», база знаний не в kb/ или в одном файле. Без LLM, только решатели фазы A."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness import solvers
from harness.solvers.coding_solver import find_files, run_tests
from harness.tasks import load_tasks

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def variants(tmp_path_factory):
    out = tmp_path_factory.mktemp("v") / "tasks"
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_variants.py"), "--out", str(out)],
                   check=True, capture_output=True)
    return {t.task_id: t for t in load_tasks(out)}


def plan(task):
    return solvers.pick(task).analyze(task, fast_path=True)


@pytest.mark.parametrize("tid", ["coding_v_pytest", "coding_v_subdir", "coding_v_rename"])
def test_coding_variants_are_really_fixed(variants, tid):
    task = variants[tid]
    p = plan(task)
    assert p.fast_answer == "DONE"
    _, tests = find_files(task.workspace)
    ok, out = run_tests(task, tests)
    assert ok and "Ran 0 tests" not in out


def test_pytest_style_failure_is_not_reported_as_pass(variants, tmp_path):
    # без исправления: тесты-функции должны падать, а не «проходить», ничего не запустив
    task = variants["coding_v_pytest"]
    src = ROOT / "examples" / "tasks" / "coding_0003" / "workspace" / "mylib.py"
    (task.workspace / "mylib.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    _, tests = find_files(task.workspace)
    ok, out = run_tests(task, tests)
    assert not ok
    assert "FAIL: test_" in out


def test_no_tests_is_not_success(tmp_path):
    from harness.tasks import Task
    ws = tmp_path / "w"
    ws.mkdir()
    (ws / "mylib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (ws / "test_mylib.py").write_text("import unittest\n", encoding="utf-8")   # ни одного теста
    task = Task("coding_x", ws, "coding", "Fix mylib.py.", 1.0)
    ok, _ = run_tests(task, [ws / "test_mylib.py"])
    assert not ok


@pytest.mark.parametrize("tid,answer", [("data_v_semicolon", "158"), ("data_v_subdir", "158"),
                                        ("search_v_docs", "1983"), ("search_v_onefile", "Osaka")])
def test_data_and_search_variants(variants, tid, answer):
    p = plan(variants[tid])
    assert p.feasible is not False
    assert p.fast_answer == answer


def test_unparsed_data_question_goes_to_agent_not_infeasible(variants):
    p = plan(variants["data_v_average"])
    assert p.feasible is not False and p.fast_answer is None
    assert "sales.csv" in p.context


def test_variants_have_expected_answers(variants):
    for t in variants.values():
        j = json.loads((t.dir / "task.json").read_text(encoding="utf-8"))
        assert j["expected"]
