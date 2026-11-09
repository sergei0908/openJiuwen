"""Признаки задач для оценщика расхода: считаются без LLM и только числами."""
import shutil
from pathlib import Path

import pytest

from harness import features, solvers
from harness.tasks import load_tasks

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tasks(tmp_path):
    shutil.copytree(ROOT / "examples" / "tasks", tmp_path / "tasks")
    return {t.task_id: t for t in load_tasks(tmp_path / "tasks")}


def _feats(task):
    plan = solvers.pick(task).analyze(task, fast_path=False)
    return features.extract(task, plan)


def test_all_features_are_numbers(tasks):
    for task in tasks.values():
        f = _feats(task)
        assert f["instr_chars"] == len(task.instruction)
        assert all(isinstance(v, (int, float)) for v in f.values()), task.task_id


def test_type_specific_features(tasks):
    by_type = {}
    for task in tasks.values():
        by_type.setdefault(task.task_type, _feats(task))
    assert by_type["math"]["math_numbers"] >= 2
    assert by_type["search"]["kb_files"] >= 2
    assert by_type["search"]["search_hops"] >= 1
    d = by_type["data_analysis"]
    assert d["data_files"] == 1 and d["data_rows"] > 0 and d["data_cols"] >= 2
    c = by_type["coding"]
    assert c["coding_targets"] == 1 and c["coding_tests"] >= 1 and c["coding_funcs"] >= 1
    assert c["coding_failing"] >= 1


def test_broken_task_gives_partial_features(tmp_path):
    from harness.solvers.base import Plan
    from harness.tasks import Task

    task = Task("coding_x", tmp_path / "missing", "coding", "Fix it.", 1.0)
    f = features.extract(task, Plan())
    assert f["instr_chars"] == len("Fix it.")
    assert "coding_targets" in f or f.keys() >= {"instr_chars", "context_chars"}
