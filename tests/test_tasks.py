"""Тесты загрузки задач, бюджета и записи ответов (требования организаторов)."""
import json

import pytest

from harness import runner, tasks
from harness.solvers import coding_solver
from harness.tasks import (
    AnswerWriter,
    Task,
    can_write,
    find_budget,
    load_tasks,
    write_file,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("BUDGET", "TOKEN_BUDGET", "LLM_BUDGET", "OUTPUT_TOKEN_BUDGET", "DEFAULT_BUDGET"):
        monkeypatch.delenv(k, raising=False)


def make_task(tmp_path, workspace=True):
    if workspace:
        (tmp_path / "workspace").mkdir()
    return Task(task_id="t", dir=tmp_path, task_type="math", instruction="", reward=1)


def test_budget_comes_from_budget_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("BUDGET", "12345")
    assert find_budget(tmp_path) == 12345


def test_budget_variable_wins_over_others(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN_BUDGET", "1")
    monkeypatch.setenv("BUDGET", "2")
    assert find_budget(tmp_path) == 2


def test_budget_defaults_to_100k(tmp_path):
    assert find_budget(tmp_path) == 100000


def test_empty_budget_variable_uses_default(tmp_path, monkeypatch):
    monkeypatch.setenv("BUDGET", "")
    assert find_budget(tmp_path) == 100000


def test_answer_is_written_next_to_task_json(tmp_path):
    AnswerWriter().write(make_task(tmp_path), " 42 ")
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "42\n"


def test_output_file_is_updated_after_each_answer(tmp_path):
    out = tmp_path / "out" / "solutions.json"
    writer = AnswerWriter(out)
    writer.write(make_task(tmp_path), "42")
    assert json.loads(out.read_text(encoding="utf-8")) == {"t": "42"}


def blocked_task(tmp_path):
    """Каталог задачи, куда нельзя записать answer.txt: на этом месте лежит каталог."""
    task = make_task(tmp_path, workspace=False)
    (tmp_path / "answer.txt").mkdir()
    return task


def test_write_failure_is_reported_not_raised(tmp_path, capsys):
    assert AnswerWriter().write(blocked_task(tmp_path), "42") is False
    captured = capsys.readouterr()
    assert "cannot write answer for t" in captured.err
    assert "ANSWER t 42" in captured.out


def test_answer_goes_to_out_directory_when_task_dir_is_closed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    assert AnswerWriter(out).write(blocked_task(task_dir), "42") is True
    assert json.loads((out / "solutions.json").read_text(encoding="utf-8")) == {"t": "42"}
    assert (out / "task" / "answer.txt").read_text(encoding="utf-8") == "42\n"


def test_no_extra_files_in_out_when_task_dir_is_open(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    assert AnswerWriter(out).write(make_task(task_dir), "42") is True
    assert [p.name for p in out.iterdir()] == ["solutions.json"]


def test_write_file_replaces_read_only_file_through_directory(tmp_path, monkeypatch):
    target = tmp_path / "mylib.py"
    target.write_text("old", encoding="utf-8")
    real = tasks.Path.write_text

    def closed_file(self, *a, **k):
        if self.name == "mylib.py":
            raise PermissionError("read-only")
        return real(self, *a, **k)
    monkeypatch.setattr(tasks.Path, "write_text", closed_file)
    write_file(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["mylib.py"]


def test_write_file_raises_when_nothing_helps(tmp_path):
    with pytest.raises(OSError, match="."):
        write_file(tmp_path / "missing" / "f.txt", "x")


def test_lost_answer_gives_no_reward_and_is_counted(tmp_path):
    r = runner.Runner(tmp_path)
    r.save(blocked_task(tmp_path), "42")
    assert r.stats["answers_lost"] == 1
    assert r.stats["reward_claimed"] == 0


def test_saved_answer_gives_reward(tmp_path):
    r = runner.Runner(tmp_path)
    r.save(make_task(tmp_path, workspace=False), "42")
    assert r.stats["answers_lost"] == 0
    assert r.stats["reward_claimed"] == 1


def test_coding_task_without_write_permission_spends_no_tokens(tmp_path, monkeypatch, capsys):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "mylib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (ws / "test_mylib.py").write_text(
        "import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n    def test(self):\n"
        "        self.assertEqual(m.f(), 2)\n\nif __name__ == '__main__':\n    unittest.main()\n", encoding="utf-8")
    monkeypatch.setattr(coding_solver, "can_fix", lambda *_: False)
    task = Task(task_id="t", dir=tmp_path, task_type="coding", reward=1,
                instruction="Run `python3 test_mylib.py` in the current directory.")
    plan = coding_solver.analyze(task)
    assert plan.feasible is False
    assert "cannot modify files of t" in capsys.readouterr().err


def test_can_write_detects_writable_directory_and_leaves_no_files(tmp_path):
    assert can_write(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_can_write_detects_missing_directory(tmp_path):
    assert not can_write(tmp_path / "missing")


def test_expected_field_is_optional(tmp_path):
    d = tmp_path / "math_1"
    d.mkdir()
    (d / "task.json").write_text(json.dumps(
        {"task_id": "math_1", "task_type": "math", "instruction": "x", "reward": 3, "check": "exact_int"}))
    task = load_tasks(tmp_path)[0]
    assert (task.task_id, task.reward) == ("math_1", 3)
