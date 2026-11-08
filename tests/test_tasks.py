"""Тесты загрузки задач, бюджета и записи ответов (требования организаторов)."""
import json

import pytest

from harness.tasks import AnswerWriter, Task, can_write, find_budget, load_tasks


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


def test_write_failure_is_reported_not_raised(tmp_path, capsys):
    (tmp_path / "answer.txt").mkdir()  # запись в каталог - ошибка, как в каталоге без прав
    AnswerWriter().write(make_task(tmp_path, workspace=False), "42")
    assert "cannot write" in capsys.readouterr().err


def test_can_write_detects_writable_directory_and_leaves_no_files(tmp_path):
    assert can_write(make_task(tmp_path, workspace=False))
    assert list(tmp_path.iterdir()) == []


def test_can_write_detects_missing_directory(tmp_path):
    assert not can_write(make_task(tmp_path / "missing", workspace=False))


def test_expected_field_is_optional(tmp_path):
    d = tmp_path / "math_1"
    d.mkdir()
    (d / "task.json").write_text(json.dumps(
        {"task_id": "math_1", "task_type": "math", "instruction": "x", "reward": 3, "check": "exact_int"}))
    task = load_tasks(tmp_path)[0]
    assert (task.task_id, task.reward) == ("math_1", 3)
