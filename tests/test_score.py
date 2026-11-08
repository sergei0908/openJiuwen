"""Тесты оценки: --strict должен ронять CI при неверном ответе и лишнем answer.txt."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import score


def make_task(root, name, expected="42", answer=None, infeasible=False):
    d = root / name
    d.mkdir()
    j = {"task_id": name, "task_type": "math", "reward": 1, "expected": expected, "check": "exact_int"}
    if infeasible:
        j["_infeasible"] = True
    (d / "task.json").write_text(json.dumps(j))
    if answer is not None:
        (d / "answer.txt").write_text(answer + "\n")


def test_strict_passes_when_all_answers_are_right(tmp_path):
    make_task(tmp_path, "a", answer="42")
    make_task(tmp_path, "b", infeasible=True)
    assert score.main([str(tmp_path), "--strict"]) == 0


def test_strict_fails_on_wrong_answer(tmp_path):
    make_task(tmp_path, "a", answer="41")
    assert score.main([str(tmp_path), "--strict"]) == 1


def test_strict_fails_on_missing_answer(tmp_path):
    make_task(tmp_path, "a")
    assert score.main([str(tmp_path), "--strict"]) == 1


def test_strict_fails_on_answer_for_infeasible_task(tmp_path):
    make_task(tmp_path, "a", answer="42")
    make_task(tmp_path, "b", answer="0", infeasible=True)
    assert score.main([str(tmp_path), "--strict"]) == 1


def test_strict_fails_when_no_tasks_were_found(tmp_path):
    assert score.main([str(tmp_path), "--strict"]) == 1


def test_without_strict_always_succeeds(tmp_path):
    make_task(tmp_path, "a", answer="41")
    assert score.main([str(tmp_path)]) == 0
