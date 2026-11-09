"""Очистка трасс: в обучающих данных только числа, без ответов модели."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from strip_traces import clean, main


def test_answers_and_error_texts_are_removed():
    rec = {"task_id": "math_1", "type": "math", "reward": 2, "route": "llm", "cost": 80.5, "ok": True,
           "answer": "42", "err": "RuntimeError: LLM call failed: some proxy text",
           "calls": [{"model": "m", "prompt_tokens": 300, "completion_tokens": 50, "cost": 80, "text": "x"}],
           "features": {"instr_chars": 400}}
    out = clean(rec, group="seed_1")
    assert "answer" not in out
    assert out["err"] == "other"
    assert out["calls"] == [{"model": "m", "prompt_tokens": 300, "completion_tokens": 50, "cost": 80}]
    assert out["group"] == "seed_1" and out["features"] == {"instr_chars": 400}


def test_correct_is_checked_against_expected(tmp_path):
    d = tmp_path / "tasks" / "math_1"
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"task_id": "math_1", "expected": "42", "check": "exact_int"}))
    (d / "answer.txt").write_text("41\n")
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"task_id": "math_1", "type": "math", "route": "llm", "cost": 60, "ok": True}) + "\n")
    out = tmp_path / "clean.jsonl"
    assert main([str(raw), "--tasks-dir", str(tmp_path / "tasks"), "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["ok"] is True and rec["correct"] is False
