"""Тесты LLM-пути для coding: вклейка функций, сокращённый контекст, попытки."""
import asyncio

import pytest

from harness import config
from harness.solvers.base import Plan
from harness.solvers.coding_solver import (
    compact_test_output,
    failing_context,
    merge_reply,
    run_tests,
    solve_llm,
)
from harness.tasks import Task

SRC = "import math\n\nLIMIT = 3\n\n\ndef total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n\n\ndef helper(x):\n    return x + 1\n\n\ndef uses_helper(x):\n    return helper(x) * 2\n"
TESTS = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
         "    def test_total(self):\n        self.assertEqual(m.total([1, 2]), 3)\n\n"
         "    def test_uses_helper(self):\n        self.assertEqual(m.uses_helper(1), 4)\n\n\n"
         "if __name__ == '__main__':\n    unittest.main()\n")
FIX = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"


def make_task(tmp_path, src=SRC, tests=TESTS, reward=1):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "mylib.py").write_text(src, encoding="utf-8")
    (ws / "test_mylib.py").write_text(tests, encoding="utf-8")
    return Task(task_id="t", dir=tmp_path, task_type="coding", reward=reward,
                instruction="Run `python3 test_mylib.py` in the current directory; some tests fail.")


class FakeLLM:
    """Вместо агента openJiuwen: каждый запуск агента вызывает update_code с очередным ответом
    (None - модель ничего не прислала) и останавливается, как настоящий, по stop()."""

    def __init__(self, replies):
        self.replies, self.prompts, self.models, self.results = list(replies), [], [], []
        self.ledger = self

    def can_afford(self, _):
        return True

    def agent(self, *, system_prompt, query, tools, model=None, max_iterations=4, max_tokens=800, stop=None):
        self.prompts.append(system_prompt + "\n" + query)
        self.models.append(model)
        code = self.replies.pop(0)
        if code is None:
            return "no code"
        update = next(t for t in tools if t.card.name == "update_code")
        self.results.append(asyncio.run(update.invoke({"code": code})))
        return "FINAL: DONE" if stop and stop() else "FINAL: IMPOSSIBLE"


def test_merge_replaces_only_named_function():
    merged = merge_reply(SRC, FIX)
    assert "s += x" in merged
    assert "s -= x" not in merged
    assert "def helper(x):" in merged
    assert "LIMIT = 3" in merged
    assert merged.index("def total") < merged.index("def helper")


def test_merge_accepts_full_file_reply():
    merged = merge_reply(SRC, SRC.replace("s -= x", "s += x"))
    assert merged.count("def total") == 1
    assert "s += x" in merged


def test_merge_appends_new_function_and_import():
    merged = merge_reply(SRC, "import os\n\ndef extra():\n    return os.sep\n")
    assert "def extra():" in merged
    assert merged.startswith("import os")


def test_merge_replaces_constant():
    merged = merge_reply(SRC, "LIMIT = 4\n")
    assert "LIMIT = 4" in merged
    assert "LIMIT = 3" not in merged


def test_merge_rejects_syntax_error_and_empty_reply():
    assert merge_reply(SRC, "def broken(:\n") is None
    assert merge_reply(SRC, "x + 1\n") is None


def test_context_has_only_failing_test_and_its_functions(tmp_path):
    task = make_task(tmp_path)
    _, out = run_tests(task, [task.workspace / "test_mylib.py"])
    ctx = failing_context([task.workspace / "mylib.py"], [task.workspace / "test_mylib.py"], out)
    assert "def total" in ctx
    assert "def test_total" in ctx
    assert "def helper" not in ctx
    assert "test_uses_helper" not in ctx


def test_context_follows_calls_inside_library(tmp_path):
    task = make_task(tmp_path, src=SRC.replace("x + 1", "x + 9"), tests=TESTS.replace("m.total([1, 2]), 3", "m.total([1, 2]), -3"))
    _, out = run_tests(task, [task.workspace / "test_mylib.py"])
    ctx = failing_context([task.workspace / "mylib.py"], [task.workspace / "test_mylib.py"], out)
    assert "def uses_helper" in ctx
    assert "def helper" in ctx
    assert "def total" not in ctx


def test_context_is_none_when_nothing_failed_is_named(tmp_path):
    task = make_task(tmp_path)
    assert failing_context([task.workspace / "mylib.py"], [task.workspace / "test_mylib.py"], "boom") is None


def test_compact_output_keeps_failures_and_drops_paths(tmp_path):
    task = make_task(tmp_path)
    _, out = run_tests(task, [task.workspace / "test_mylib.py"])
    short = compact_test_output(out)
    assert "FAIL: test_total" in short
    assert "AssertionError: -3 != 3" in short
    assert "FAILED (failures=1)" in short
    assert str(tmp_path) not in short
    assert "Traceback" not in short
    assert len(short) < len(out)


def test_compact_output_passes_unknown_format_through():
    assert compact_test_output("weird output") == "weird output"


@pytest.fixture(autouse=True)
def _need_openjiuwen():
    pytest.importorskip("openjiuwen")


def test_llm_fix_is_merged_and_verified(tmp_path):
    task = make_task(tmp_path)
    llm = FakeLLM([FIX])
    assert solve_llm(task, llm, Plan()) == "DONE"
    assert len(llm.prompts) == 1
    assert llm.results == ["updated mylib.py; all tests passed"]
    assert "def helper" in (task.workspace / "mylib.py").read_text(encoding="utf-8")
    assert "only the corrected functions" in llm.prompts[0].lower()
    assert "def helper" not in llm.prompts[0]


def test_tool_reports_failing_tests(tmp_path):
    task = make_task(tmp_path)
    llm = FakeLLM(["def total(xs):\n    return 0\n", FIX])
    assert solve_llm(task, llm, Plan()) == "DONE"
    assert llm.results[0].startswith("updated mylib.py; tests failed")
    assert "test_total" in llm.results[0]


def test_second_attempt_sees_full_files(tmp_path):
    task = make_task(tmp_path, reward=1)
    llm = FakeLLM(["def total(xs):\n    return 0\n", FIX])
    assert solve_llm(task, llm, Plan()) == "DONE"
    assert len(llm.prompts) == 2
    assert "def helper" not in llm.prompts[0]
    assert "def helper" in llm.prompts[1]


def test_low_reward_task_does_not_escalate_to_reasoning_model(tmp_path):
    task = make_task(tmp_path, reward=1)
    llm = FakeLLM([None, FIX])
    assert solve_llm(task, llm, Plan()) == "DONE"
    assert llm.models == [config.MODEL_FAST, config.MODEL_FAST]


def test_high_reward_task_escalates_to_strong_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_REASONING", True)
    task = make_task(tmp_path, reward=3)
    llm = FakeLLM([None, FIX])
    assert solve_llm(task, llm, Plan()) == "DONE"
    assert llm.models == [config.MODEL_FAST, config.MODEL_STRONG]


def test_failed_repair_restores_original_file(tmp_path):
    task = make_task(tmp_path)
    bad = "def total(xs):\n    return 0\n"
    assert solve_llm(task, FakeLLM([bad, bad]), Plan()) is None
    assert (task.workspace / "mylib.py").read_text(encoding="utf-8") == SRC


def test_budget_break_restores_original_file(tmp_path):
    from harness.oj_agent import TaskBudgetExceeded

    class Broke(FakeLLM):
        def agent(self, **kw):
            super().agent(**kw)
            raise TaskBudgetExceeded("cap")

    task = make_task(tmp_path)
    with pytest.raises(TaskBudgetExceeded):
        solve_llm(task, Broke(["def total(xs):\n    return 0\n"]), Plan())
    assert (task.workspace / "mylib.py").read_text(encoding="utf-8") == SRC
