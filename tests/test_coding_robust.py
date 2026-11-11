"""Находки Сергея 11.11: объекты с состоянием - не «противоречивые тесты»; команда тестов всегда наша."""
import subprocess
import sys

from harness.solvers import coding_solver
from harness.solvers.coding_solver import contradictory_tests, run_tests
from harness.tasks import Task

STACK = '''class Stack:
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def pop(self):
        return self.items.pop(0)
'''

STACK_TEST = '''import unittest

import mylib as m


class T(unittest.TestCase):
    def test_pop_order(self):
        s = m.Stack()
        s.push(1)
        s.push(2)
        self.assertEqual(s.pop(), 2)
        self.assertEqual(s.pop(), 1)


if __name__ == "__main__":
    unittest.main()
'''


def task_with(tmp_path, src, test, instruction="Fix mylib.py so that the tests pass."):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "mylib.py").write_text(src)
    (ws / "test_mylib.py").write_text(test)
    return Task(task_id="t", dir=tmp_path, task_type="coding", instruction=instruction, reward=1), ws


def test_stateful_object_is_not_contradictory(tmp_path):
    task, ws = task_with(tmp_path, STACK, STACK_TEST)
    assert not contradictory_tests([ws / "test_mylib.py"])
    assert coding_solver.analyze(task, fast_path=False).feasible is not False


def test_counter_method_is_not_contradictory(tmp_path):
    t = tmp_path / "test_c.py"
    t.write_text("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
                 "    def test(self):\n        c = m.Counter()\n"
                 "        self.assertEqual(c.next(), 1)\n        self.assertEqual(c.next(), 2)\n")
    assert not contradictory_tests([t])


def test_real_contradiction_still_found(tmp_path):
    t = tmp_path / "test_x.py"
    t.write_text("import unittest\nimport mylib as m\nfrom mylib import total\n\nclass T(unittest.TestCase):\n"
                 "    def test(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n"
                 "        self.assertEqual(m.total([1, 2, 3]), 7)\n")
    assert contradictory_tests([t])
    t.write_text("import unittest\nfrom mylib import total\n\nclass T(unittest.TestCase):\n"
                 "    def test(self):\n        self.assertEqual(total([1]), 1)\n        self.assertEqual(total([1]), 2)\n")
    assert contradictory_tests([t])


def test_call_with_variables_is_not_compared(tmp_path):
    t = tmp_path / "test_v.py"
    t.write_text("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
                 "    def test(self):\n        xs = [1]\n        self.assertEqual(m.total(xs), 1)\n"
                 "        xs.append(2)\n        self.assertEqual(m.total(xs), 3)\n")
    assert not contradictory_tests([t])


def test_command_from_instruction_is_ignored(tmp_path):
    # две команды в условии: раньше запускалась только первая, и задача «решалась» без починки
    instr = "Run `python3 test_other.py` and `python3 test_mylib.py`; fix mylib.py."
    task, ws = task_with(tmp_path, "def f():\n    return 1\n",
                         "import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
                         "    def test(self):\n        self.assertEqual(m.f(), 2)\n", instr)
    (ws / "test_other.py").write_text("import unittest\n\nclass T(unittest.TestCase):\n"
                                      "    def test(self):\n        pass\n")
    cmd = coding_solver.test_cmd(task, sorted(ws.glob("test*.py")))
    assert "test_mylib" in cmd and "test_other" in cmd
    ok, _ = run_tests(task, sorted(ws.glob("test*.py")))
    assert not ok


def test_pytest_in_instruction_does_not_break_runs(tmp_path):
    task, ws = task_with(tmp_path, "def f():\n    return 2\n",
                         "import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
                         "    def test(self):\n        self.assertEqual(m.f(), 2)\n",
                         "Run `pytest -q` in the current directory and fix mylib.py.")
    assert "pytest" not in coding_solver.test_cmd(task, [ws / "test_mylib.py"])
    ok, out = run_tests(task, [ws / "test_mylib.py"])
    assert ok, out


def test_stack_fixed_passes_with_our_command(tmp_path):
    task, ws = task_with(tmp_path, STACK.replace("pop(0)", "pop()"), STACK_TEST)
    ok, out = run_tests(task, [ws / "test_mylib.py"])
    assert ok, out
    r = subprocess.run([sys.executable, "test_mylib.py"], check=False, cwd=ws, capture_output=True, text=True)
    assert r.returncode == 0
