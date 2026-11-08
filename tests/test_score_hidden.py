"""score.py должен засчитывать coding-задачу только при прохождении скрытых тестов."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import score

TEST = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n    def test(self):\n"
        "        self.assertEqual(m.f(1), 1)\n\nif __name__ == '__main__':\n    unittest.main()\n")
HIDDEN = TEST.replace("m.f(1), 1", "m.f(2), 2")


def make(tmp_path, src):
    d = tmp_path / "coding_1"
    (d / "workspace").mkdir(parents=True)
    (d / "workspace" / "mylib.py").write_text(src)
    (d / "workspace" / "test_mylib.py").write_text(TEST)
    (d / "hidden_test.py").write_text(HIDDEN)
    (d / "answer.txt").write_text("DONE\n")
    (d / "task.json").write_text(json.dumps({"task_id": "coding_1", "task_type": "coding", "reward": 2,
                                             "expected": "tests_pass", "check": "unittest"}))


def test_real_fix_passes(tmp_path, capsys):
    make(tmp_path, "def f(x):\n    return x\n")
    assert score.main([str(tmp_path), "--strict"]) == 0
    assert "0 OVERFIT" in capsys.readouterr().out


def test_overfit_fix_fails(tmp_path, capsys):
    make(tmp_path, "def f(x):\n    return 1\n")
    assert score.main([str(tmp_path), "--strict"]) == 1
    assert "1 OVERFIT repairs: coding_1" in capsys.readouterr().out
