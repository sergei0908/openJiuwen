"""Перебор на больших файлах: мутации только в функциях, до которых доходят тесты."""
import ast
import time

from harness.solvers import apr

SRC = '''
def helper_a(x):
    return x + 1


def used(x):
    return inner(x) * 2


def inner(x):
    return x - 1


class Box:
    def size(self):
        return 3 + 4


LIMIT = 10 + 5
'''


def _owners(tree, cands):
    nodes = list(ast.walk(tree))
    names = {}
    for unit in tree.body:
        for n in ast.walk(unit):
            names[id(n)] = getattr(unit, "name", "<module>")
    return {names[id(nodes[m[1]])] for m in cands}


def test_keeps_tested_functions_and_their_callees(tmp_path):
    t = tmp_path / "test_m.py"
    t.write_text("import m\n\nassert m.used(3) == 4\n")
    tree = ast.parse(SRC)
    kept = apr._reachable_only(tree, apr._mutations(tree), [t])
    assert _owners(tree, kept) == {"used", "inner", "<module>"}  # helper_a и Box тестами не задеты


def test_class_referenced_by_tests_is_kept(tmp_path):
    t = tmp_path / "test_m.py"
    t.write_text("from m import Box\n\nassert Box().size() == 7\n")
    tree = ast.parse(SRC)
    assert "Box" in _owners(tree, apr._reachable_only(tree, apr._mutations(tree), [t]))


def test_no_known_names_means_no_filtering(tmp_path):
    t = tmp_path / "test_m.py"
    t.write_text("assert True\n")
    tree = ast.parse(SRC)
    cands = apr._mutations(tree)
    assert apr._reachable_only(tree, cands, [t]) == cands


def test_unreadable_tests_mean_no_filtering(tmp_path):
    tree = ast.parse(SRC)
    cands = apr._mutations(tree)
    assert apr._reachable_only(tree, cands, [tmp_path / "missing.py"]) == cands


def test_large_file_is_repaired_quickly(tmp_path):
    filler = "\n\n".join(f"def helper_{i}(x):\n    y = x * {i}\n    if y > 100:\n        return y - 1\n"
                         f"    return y + 1\n" for i in range(150))
    (tmp_path / "mylib.py").write_text(filler + "\n\n\ndef total(xs):\n    return sum(xs) + 1\n")
    (tmp_path / "test_mylib.py").write_text(
        "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
        "    def test_total(self):\n        self.assertEqual(m.total([1, 2]), 3)\n"
        "        self.assertEqual(m.total([]), 0)\n\n\nif __name__ == '__main__':\n    unittest.main()\n")
    t = time.time()
    fixed, _, info = apr.repair(tmp_path, tmp_path / "mylib.py", [tmp_path / "test_mylib.py"])
    assert fixed is not None and "sum(xs) + 1" not in fixed   # починка найдена (любая из верных)
    assert info["mutants"] < 50          # без отбора - больше тысячи вариантов
    assert time.time() - t < 15


def test_task_code_does_not_see_the_api_key(tmp_path, monkeypatch):
    # перебор запускает код задачи: ключа прокси в его окружении быть не должно (как в песочнице агента)
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    env = apr.driver_env()
    assert "LLM_API_KEY" not in env and env["PYTHONDONTWRITEBYTECODE"] == "1"
    (tmp_path / "mylib.py").write_text("import os\n\n\ndef key():\n    return os.environ.get('LLM_API_KEY', 'none') + 'x'\n")
    (tmp_path / "test_mylib.py").write_text(
        "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
        "    def test_key(self):\n        self.assertEqual(m.key(), 'none')\n\n\n"
        "if __name__ == '__main__':\n    unittest.main()\n")
    fixed, _, _ = apr.repair(tmp_path, tmp_path / "mylib.py", [tmp_path / "test_mylib.py"])
    # если бы код задачи видел ключ, вызов вернул бы 'secret-keyx'; без ключа - 'nonex', и перебор чинит «+ 'x'»
    assert fixed is not None and "secret" not in fixed
