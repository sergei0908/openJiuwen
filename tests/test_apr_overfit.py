"""Защита от подгонки починки перебором под видимые тесты (coding_solver.suspicious_patch)."""
from harness.solvers.coding_solver import suspicious_patch

AVG_BUG = "def average(xs):\n    return sum(xs) // len(xs) + 1\n"
MED_BUG = "def median(xs):\n    ys = sorted(xs)\n    return ys[0]\n"


def test_neutralized_constant_is_suspicious():
    fixed = "def average(xs):\n    return sum(xs) // len(xs) + 0\n"
    assert "identity" in suspicious_patch(AVG_BUG, fixed)


def test_guessed_constant_index_is_suspicious():
    fixed = "def median(xs):\n    ys = sorted(xs)\n    return ys[1]\n"
    assert "constant index" in suspicious_patch(MED_BUG, fixed)


def test_ordinary_fixes_are_trusted():
    assert suspicious_patch("def f(a, b):\n    return a - b\n", "def f(a, b):\n    return a + b\n") is None
    assert suspicious_patch("def u(xs):\n    return sorted(xs)\n", "def u(xs):\n    return sorted(set(xs))\n") is None
    assert suspicious_patch("def t(xs, n):\n    return xs[n:]\n", "def t(xs, n):\n    return xs[:n]\n") is None
    assert suspicious_patch(MED_BUG, "def median(xs):\n    ys = sorted(xs)\n    return ys[len(ys) // 2]\n") is None


def test_only_changed_functions_are_checked():
    other = "def g(x):\n    return x + 0\n"
    orig = other + "def f(a, b):\n    return a - b\n"
    fixed = other + "def f(a, b):\n    return a + b\n"
    assert suspicious_patch(orig, fixed) is None


def test_unparsable_source_is_not_suspicious():
    assert suspicious_patch("def f(:\n", "def f(:\n") is None
