"""Перебор заподозрил подгонку - модель получает предупреждение про эту функцию (seed 501, coding_0090)."""
from harness.solvers.base import Plan
from harness.solvers.coding_solver import suspect_hint


def test_no_suspect_no_hint():
    assert suspect_hint(Plan()) == ""


def test_hint_names_the_function():
    p = Plan()
    p.notes["apr_suspect"] = {"mylib.py": "average: identity Add 0"}
    h = suspect_hint(p)
    assert "`average`" in h and "it most likely has another bug" in h and "all valid inputs" in h


def test_several_functions():
    p = Plan()
    p.notes["apr_suspect"] = {"a.py": "median: constant index [0] -> [1]", "b.py": "average: identity Add 0"}
    h = suspect_hint(p)
    assert "`average`, `median`" in h and "they most likely have" in h
