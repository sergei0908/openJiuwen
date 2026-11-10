"""Ответ с заданным числом знаков после точки: «Round to 2 decimal places» -> 27.50, а не 27.5."""
from pathlib import Path

import pytest

from harness.tasks import AnswerWriter, Task, match_decimals

ROUND2 = "What is the average number of units per order? Round to 2 decimal places."


@pytest.mark.parametrize("instruction,answer,expected", [
    (ROUND2, "27.5", "27.50"),
    (ROUND2, "27", "27.00"),
    (ROUND2, "27.505", "27.51"),          # лишние знаки - школьное округление
    (ROUND2, "-3.1", "-3.10"),
    ("Give the answer to one decimal place.", "4", "4.0"),
    ("Answer with 3 decimals.", "0.5", "0.500"),
    ("Round to 0 decimal places.", "2.5", "3"),
])
def test_pads_to_requested_places(instruction, answer, expected):
    assert match_decimals(instruction, answer) == expected


@pytest.mark.parametrize("instruction,answer", [
    ("How many boxes are left (rounded down)?", "389"),            # знаки не заданы
    (ROUND2, "Osaka"),                                                # не число
    (ROUND2, "1,250.5"),                                              # не одно число в привычной записи
    (ROUND2, "DONE"),
    ("Round prices to 2 decimal places and the total to 1 decimal place.", "12.5"),  # неоднозначно
    ("", "27.5"),
    ("The price column has 2 decimal places. How many orders are there?", "12"),  # описание данных
    ("Prices are stored with two decimals. What is the total number of units?", "158"),
])
def test_leaves_other_answers_alone(instruction, answer):
    assert match_decimals(instruction, answer) == answer


def test_writer_applies_it(tmp_path):
    t = Task("data_analysis_0001", tmp_path, "data_analysis", ROUND2, 3.0)
    assert AnswerWriter().write(t, "27.5")
    assert Path(tmp_path / "answer.txt").read_text() == "27.50\n"
