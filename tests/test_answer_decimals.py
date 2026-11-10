"""«Round to 2 decimal places»: лишние знаки округляем, нули в конце не дописываем (25.4, а не 25.40).

Эталон у задач Маши - как round(x, 2) в Python: 25.4. Дописанный ноль (25.40) стоил задачи maria_11."""
from pathlib import Path

import pytest

from harness.tasks import AnswerWriter, Task, match_decimals

ROUND2 = "What is the average number of units per order? Round to 2 decimal places."


@pytest.mark.parametrize("instruction,answer,expected", [
    (ROUND2, "27.505", "27.51"),          # лишние знаки - школьное округление
    (ROUND2, "27.4999", "27.5"),          # без нуля в конце
    (ROUND2, "26.999", "27"),
    (ROUND2, "-3.14159", "-3.14"),
    (ROUND2, "-0.001", "0"),
    ("Give the answer to one decimal place.", "4.25", "4.3"),
    ("Answer with 3 decimals.", "0.12345", "0.123"),
    ("Round to 0 decimal places.", "2.5", "3"),
])
def test_rounds_extra_digits(instruction, answer, expected):
    assert match_decimals(instruction, answer) == expected


@pytest.mark.parametrize("instruction,answer", [
    (ROUND2, "25.4"),                                                 # maria_11: не 25.40
    (ROUND2, "27"),
    (ROUND2, "27.51"),
    ("Use weather.csv and round to one decimal place.", "3.3"),
    ("How many boxes are left (rounded down)?", "389"),            # знаки не заданы
    (ROUND2, "Osaka"),                                                # не число
    (ROUND2, "1,250.5"),                                              # не одно число в привычной записи
    (ROUND2, "DONE"),
    ("Round prices to 2 decimal places and the total to 1 decimal place.", "12.555"),  # неоднозначно
    ("", "27.555"),
    ("The price column has 2 decimal places. How many orders are there?", "12"),  # описание данных
    ("Prices are stored with two decimals. What is the average price?", "4.567"),
])
def test_leaves_other_answers_alone(instruction, answer):
    assert match_decimals(instruction, answer) == answer


def test_writer_applies_it(tmp_path):
    t = Task("data_analysis_0001", tmp_path, "data_analysis", ROUND2, 3.0)
    assert AnswerWriter().write(t, "27.505")
    assert Path(tmp_path / "answer.txt").read_text() == "27.51\n"
    assert AnswerWriter().write(t, "25.4")
    assert Path(tmp_path / "answer.txt").read_text() == "25.4\n"
