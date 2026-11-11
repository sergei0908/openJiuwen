"""Ошибки, найденные на задачах Маши: вопрос с именем файла, дробь в ответе."""
from harness.solvers.base import clean_answer, extract_question


def test_question_with_file_name_is_not_cut():
    q = extract_question("What is the average discount given on widget orders in sales.csv? Use bash.")
    assert q == "What is the average discount given on widget orders in sales.csv?"
    assert extract_question("Rosa works. In which city is that firm based? Work only here.") == \
        "In which city is that firm based?"
    assert extract_question("No question here.") == "No question here."


def test_terminating_fraction_becomes_decimal():
    assert clean_answer("36/5") == "7.2"
    assert clean_answer("-3/4") == "-0.75"
    assert clean_answer("10/2") == "5"
    assert clean_answer("7/3") == "7/3"      # бесконечная дробь - оставляем как есть
    assert clean_answer("12.0") == "12"

