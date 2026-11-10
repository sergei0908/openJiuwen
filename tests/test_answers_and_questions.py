"""Ошибки, найденные на задачах Маши: вопрос с именем файла, дробь в ответе, величина, которой нет в таблице."""
from harness.solvers.base import clean_answer, extract_question
from harness.solvers.data_solver import missing_measure

ROWS = [{"order_id": "1", "region": "east", "product": "widget", "month": "4", "units": "2", "price": "12"}]


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


def test_missing_measure_is_detected():
    assert missing_measure("What is the average discount given on widget orders in sales.csv?", ROWS) == "discount"


def test_existing_and_derived_measures_are_fine():
    for q in ("What is the total revenue of widgets?", "What is the total number of units of widget?",
              "What is the average number of units per order in the 'north' region?",
              "What is the sum of all 'math' scores from 2020 onwards?", "What is the highest price?"):
        assert missing_measure(q, ROWS) is None, q
