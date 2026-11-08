"""Тесты решателя вопросов по CSV: выполнимые задачи и задачи, которые нельзя решить по данным."""
import pytest

from harness.solvers.data_solver import analyze
from harness.tasks import Task

CSV = (
    "order_id,region,product,month,units,price\n"
    "1,north,widget,1,5,10\n"
    "2,south,widget,2,7,20\n"
    "3,north,gadget,6,3,30\n"
)
TAIL = " Use bash with python3 to compute it."
HEAD = "The file sales.csv has columns order_id,region,product,month,units,price (3 rows). "


def make_task(tmp_path, question, csv_text=CSV):
    ws = tmp_path / "workspace"
    ws.mkdir()
    if csv_text is not None:
        (ws / "sales.csv").write_text(csv_text, encoding="utf-8")
    return Task(task_id="t", dir=tmp_path, task_type="data_analysis",
                instruction=HEAD + question + TAIL, reward=1)


@pytest.mark.parametrize("question", [
    "How many units of product 'doohickey' were sold in months 1 through 2 inclusive?",
    "How many units of product 'widget' in region 'atlantis' were sold in months 1 through 2 inclusive?",
    "How many units of product 'widget' were sold in months 13 through 20 inclusive?",
    "How many units of product 'widget' were sold in months 0 through 2 inclusive?",
    "How many units of product 'widget' were sold in months 5 through 2 inclusive?",
    "How many units of product 'widget' were sold in month 13?",
])
def test_infeasible_questions_get_no_answer(tmp_path, question):
    plan = analyze(make_task(tmp_path, question))
    assert plan.feasible is False
    assert plan.fast_answer is None


def test_product_without_sales_in_months_is_zero_not_infeasible(tmp_path):
    q = "How many units of product 'gadget' were sold in months 1 through 2 inclusive?"
    plan = analyze(make_task(tmp_path, q))
    assert plan.feasible is not False
    assert plan.fast_answer == "0"


def test_existing_product_is_summed(tmp_path):
    q = "How many units of product 'widget' were sold in months 1 through 2 inclusive?"
    assert analyze(make_task(tmp_path, q)).fast_answer == "12"


def test_product_match_ignores_case(tmp_path):
    q = "How many units of product 'Widget' were sold in months 1 through 2 inclusive?"
    assert analyze(make_task(tmp_path, q)).fast_answer == "12"


def test_missing_csv_file_is_infeasible(tmp_path):
    q = "How many units of product 'widget' were sold in months 1 through 2 inclusive?"
    assert analyze(make_task(tmp_path, q, csv_text=None)).feasible is False


def test_unparsed_question_about_missing_product_skips_llm(tmp_path):
    q = "What share of orders is product 'doohickey' responsible for?"
    assert analyze(make_task(tmp_path, q)).feasible is False


def test_unparsed_question_about_known_product_goes_to_llm(tmp_path):
    q = "What share of orders is product 'widget' responsible for?"
    plan = analyze(make_task(tmp_path, q))
    assert plan.feasible is not False
    assert plan.fast_answer is None
    assert "sales.csv" in plan.context


WIDER = (
    "order_id,region,product,month,units,price\n"
    "1,north,widget,1,5,10\n"
    "2,south,widget,2,7,20\n"
    "3,north,gadget,6,3,30\n"
    "4,north,widget,3,2,50\n"
    "5,east,gizmo,2,1,99\n"
)


@pytest.mark.parametrize(("question", "answer"), [
    # формулировки из примеров организаторов
    ("How many units of product 'widget' were sold in months 1 through 2 inclusive?", "12"),
    # новые формулировки
    ("What is the total revenue (units times price) of product 'widget' in region 'north' for months 1 to 3?", "150"),
    ("What is the total revenue (units * price) of product 'widget' in months 1 through 3 inclusive?", "290"),
    ("What is the total revenue of product 'gadget'?", "90"),
    ("How many orders for product 'widget' were placed in region 'north' in months 1 through 3 inclusive?", "2"),
    ("How many orders were placed in month 2?", "2"),
    ("What is the total number of units of product 'widget' sold between months 1 and 3?", "14"),
    ("How many units of product 'widget' were sold in the 'north' region from month 1 to month 3?", "7"),
    ("How many units were sold in month 2?", "8"),
    ("How many units of product 'widget' were sold during month 2?", "7"),
    ("Which product sold the most units in months 1 through 3 inclusive?", "widget"),
    ("Which region sold the most units?", "north"),
    ("Which product generated the most revenue in month 2?", "widget"),
])
def test_supported_phrasings_are_solved_without_llm(tmp_path, question, answer):
    plan = analyze(make_task(tmp_path, question, csv_text=WIDER))
    assert plan.fast_answer == answer


@pytest.mark.parametrize("question", [
    "What is the average price of product 'widget'?",
    "How many units of product 'widget' were returned in month 2?",
    "What is the total revenue of product 'widget' excluding region 'north'?",
    "How many units of product 'widget' were sold in the second quarter?",
])
def test_unknown_phrasing_goes_to_llm_instead_of_guessing(tmp_path, question):
    plan = analyze(make_task(tmp_path, question, csv_text=WIDER))
    assert plan.fast_answer is None
    assert plan.feasible is not False


def test_tie_for_top_product_goes_to_llm(tmp_path):
    csv_text = "order_id,region,product,month,units,price\n1,north,a,1,5,10\n2,north,b,1,5,10\n"
    plan = analyze(make_task(tmp_path, "Which product sold the most units?", csv_text=csv_text))
    assert plan.fast_answer is None


def test_top_with_no_sales_in_range_goes_to_llm(tmp_path):
    plan = analyze(make_task(tmp_path, "Which product sold the most units in month 12?", csv_text=WIDER))
    assert plan.fast_answer is None
