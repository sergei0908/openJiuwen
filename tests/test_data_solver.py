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
