"""Заголовки CSV в другом регистре или с пробелами - тот же столбец, а не «товара нет в файле»."""
import pytest

from harness.solvers.data_solver import analyze, read_rows
from harness.tasks import Task

ROWS = [(1, "north", "gadget", 3, 10, 20), (2, "south", "widget", 4, 5, 30), (3, "north", "gadget", 5, 7, 20),
        (4, "north", "gadget", 9, 100, 20)]
Q = ("The file sales.csv has columns {cols} (4 rows). How many units of product 'gadget' were sold in "
     "months 3 through 5 inclusive? Use bash with python3 to compute it.")


def make_task(tmp_path, header, sep=","):
    ws = tmp_path / "workspace"
    ws.mkdir()
    lines = [sep.join(header)] + [sep.join(map(str, r)) for r in ROWS]
    (ws / "sales.csv").write_text("\n".join(lines) + "\n")
    return Task("data_analysis_x", tmp_path, "data_analysis", Q.format(cols=sep.join(header)), 3.0)


@pytest.mark.parametrize("header,sep", [
    (["order_id", "region", "product", "month", "units", "price"], ","),     # как в примерах
    (["Order_ID", "Region", "Product", "Month", "Units", "Price"], ","),
    (["ORDER_ID", "REGION", "PRODUCT", "MONTH", "UNITS", "PRICE"], ","),
    (["order_id", "region", "product", "month", "units", "price"], ", "),    # пробел после запятой
])
def test_header_case_and_spaces_do_not_matter(tmp_path, header, sep):
    p = analyze(make_task(tmp_path, header, sep))
    assert p.feasible is not False
    assert p.fast_answer == "17"


def test_unknown_product_is_still_infeasible(tmp_path):
    t = make_task(tmp_path, ["Order_ID", "Region", "Product", "Month", "Units", "Price"])
    t.instruction = t.instruction.replace("'gadget'", "'doohickey'")
    assert analyze(t).feasible is False


def test_read_rows_normalizes_keys_and_values(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text(" Name , Score \n Ann , 5 \n")
    assert read_rows(f) == [{"name": "Ann", "score": "5"}]


def test_orders_with_repeated_order_id_go_to_agent(tmp_path):
    # строки заказа: один order_id на несколько строк - «сколько заказов» не равно числу строк (adv_report 14)
    ws = tmp_path / "workspace"
    ws.mkdir()
    rows = [(1, "north", "gadget", 3, 1, 2), (1, "north", "gadget", 3, 2, 2), (2, "north", "gadget", 4, 1, 2)]
    (ws / "sales.csv").write_text("order_id,region,product,month,units,price\n" +
                                  "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    q = "The file sales.csv has columns order_id,region,product,month,units,price (3 rows). How many orders for " \
        "product 'gadget' in region 'north'? Use bash with python3 to compute it."
    p = analyze(Task("data_analysis_x", tmp_path, "data_analysis", q, 3.0))
    assert p.feasible is not False and p.fast_answer is None


def test_orders_with_unique_order_id_are_counted(tmp_path):
    t = make_task(tmp_path, ["order_id", "region", "product", "month", "units", "price"])
    t.instruction = t.instruction.replace("How many units of product 'gadget' were sold in months 3 through 5 "
                                          "inclusive?", "How many orders for product 'gadget'?")
    assert analyze(t).fast_answer == "3"
