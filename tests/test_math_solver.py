"""Тесты бесплатного решателя арифметических задач."""
from harness.solvers.math_solver import fast_solve

TAIL = (" How many boxes does it have at the end? Compute it with the bash tool (python3 -c)."
        " Work only inside the current directory. When you are done, write ONLY the final answer"
        " to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


def test_ships_and_receives():
    text = ("A warehouse starts with 100 boxes. It ships 10 boxes per day for 3 days."
            " It receives 2 deliveries of 15 boxes each." + TAIL)
    assert fast_solve(text) == "100"


def test_split_rounds_down():
    text = ("A warehouse starts with 101 boxes. Then the stock is split equally into 2 warehouses"
            " and it keeps one share (rounded down)." + TAIL)
    assert fast_solve(text) == "50"


def test_order_of_operations():
    # сначала делим, потом удваиваем - порядок предложений важен
    text = ("A warehouse starts with 7 boxes. Then the stock is split equally into 2 warehouses"
            " and it keeps one share (rounded down). Then the stock is doubled by merging with"
            " a sister warehouse." + TAIL)
    assert fast_solve(text) == "6"


def test_unknown_sentence_goes_to_llm():
    # незнакомая формулировка - решатель не угадывает, а возвращает None
    text = "A warehouse starts with 100 boxes. Then a storm destroys 5 boxes." + TAIL
    assert fast_solve(text) is None


def test_division_by_zero_is_impossible():
    text = ("A warehouse starts with 100 boxes. Then the stock is split equally into 0 warehouses"
            " and it keeps one share (rounded down)." + TAIL)
    assert fast_solve(text) == "IMPOSSIBLE"


def test_organizer_example_math_0000():
    # пример организаторов, награда 5
    text = ("A warehouse starts with 1752 boxes. Then the stock is split equally into 2 warehouses"
            " and it keeps one share (rounded down). Then the stock is doubled by merging with a"
            " sister warehouse. Then the stock is split equally into 3 warehouses and it keeps one"
            " share (rounded down). Then the stock is doubled by merging with a sister warehouse."
            " Then the stock is split equally into 3 warehouses and it keeps one share"
            " (rounded down)." + TAIL)
    assert fast_solve(text) == "389"


def test_customer_buys():
    text = ("A warehouse starts with 100 boxes. Then a customer buys 30 boxes from it." + TAIL)
    assert fast_solve(text) == "70"


def test_grows_threefold():
    text = ("A warehouse starts with 40 boxes. Then its stock grows threefold after an acquisition."
            " Then a customer buys 20 boxes from it." + TAIL)
    assert fast_solve(text) == "100"


def test_grows_twofold():
    text = ("A warehouse starts with 25 boxes. Then its stock grows twofold after an acquisition." + TAIL)
    assert fast_solve(text) == "50"


Q_END = " How many boxes does it have at the end?"


def test_percent_with_rounding():
    q = ("A warehouse starts with 1000 boxes. Then 25% of the stock is damaged and thrown away "
         "(rounded down to whole boxes). It ships 10 boxes." + Q_END)
    assert fast_solve(q) == "740"
    q = "A warehouse starts with 999 boxes. Then 10% of the stock is lost (rounded down)." + Q_END
    assert fast_solve(q) == "900"    # 99.9 -> списано 99


def test_percent_without_rounding_only_when_exact():
    assert fast_solve("A warehouse starts with 200 boxes. 50% of the stock is sold." + Q_END) == "100"
    assert fast_solve("A warehouse starts with 999 boxes. 10% of the stock is sold." + Q_END) is None
    assert fast_solve("A warehouse starts with 400 boxes. The stock grows by 25%." + Q_END) == "500"


def test_filler_sentences_are_skipped():
    q = ("A warehouse starts with 500 boxes. The warehouse has 12 employees. It ships 20 boxes. "
         "It was built 7 years ago. Its manager has worked there for 9 years." + Q_END)
    assert fast_solve(q) == "480"


def test_sentences_about_the_stock_are_not_filler():
    # упоминает единицу запаса или действие - не пропускаем, отдаём агенту
    assert fast_solve("A warehouse starts with 500 boxes. The back room has 40 boxes." + Q_END) is None
    assert fast_solve("A warehouse starts with 500 boxes. It has 40 boxes left over." + Q_END) is None
    assert fast_solve("A warehouse starts with 500 boxes. The truck was sent with 40 crates." + Q_END) is None


def test_filler_needs_same_unit_in_question():
    q = "A warehouse starts with 500 boxes. The warehouse has 12 crates. How many crates does it have at the end?"
    assert fast_solve(q) is None
