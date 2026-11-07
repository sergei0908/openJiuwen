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
