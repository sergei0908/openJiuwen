"""Находки прогона рабочего набора Маши на develop_16 (11.11)."""
import json
from pathlib import Path

from harness.estimator import MODEL_PATH, linear_predict
from harness.solvers.base import AGENT_GAVE_UP
from harness.solvers.search_solver import analyze
from harness.tasks import Task, match_decimals


def test_reverse_link_for_active_relation(tmp_path):
    # maria_54: «Hiro Petrov mentors exactly one person… Which company employs that person?»
    kb = tmp_path / "w" / "kb"
    kb.mkdir(parents=True)
    (kb / "hiro.md").write_text("# Hiro Petrov\n\nBorn: 1969\nEmployer: Volga Data\n")
    (kb / "nina.md").write_text("# Nina Gray\n\nBorn: 1994\nEmployer: Polar Bit\nMentor: Hiro Petrov\n")
    (kb / "polar.md").write_text("# Polar Bit\n\nHeadquarters: Norilsk\n")
    (kb / "volga.md").write_text("# Volga Data\n\nHeadquarters: Samara\n")
    instr = ("The directory kb/ contains a small knowledge base of markdown files. Hiro Petrov mentors exactly one "
             "person in the knowledge base. Which company employs that person? Work only inside the current directory.")
    p = analyze(Task("search_x", tmp_path / "w", "search", instr, 4.0))
    assert p.feasible is not False
    assert "Nina Gray" in p.context and "Polar Bit" in p.context


def test_trailing_zeros_removed_when_places_requested():
    q = "What was the average price? Round to two decimal places."
    assert match_decimals(q, "25.40") == "25.4"      # maria_11
    assert match_decimals(q, "25.00") == "25"
    assert match_decimals(q, "25.4") == "25.4"
    assert match_decimals("How many orders are there?", "25.40") == "25.40"   # знаки не просили - не трогаем


def test_agent_gave_up_is_not_an_answer():
    assert AGENT_GAVE_UP.search("Max iterations reached without completion")  # maria_27
    assert not AGENT_GAVE_UP.search("Polar Bit")


def test_scale_is_applied():
    m = json.loads(Path(MODEL_PATH).read_text(encoding="utf-8"))["types"]["math"]
    feats = {k: 1 for k in m["coef"]}
    base = linear_predict(dict(m, scale=1.0), feats)
    assert abs(linear_predict(m, feats) - base * m["scale"]) < 1e-6 or base * m["scale"] > m["clip"][1]
