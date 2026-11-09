"""Обучение оценщика по трассам (tools/fit_cost_model.py) и прогноз (harness/estimator.py)."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from fit_cost_model import main as fit

from harness.estimator import Estimator
from harness.solvers.base import Plan


def write_traces(d: Path, seeds=4, n=30):
    """data_analysis: расход растёт с числом строк; math: постоянный с шумом."""
    files = []
    for s in range(seeds):
        r = random.Random(s)
        p = d / f"seed_{s}.jsonl"
        lines = []
        for i in range(n):
            rows = r.randint(20, 2000)
            lines.append({"task_id": f"data_analysis_{i}", "type": "data_analysis", "route": "llm",
                          "cost": 90 + 0.05 * rows + r.gauss(0, 3), "ok": i % 10 != 0,
                          "features": {"instr_chars": 380, "data_rows": rows, "data_cols": r.randint(4, 8)}})
            lines.append({"task_id": f"math_{i}", "type": "math", "route": "llm",
                          "cost": 100 + r.gauss(0, 5), "ok": True,
                          "features": {"instr_chars": r.randint(300, 700), "math_numbers": r.randint(2, 9)}})
            lines.append({"task_id": f"x_{i}", "type": "math", "route": "fast", "cost": 0, "features": {}})
        p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
        files.append(str(p))
    return files


def test_fit_beats_mean_when_cost_depends_on_features(tmp_path):
    out = tmp_path / "cost_model.json"
    assert fit([*write_traces(tmp_path), "--out", str(out)]) == 0
    m = json.loads(out.read_text(encoding="utf-8"))["types"]
    d = m["data_analysis"]
    assert d["kind"] in ("linear", "loglin")
    assert d["cv_mae"] < d["cv_mae_mean"] / 3
    assert d["n"] == 120
    assert 0.8 < d["p_success"] < 0.95
    assert m["math"]["cv_mae"] < 10


def test_estimator_uses_features(tmp_path):
    out = tmp_path / "cost_model.json"
    fit([*write_traces(tmp_path), "--out", str(out)])
    est = Estimator(out)

    def plan(rows):
        p = Plan(est_cost=999)
        p.notes["features"] = {"instr_chars": 380, "data_rows": rows, "data_cols": 6}
        return p

    small, _ = est.predict("data_analysis", plan(100))
    big, p_ok = est.predict("data_analysis", plan(1800))
    assert small == pytest.approx(95, abs=8)
    assert big == pytest.approx(180, abs=12)
    assert 0.8 < p_ok < 0.95


def test_estimator_falls_back_to_heuristic(tmp_path):
    est = Estimator(tmp_path / "missing.json")
    p = Plan(est_cost=123, p_success=0.7)
    assert est.predict("coding", p) == (123, 0.7)
    out = tmp_path / "cost_model.json"
    fit([*write_traces(tmp_path), "--out", str(out)])
    p.notes["generic"] = True  # универсальный путь - эвристика, даже если модель есть
    assert Estimator(out).predict("data_analysis", p)[0] == 123


def test_old_model_format_still_works(tmp_path):
    out = tmp_path / "cost_model.json"
    out.write_text(json.dumps({"math": {"a": 10, "b": 0.5, "p_success": 0.6}}), encoding="utf-8")
    p = Plan()
    p.notes["features"] = {"prompt_tokens": 100}
    assert Estimator(out).predict("math", p) == (60, 0.6)


def test_online_calibration_is_gentle(tmp_path):
    est = Estimator(tmp_path / "missing.json")
    p = Plan(est_cost=100)
    est.observe("math", 100, 200)  # одна задача стоила вдвое дороже прогноза
    pred, _ = est.predict("math", p)
    assert 100 < pred < 120
