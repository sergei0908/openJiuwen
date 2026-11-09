"""Симуляция кривой бюджета по трассам (tools/budget_sim.py)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from budget_sim import curve, load, main, simulate


def traces(tmp_path):
    rows = [
        # дешёвая задача с большой наградой, дорогая с маленькой, провальная, бесплатная
        {"task_id": "a", "type": "math", "route": "llm", "reward": 5, "cost": 100, "ok": True, "pred_cost": 100},
        {"task_id": "b", "type": "coding", "route": "llm", "reward": 1, "cost": 900, "ok": True, "pred_cost": 900},
        {"task_id": "c", "type": "search", "route": "llm", "reward": 3, "cost": 50, "ok": False, "pred_cost": 50},
        {"task_id": "d", "type": "math", "route": "fast", "reward": 2, "cost": 0},
        {"task_id": "e", "type": "math", "route": "infeasible", "reward": 4, "cost": 0},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return [str(p)]


def test_load_skips_infeasible(tmp_path):
    assert {t["id"].split(":")[-1] for t in load(traces(tmp_path))} == {"a", "b", "c", "d"}


def test_oracle_takes_cheap_valuable_first(tmp_path):
    tasks = load(traces(tmp_path))
    assert simulate(tasks, 150, "oracle")[0] == 2 + 5   # бесплатная + «a»; «c» провальная - оракул её не берёт
    assert simulate(tasks, 10_000, "oracle")[0] == 2 + 5 + 1


def test_reward_order_wastes_budget_on_failures(tmp_path):
    tasks = load(traces(tmp_path))
    score, spent = simulate(tasks, 120, "reward")
    assert score == 2 + 5 and spent == 100


def test_task_over_remaining_budget_is_cut(tmp_path):
    tasks = load(traces(tmp_path))
    tasks[0]["pred_cost"] = 10   # прогноз сильно занижен: задача не помещается и обрывается
    score, spent = simulate(tasks, 60, "est")
    assert score == 2 and spent == 60


def test_curve_and_report(tmp_path, capsys):
    tasks = load(traces(tmp_path))
    table = curve(tasks, [100, 1000], shuffles=5)
    assert set(table) == {"random", "reward", "est", "oracle"}
    assert all(table["oracle"][i] >= table["random"][i] for i in range(2))
    assert main([*traces(tmp_path), "--budgets", "100,1000", "--png", str(tmp_path / "c.png")]) == 0
    assert (tmp_path / "c.png").stat().st_size > 1000
    assert "est/oracle" in capsys.readouterr().out
