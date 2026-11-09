"""Порядок LLM-задач в фазе B (ORDER): est, reward, random."""
import shutil
from pathlib import Path

import pytest

from harness import config
from harness.runner import Runner

ROOT = Path(__file__).resolve().parent.parent


def _order(tmp_path, monkeypatch, order, seed=0):
    d = tmp_path / f"{order}_{seed}_{len(list(tmp_path.iterdir()))}"
    shutil.copytree(ROOT / "examples" / "tasks", d)
    monkeypatch.setattr(config, "FAST_PATH", False)   # все задачи идут в очередь LLM
    monkeypatch.setattr(config, "PARALLEL", 1)
    monkeypatch.setattr(config, "ORDER", order)
    monkeypatch.setattr(config, "ORDER_SEED", seed)
    monkeypatch.setattr(config, "TRAJ_PATH", str(tmp_path / "traj.jsonl"))
    r = Runner(d)
    r.llm.base = "http://unused"
    seen = []
    monkeypatch.setattr(Runner, "solve_one",
                        lambda self, task, mod, plan, pred: seen.append((task, pred, self.est.predict(task.task_type, plan)[1])))
    r.run()
    return seen


def test_reward_order_takes_highest_reward_first(tmp_path, monkeypatch):
    seen = _order(tmp_path, monkeypatch, "reward")
    rewards = [t.reward for t, _, _ in seen]
    assert len(seen) == 8
    assert rewards == sorted(rewards, reverse=True)


def test_est_order_is_reward_per_predicted_token(tmp_path, monkeypatch):
    seen = _order(tmp_path, monkeypatch, "est")
    ratios = [t.reward * p_ok / pred for t, pred, p_ok in seen]
    assert ratios == pytest.approx(sorted(ratios, reverse=True))


def test_random_order_is_reproducible(tmp_path, monkeypatch):
    a = [t.task_id for t, _, _ in _order(tmp_path, monkeypatch, "random", seed=1)]
    b = [t.task_id for t, _, _ in _order(tmp_path, monkeypatch, "random", seed=1)]
    c = [t.task_id for t, _, _ in _order(tmp_path, monkeypatch, "random", seed=2)]
    assert a == b
    assert sorted(a) == sorted(c)
    assert a != c
