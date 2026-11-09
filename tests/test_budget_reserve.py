"""Резерв бюджета: параллельные задачи не выходят за общий бюджет."""
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness import config
from harness.llm import Ledger

ROOT = Path(__file__).resolve().parent.parent


def test_reserve_blocks_parallel_overcommit(monkeypatch):
    monkeypatch.setattr(config, "RESERVE_FRAC", 0.0)
    led = Ledger(budget=100)
    assert led.try_reserve(60)
    assert not led.try_reserve(60)          # второй воркер видит остаток за вычетом резерва
    assert led.free() == pytest.approx(40)
    led.release(60)
    assert led.try_reserve(60)


def test_task_spending_consumes_own_reservation(monkeypatch):
    from harness.llm import LLM
    from harness.runner import TaskLLM

    monkeypatch.setattr(config, "RESERVE_FRAC", 0.0)
    llm = LLM(Ledger(budget=100))
    assert llm.ledger.try_reserve(50)
    t = TaskLLM(llm, cap=1000, reserve=50)
    llm.ledger.add(300, 0)                   # 30 в единицах стоимости
    t.record("m", 300, 0, 30.0)
    assert t.res_left == pytest.approx(20)
    assert llm.ledger.reserved == pytest.approx(20)
    assert t.available() == pytest.approx(70 + 20 - 20)  # free (100-30-20) + свой резерв 20
    t.finish()
    assert llm.ledger.reserved == 0


def test_unlimited_budget_needs_no_reservation():
    led = Ledger(budget=None)
    assert led.try_reserve(10_000)
    assert led.reserved == 0


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("budget", [600, 1500])
def test_tight_budget_is_not_exceeded(tmp_path, monkeypatch, budget):
    pytest.importorskip("openjiuwen")
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "mock_proxy.py"), "--port", str(port),
                             "--max-concurrent", "3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        tasks = tmp_path / "tasks"
        subprocess.run([sys.executable, str(ROOT / "tools" / "gen_tasks.py"), "--out", str(tasks), "--n", "30",
                        "--seed", "7"], check=True, capture_output=True)
        monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("LLM_API_KEY", "test")
        monkeypatch.setattr(config, "FAST_PATH", False)
        monkeypatch.setattr(config, "PARALLEL", 5)
        monkeypatch.setattr(config, "TRAJ_PATH", str(tmp_path / "traj.jsonl"))
        from harness.runner import Runner
        monkeypatch.setenv("BUDGET", str(budget))
        r = Runner(tasks)
        stats = r.run()
        assert stats["llm_ok"] > 0
        assert stats["tokens_spent"] <= budget
        assert r.ledger.reserved == pytest.approx(0, abs=1e-6)
    finally:
        proc.terminate()
        proc.wait(5)
