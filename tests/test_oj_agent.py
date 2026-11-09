"""Тесты LLM-пути через агента openJiuwen (на мок-прокси, без сети и ключа)."""
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness.oj_agent import final_answer

ROOT = Path(__file__).resolve().parent.parent


def test_final_answer_marker():
    assert final_answer("thinking...\nFINAL: 389") == "389"
    assert final_answer("FINAL: Osaka\nextra line") == "Osaka"


def test_final_answer_without_marker_takes_last_line():
    assert final_answer("The result is\n42\n") == "42"
    assert final_answer("") == ""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_proxy(monkeypatch):
    pytest.importorskip("openjiuwen")
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "mock_proxy.py"), "--port", str(port),
                             "--max-concurrent", "3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("LLM_API_KEY", "test")
    yield port
    proc.terminate()
    proc.wait(5)


def test_agent_runs_tool_and_charges_budget(mock_proxy, tmp_path):
    from harness.llm import LLM, Ledger
    from harness.oj_agent import make_python_tool
    from harness.runner import TaskLLM

    llm = LLM(Ledger(budget=10_000))
    task = TaskLLM(llm, cap=1_000)
    text = task.agent(system_prompt="Use the python tool, then reply 'FINAL: <answer>'.",
                      query="Compute 40 + 2.", tools=[make_python_tool(tmp_path)], max_iterations=3)

    # мок вызывает python (печатает 42), затем отвечает FINAL: <вывод инструмента>
    assert final_answer(text) == "42"
    assert len(task.calls) == 2              # вызов инструмента + финальный ответ
    assert task.spent > 0
    assert llm.ledger.spent == pytest.approx(task.spent)
    # служебные файлы DeepAgent не попадают в папку задачи
    assert not (tmp_path / "AGENT.md").exists()


def test_task_cap_stops_agent(mock_proxy, tmp_path):
    from harness.llm import LLM, Ledger
    from harness.oj_agent import TaskBudgetExceeded, make_python_tool
    from harness.runner import TaskLLM

    llm = LLM(Ledger(budget=10_000))
    task = TaskLLM(llm, cap=1)               # бюджет задачи кончится после первого вызова
    with pytest.raises(TaskBudgetExceeded):
        task.agent(system_prompt="Use the python tool.", query="Compute 40 + 2.",
                   tools=[make_python_tool(tmp_path)], max_iterations=3)
    assert len(task.calls) == 1


def test_agent_stops_when_tests_pass(mock_proxy, tmp_path):
    from harness.llm import LLM, Ledger
    from harness.oj_agent import make_coding_tools
    from harness.runner import TaskLLM

    (tmp_path / "mylib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    tools, state = make_coding_tools(tmp_path, {"mylib.py"}, [sys.executable, "-c", "pass"])
    task = TaskLLM(LLM(Ledger(budget=10_000)), cap=1_000)
    text = task.agent(system_prompt="Fix the code with update_code.",
                      query="```python\ndef f():\n    return 2\n```", tools=tools, max_iterations=3,
                      stop=lambda: state["passed"])

    # тесты прошли после первого вызова инструмента - финальный ответ модели не заказываем
    assert state == {"passed": True, "updates": 1}
    assert final_answer(text) == "DONE"
    assert len(task.calls) == 1
    assert "return 2" in (tmp_path / "mylib.py").read_text(encoding="utf-8")


def test_agent_stops_when_script_prints_final(mock_proxy, tmp_path):
    from harness.llm import LLM, Ledger
    from harness.oj_agent import final_stop, make_python_tool
    from harness.runner import TaskLLM
    from harness.solvers.base import PRINT_FINAL

    state = {}
    task = TaskLLM(LLM(Ledger(budget=10_000)), cap=1_000)
    text = task.agent(system_prompt="Use the python tool. " + PRINT_FINAL, query="Compute 40 + 2.",
                      tools=[make_python_tool(tmp_path, state)], max_iterations=3, stop=final_stop(state))

    # ответ напечатан скриптом - второй вызов модели («переписать ответ») не нужен
    assert state == {"final": "42"}
    assert final_answer(text) == "42"
    assert len(task.calls) == 1


def test_failed_script_final_is_ignored(tmp_path):
    pytest.importorskip("openjiuwen")
    import asyncio

    from harness.oj_agent import make_python_tool

    state = {}
    tool = make_python_tool(tmp_path, state)
    asyncio.run(tool.invoke({"code": "print('FINAL: 1')\nraise SystemExit(1)"}))
    assert state == {}


def test_agents_do_not_leak_threads_or_files(mock_proxy, tmp_path):
    """filelock 4.x + openJiuwen 0.1.19: каждая файловая блокировка оставляла поток и SQLite-файл
    (~14 потоков на агента), и под --pids-limit 512 прогон падал с «can't start new thread»."""
    import os
    import threading

    from harness.llm import LLM, Ledger
    from harness.oj_agent import make_python_tool
    from harness.runner import TaskLLM

    llm = LLM(Ledger(budget=10_000))

    def run_one():
        task = TaskLLM(llm, cap=1_000)
        task.agent(system_prompt="Use the python tool.", query="Compute 40 + 2.",
                   tools=[make_python_tool(tmp_path)], max_iterations=3)

    def fds():
        return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else 0

    def rwlock_threads():
        return sum(t.name.startswith("openjiuwen-rwlock") for t in threading.enumerate())

    run_one()  # первый агент поднимает общий цикл и служебные потоки
    threads, files, rw = threading.active_count(), fds(), rwlock_threads()
    for _ in range(3):
        run_one()
    # с утечкой было бы +42 потока и +84 файла
    assert threading.active_count() <= threads + 2
    assert fds() <= files + 6
    assert rwlock_threads() <= rw
