"""Большие задачи: скрипт агента может вывести всю таблицу или зависнуть - задача не должна сломаться."""
import asyncio
import time

import pytest


def test_huge_output_is_clipped_and_final_is_kept(tmp_path):
    pytest.importorskip("openjiuwen")
    from harness.oj_agent import TOOL_OUTPUT_LIMIT, make_python_tool

    rows = "\n".join(f"{i},north,gadget,{i % 12 + 1},{i % 40},{i % 90 + 10}" for i in range(50_000))
    (tmp_path / "sales.csv").write_text("order_id,region,product,month,units,price\n" + rows + "\n")
    state = {}
    tool = make_python_tool(tmp_path, state)
    code = "print(open('sales.csv').read())\nprint('FINAL:', 123)"   # вся таблица в вывод, 1,3 МБ
    out = asyncio.run(tool.invoke({"code": code}))
    text = str(getattr(out, "content", out))
    assert len(text) <= TOOL_OUTPUT_LIMIT + 200      # в запрос модели уходит только хвост
    assert "FINAL: 123" in text and "[exit 0]" in text
    assert state == {"final": "123"}                 # ответ найден во всём выводе, а не в обрезанном


def test_hanging_script_is_stopped(tmp_path):
    from harness import sandbox

    t = time.time()
    rc, _, err = sandbox.run_python("import time\ntime.sleep(60)", tmp_path, timeout=2)
    assert rc == -9 and err == "TIMEOUT"
    assert time.time() - t < 10
