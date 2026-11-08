"""Запасной агент для задач незнакомого вида: агент openJiuwen с инструментом python.

Жёсткие лимиты: не больше MAX_STEPS шагов, вывод инструмента обрезается (oj_agent.TOOL_OUTPUT_LIMIT).
"""

from .. import sandbox
from .base import ANSWER_RULES, clean_answer, is_unsolvable, strip_boilerplate

MAX_STEPS = 6

SYSTEM = ("You solve a task inside its directory. Use the python tool to inspect files and compute; "
          "it can also edit files. Keep tool output short. " + ANSWER_RULES +
          " Finish with one line 'FINAL: <answer>' (for tasks that only edit files: 'FINAL: DONE').")


def solve_generic(task, llm, plan, steps=MAX_STEPS):
    from ..oj_agent import final_answer, make_python_tool
    listing = sandbox.run(["bash", "-lc", "ls -la | head -40"], task.workspace, 10)[1]
    query = f"Task: {strip_boilerplate(task.instruction)}\n\nDirectory listing:\n{listing[:1500]}"
    text = llm.agent(system_prompt=SYSTEM, query=query, tools=[make_python_tool(task.workspace)],
                     max_iterations=steps, max_tokens=500)
    ans = clean_answer(final_answer(text))
    return None if is_unsolvable(ans) else ans


def analyze(task, fast_path=True):
    from .base import Plan, approx_tokens
    p = Plan()
    p.notes["generic"] = True
    p.est_cost = (approx_tokens(task.instruction) + 500) * 0.1 * 4 + 150 * 4
    p.p_success = 0.4
    return p


def solve_llm(task, llm, plan, strong=False):
    return solve_generic(task, llm, plan)
