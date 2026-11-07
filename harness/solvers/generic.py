"""Запасной агент для незнакомых задач: короткий цикл «команда bash -> наблюдение».

Жёсткие лимиты: не больше MAX_STEPS шагов, вывод команд обрезается, история не растёт
бесконечно (оставляем только последние наблюдения) - иначе входные токены съедают бюджет.
"""
import re

from .. import sandbox
from .base import ANSWER_RULES, clean_answer, is_unsolvable, strip_boilerplate

MAX_STEPS = 6
OBS_LIMIT = 1200

SYSTEM = ("You solve a task inside a Linux directory using bash. Each reply must be EITHER one "
          "```bash block with a single command to run, OR a line 'FINAL: <answer>'. Be brief, "
          "no explanations. Prefer python3 -c for computation. " + ANSWER_RULES +
          " For tasks that require editing files, edit them with bash/python, then reply FINAL: DONE.")


def solve_generic(task, llm, plan, steps=MAX_STEPS):
    from .. import config
    listing = sandbox.run(["bash", "-lc", "ls -la | head -40"], task.workspace, 10)[1]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Task: {strip_boilerplate(task.instruction)}\n\n"
                                        f"Directory listing:\n{listing[:1500]}"}]
    for _ in range(steps):
        text, _cost, _meta = llm.chat(msgs, model=config.MODEL_FAST, max_tokens=500)
        m = re.search(r"FINAL:\s*(.+)", text)
        if m:
            ans = clean_answer(m.group(1).splitlines()[0])
            return None if is_unsolvable(ans) else ans
        cmd = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", text, re.DOTALL)
        if not cmd:
            msgs.append({"role": "assistant", "content": text[:500]})
            msgs.append({"role": "user", "content": "Reply with a ```bash block or FINAL: <answer>."})
            continue
        rc, out, err = sandbox.run(["bash", "-lc", cmd.group(1)], task.workspace, 30)
        obs = (out + ("\n[stderr]\n" + err if err else ""))[-OBS_LIMIT:]
        msgs.append({"role": "assistant", "content": text[:800]})
        msgs.append({"role": "user", "content": f"exit={rc}\n{obs}"})
        # компактная история: system + задача + последние 2 шага
        if len(msgs) > 6:
            msgs = msgs[:2] + msgs[-4:]
    return None


def analyze(task, fast_path=True):
    from .base import Plan, approx_tokens
    p = Plan()
    p.notes["generic"] = True
    p.est_cost = (approx_tokens(task.instruction) + 500) * 0.1 * 4 + 150 * 4
    p.p_success = 0.4
    return p


def solve_llm(task, llm, plan, strong=False):
    return solve_generic(task, llm, plan)
