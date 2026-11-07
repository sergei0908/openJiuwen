"""Починка кода под тесты: APR мутациями (0 токенов) -> LLM с проверкой тестами."""
import re
from pathlib import Path

from .. import sandbox
from . import apr
from .base import Plan, approx_tokens, strip_boilerplate


def find_files(ws: Path):
    py = sorted(p for p in ws.glob("*.py") if p.is_file())
    tests = [p for p in py if p.name.startswith("test") or p.name.endswith("_test.py")]
    targets = [p for p in py if p not in tests]
    return targets, tests


def test_cmd(task, tests):
    m = re.search(r"`(python3? [^`]+)`", task.instruction)
    if m:
        return m.group(1).split()
    return ["python3", "-m", "unittest", "-q"] + [t.stem for t in tests]


def run_tests(task, tests):
    cmd = test_cmd(task, tests)
    cmd[0] = sandbox.PY
    rc, out, err = sandbox.run(cmd, task.workspace, timeout=60)
    return rc == 0, (out + "\n" + err).strip()


def contradictory_tests(tests):
    """Одинаковый вызов с разными ожидаемыми значениями в assertEqual -> починить нельзя."""
    import ast
    seen = {}
    for t in tests:
        try:
            tree = ast.parse(t.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in ("assertEqual", "assertEquals") and len(n.args) == 2:
                a, b = n.args
                if isinstance(b, ast.Call) and not isinstance(a, ast.Call):
                    a, b = b, a
                if not isinstance(a, ast.Call):
                    continue
                key, val = ast.dump(a), ast.dump(b)
                if key in seen and seen[key] != val and isinstance(b, ast.Constant) \
                        and isinstance(seen.get(key + "#node"), ast.Constant):
                    return True
                seen.setdefault(key, val)
                seen.setdefault(key + "#node", b)
    return False


def analyze(task, fast_path=True):
    p = Plan()
    targets, tests = find_files(task.workspace)
    p.notes.update(targets=[t.name for t in targets], tests=[t.name for t in tests])
    if not tests or not targets:
        p.notes["generic"] = True
        p.est_cost = 2500
        p.p_success = 0.4
        return p
    ok, out = run_tests(task, tests)
    if ok:
        p.fast_done = True
        p.fast_answer = "DONE"
        return p
    p.notes["fail_out"] = out[-1500:]
    if fast_path:
        for tgt in targets:
            fixed, partial, info = apr.repair(task.workspace, tgt, tests)
            p.notes.setdefault("apr", {})[tgt.name] = info
            if fixed is not None:
                orig = tgt.read_text(encoding="utf-8")
                tgt.write_text(fixed + ("\n" if not fixed.endswith("\n") else ""), encoding="utf-8")
                ok, out = run_tests(task, tests)
                if ok:
                    p.fast_done = True
                    p.fast_answer = "DONE"
                    return p
                tgt.write_text(orig, encoding="utf-8")
            elif partial and info.get("applied"):
                p.notes.setdefault("partial", {})[tgt.name] = partial
    if contradictory_tests(tests):
        p.feasible = False
        return p
    size = sum(len(t.read_text(encoding="utf-8")) for t in targets + tests)
    src_tok = sum(approx_tokens(t.read_text(encoding="utf-8")) for t in targets)
    p.est_cost = (size / 4 + 400) * 0.1 + src_tok * 1.1 + 40
    p.p_success = 0.75
    return p


PROMPT = ("{task}\n\n{files}\n\nCurrent test output:\n```\n{out}\n```\n\n"
          "Fix the bugs in the non-test file(s). Do not change the tests. Reply with the COMPLETE "
          "corrected content of each changed file, each in its own ```python block whose first "
          "line is a comment `# FILE: <filename>`. No explanations.")


def _parse_files(text, targets):
    blocks = sandbox.CODE_RE.findall(text or "")
    res = {}
    names = [t.name for t in targets]
    for b in blocks:
        m = re.match(r"\s*#\s*FILE:\s*(\S+)\s*\n", b)
        if m and m.group(1) in names:
            res[m.group(1)] = b[m.end():]
        elif len(names) == 1 and b.strip():
            res.setdefault(names[0], b)
    return res


def solve_llm(task, llm, plan, strong=False):
    from .. import config
    if plan.notes.get("generic"):
        from .generic import solve_generic
        return solve_generic(task, llm, plan)
    targets, tests = find_files(task.workspace)
    originals = {t.name: t.read_text(encoding="utf-8") for t in targets}
    # стартуем с частичной APR-починки: меньше багов остаётся модели
    for name, src in plan.notes.get("partial", {}).items():
        (task.workspace / name).write_text(src + "\n", encoding="utf-8")
    ok, out = run_tests(task, tests)
    if ok:
        return "DONE"
    task_text = strip_boilerplate(task.instruction)
    attempts = [(config.MODEL_FAST, 2500)]
    if config.ALLOW_REASONING and task.reward >= 2:
        attempts.append((config.MODEL_STRONG, config.MAX_OUT_STRONG + 2500))
    else:
        attempts.append((config.MODEL_FAST, 2500))
    for model, max_out in attempts:
        files = "\n\n".join(f"File {t.name}:\n```python\n{t.read_text(encoding='utf-8')}```"
                            for t in targets + tests)
        msgs = [{"role": "user", "content": PROMPT.format(task=task_text, files=files,
                                                           out=out[-1200:])}]
        text, _cost, _meta = llm.chat(msgs, model=model, max_tokens=max_out)
        new = _parse_files(text, targets)
        if not new:
            continue
        for name, src in new.items():
            (task.workspace / name).write_text(src, encoding="utf-8")
        ok, out = run_tests(task, tests)
        if ok:
            return "DONE"
        if not llm.ledger.can_afford(plan.est_cost):
            break
    for name, src in originals.items():  # не оставляем испорченные файлы
        (task.workspace / name).write_text(src, encoding="utf-8")
    return None
