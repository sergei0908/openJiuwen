"""Починка кода под тесты: APR мутациями (0 токенов) -> LLM с проверкой тестами.

Выход модели стоит в 10 раз дороже входа, поэтому модель присылает только изменённые функции, а
харнесс вклеивает их в файл. Первая попытка видит только упавшие тесты и связанные с ними функции,
вторая - файлы целиком: ошибка может быть в месте, которое мы не показали.
"""
import ast
import os
import re
import sys
from pathlib import Path

from .. import sandbox
from ..tasks import can_write, write_file
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


def can_fix(ws, targets):
    return can_write(ws) or all(os.access(t, os.W_OK) for t in targets)


def contradictory_tests(tests):
    """Одинаковый вызов с разными ожидаемыми значениями в assertEqual -> починить нельзя."""
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
    if not can_fix(task.workspace, targets):  # исправление некуда сохранить: не тратим токены
        print(f"ERROR cannot modify files of {task.task_id}: no write permission", file=sys.stderr, flush=True)
        p.feasible = False
        p.notes["unwritable"] = True
        return p
    if contradictory_tests(tests):  # до перебора: на такой задаче он только потратит время
        p.feasible = False
        return p
    if fast_path:
        for tgt in targets:
            fixed, partial, info = apr.repair(task.workspace, tgt, tests)
            p.notes.setdefault("apr", {})[tgt.name] = info
            if fixed is not None:
                orig = tgt.read_text(encoding="utf-8")
                write_file(tgt, fixed + ("\n" if not fixed.endswith("\n") else ""))
                ok, out = run_tests(task, tests)
                if ok:
                    p.fast_done = True
                    p.fast_answer = "DONE"
                    return p
                write_file(tgt, orig)
            elif partial and info.get("applied"):
                p.notes.setdefault("partial", {})[tgt.name] = partial
    size = sum(len(t.read_text(encoding="utf-8")) for t in targets + tests)
    src_tok = sum(approx_tokens(t.read_text(encoding="utf-8")) for t in targets)
    p.est_cost = (size / 8 + 150) * 0.1 + src_tok * 0.35 + 35
    p.p_success = 0.75
    return p


PROMPT = ("{task}\n\n{files}\n\nCurrent test output:\n```\n{out}\n```\n\n"
          "Fix the bugs. Do not change the tests. Reply with ONLY the corrected functions, each one "
          "complete, in one ```python block. Do not repeat unchanged functions. {multi}No explanations.")
MULTI = "Start the block of each changed file with a comment `# FILE: <filename>`. "


def _top_level(tree):
    """Имя -> узел верхнего уровня: функции, классы и простые присваивания."""
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = n
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = n
    return out


def merge_reply(src, reply):
    """Вклеивает присланные функции в исходный файл. None, если ответ нельзя использовать."""
    try:
        new_tree, old_tree = ast.parse(reply), ast.parse(src)
    except SyntaxError:
        return None
    new, old = _top_level(new_tree), _top_level(old_tree)
    if not new:
        return None
    lines = src.splitlines()
    reply_lines = reply.splitlines()
    spans = []
    added = []
    for name, node in new.items():
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        text = reply_lines[first - 1:node.end_lineno]
        if name in old:
            o = old[name]
            spans.append((min([o.lineno] + [d.lineno for d in getattr(o, "decorator_list", [])]), o.end_lineno, text))
        else:
            added.extend(["", ""] + text)
    for first, last, text in sorted(spans, key=lambda x: x[0], reverse=True):
        lines[first - 1:last] = text
    known = {ast.unparse(n) for n in old_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    imports = [ast.unparse(n) for n in new_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
               and ast.unparse(n) not in known]
    merged = "\n".join(imports + lines + added) + "\n"
    try:
        ast.parse(merged)
    except SyntaxError:
        return None
    return merged


def _used_names(node):
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.Name):
            names.add(n.id)
    return names


def failing_context(targets, tests, out):
    """Упавшие тесты и функции, которые они вызывают (с вызываемыми из них). None, если не выделить."""
    failed = set(re.findall(r"^(?:FAIL|ERROR): (\w+)", out, flags=re.MULTILINE))
    if not failed or len(targets) != 1:
        return None
    try:
        src = targets[0].read_text(encoding="utf-8")
        funcs = _top_level(ast.parse(src))
        parts, used = [], set()
        for t in tests:
            text = t.read_text(encoding="utf-8")
            for n in ast.walk(ast.parse(text)):
                if isinstance(n, ast.FunctionDef) and n.name in failed:
                    parts.append(ast.get_source_segment(text, n, padded=True))
                    used |= _used_names(n)
    except (SyntaxError, OSError, UnicodeDecodeError):
        return None
    chosen = used & set(funcs)
    if not chosen or not parts:
        return None
    queue = list(chosen)
    while queue:  # функции, которые вызываются из выбранных
        for name in _used_names(funcs[queue.pop()]) & set(funcs):
            if name not in chosen:
                chosen.add(name)
                queue.append(name)
    picked = [ast.get_source_segment(src, funcs[n]) for n in funcs if n in chosen]
    return (f"Functions from {targets[0].name}:\n```python\n" + "\n\n".join(picked) + "\n```\n\n"
            "Failing tests:\n```python\n" + "\n\n".join(parts) + "\n```")


def _parse_files(text, targets):
    blocks = sandbox.CODE_RE.findall(text or "")
    res = {}
    names = [t.name for t in targets]
    for b in blocks:
        m = re.match(r"\s*#\s*FILE:\s*(\S+)\s*\n", b)
        if m and m.group(1) in names:
            res[m.group(1)] = b[m.end():]
        elif len(names) == 1 and b.strip():
            res[names[0]] = (res[names[0]] + "\n\n" + b) if names[0] in res else b
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
        write_file(task.workspace / name, src + "\n")
    ok, out = run_tests(task, tests)
    if ok:
        return "DONE"
    task_text = strip_boilerplate(task.instruction)
    attempts = [(config.MODEL_FAST, 2500)]
    if config.ALLOW_REASONING and task.reward >= 2:
        attempts.append((config.MODEL_STRONG, config.MAX_OUT_STRONG + 2500))
    else:
        attempts.append((config.MODEL_FAST, 2500))
    multi = MULTI if len(targets) > 1 else ""
    for attempt, (model, max_out) in enumerate(attempts):
        files = failing_context(targets, tests, out) if attempt == 0 else None
        if files is None:
            files = "\n\n".join(f"File {t.name}:\n```python\n{t.read_text(encoding='utf-8')}```"
                                for t in targets + tests)
        msgs = [{"role": "user", "content": PROMPT.format(task=task_text, files=files, multi=multi,
                                                           out=out[-1200:])}]
        text, _cost, _meta = llm.chat(msgs, model=model, max_tokens=max_out)
        changed = False
        for name, reply in _parse_files(text, targets).items():
            path = task.workspace / name
            merged = merge_reply(path.read_text(encoding="utf-8"), reply)
            if merged is not None:
                write_file(path, merged)
                changed = True
        if not changed:
            continue
        ok, out = run_tests(task, tests)
        if ok:
            return "DONE"
        if not llm.ledger.can_afford(plan.est_cost):
            break
    for name, src in originals.items():  # не оставляем испорченные файлы
        write_file(task.workspace / name, src)
    return None
