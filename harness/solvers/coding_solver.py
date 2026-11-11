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

SKIP_DIRS = {"__pycache__", "venv", ".venv", "node_modules", "site-packages"}
NOT_TARGETS = {"__init__.py", "conftest.py", "setup.py"}
RUNNER = str(Path(__file__).resolve().parent.parent / "testrun.py")
NO_TESTS = re.compile(r"^Ran 0 tests|NO TESTS RAN|no tests ran|collected 0 items", re.MULTILINE)


def find_files(ws: Path):
    """Код и тесты задачи, в том числе в подпапках (tests/, src/); служебные папки пропускаем."""
    py = sorted(p for p in ws.rglob("*.py") if p.is_file()
                and not any(part.startswith(".") or part in SKIP_DIRS for part in p.relative_to(ws).parts))
    tests = [p for p in py if p.name.startswith("test") or p.name.endswith("_test.py")]
    targets = [p for p in py if p not in tests and p.name not in NOT_TARGETS]
    return targets, tests


def rel(path, ws):
    """Имя файла задачи относительно её папки: «mylib.py» или «src/mylib.py»."""
    return path.relative_to(ws).as_posix()


def _unittest_style(tests):
    try:
        return all("unittest" in t.read_text(encoding="utf-8", errors="replace") for t in tests)
    except OSError:
        return False


def test_cmd(task, tests):
    """Команда тестов: всегда наша, а не из условия. Тесты в корне и на unittest - `python -m unittest`
    по всем тестовым файлам. Тесты в подпапке или в стиле pytest - наш запускатель harness/testrun.py
    (pytest в контейнере нет): иначе `python test_x.py` с функциями-тестами «проходит», ничего не проверив.

    Команду из условия не берём: если команд две, первая запускает не все тесты (DONE без починки),
    а `pytest` в условии падает всегда - модуля нет в образе."""
    ws = task.workspace
    flat = all(t.parent == ws for t in tests)
    if flat and _unittest_style(tests) and all(t.stem.isidentifier() for t in tests):
        return ["python3", "-m", "unittest", "-q"] + [t.stem for t in tests]
    return ["python3", RUNNER] + [str(t.relative_to(ws)) for t in tests]


def run_tests(task, tests):
    cmd = test_cmd(task, tests)
    cmd[0] = sandbox.PY
    rc, out, err = sandbox.run(cmd, task.workspace, timeout=60)
    out = (out + "\n" + err).strip()
    return rc == 0 and not NO_TESTS.search(out), out


def can_fix(ws, targets):
    return can_write(ws) or all(os.access(t, os.W_OK) for t in targets)


def _module_names(tree):
    """Имена, под которыми тест видит модули и функции из импортов (`import mylib as m`, `from mylib import f`)."""
    mods, funcs = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            funcs.update(a.asname or a.name for a in n.names)
    return mods, funcs


def _pure_call(node, mods, funcs):
    """Вызов функции модуля с константными аргументами: m.f(1, [2, 3]) или f('a').
    Методы объектов (s.pop(), counter.next()) сюда не попадают: у объекта есть состояние, и один и тот же
    вызов законно возвращает разное."""
    f = node.func
    if isinstance(f, ast.Attribute):
        if not (isinstance(f.value, ast.Name) and f.value.id in mods):
            return False
    elif not (isinstance(f, ast.Name) and f.id in funcs):
        return False
    try:
        for a in [*node.args, *(k.value for k in node.keywords)]:
            ast.literal_eval(a)
    except ValueError:
        return False
    return True


def contradictory_tests(tests):
    """Одинаковый вызов функции модуля с разными ожидаемыми константами в assertEqual -> починить нельзя.

    Только «уверен или молчит»: вызовы методов объектов и вызовы с переменными в аргументах не
    сравниваем - стек, счётчик или генератор законно отвечают по-разному на один и тот же вызов."""
    seen = {}
    for t in tests:
        try:
            tree = ast.parse(t.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        mods, funcs = _module_names(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in ("assertEqual", "assertEquals") and len(n.args) == 2:
                a, b = n.args
                if isinstance(b, ast.Call) and not isinstance(a, ast.Call):
                    a, b = b, a
                if not isinstance(a, ast.Call) or not isinstance(b, ast.Constant) or not _pure_call(a, mods, funcs):
                    continue
                key = ast.dump(a)
                if key in seen and seen[key] != b.value:
                    return True
                seen.setdefault(key, b.value)
    return False


IDENTITY = {ast.Add: 0, ast.Sub: 0, ast.Mult: 1, ast.Div: 1, ast.FloorDiv: 1, ast.Pow: 1}


def _funcs(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _const_indices(fn):
    return sorted(n.slice.value for n in ast.walk(fn)
                  if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                  and isinstance(n.slice.value, int))


def suspicious_patch(orig, fixed):
    """Похожа ли починка перебором на подгонку под видимые тесты, а не на исправление.

    Перебор меняет по одному узлу за раз, и на багах из двух правок находит «случайное» решение,
    которое проходит видимые тесты, но не скрытые (замер на 600 задачах: 7 таких починок):
      - нейтрализованная константа: `x // n + 1` -> `x // n + 0` (настоящий баг был ещё и в `//`);
      - подобранный константный индекс: `ys[0]` -> `ys[1]` (правильно `ys[len(ys) // 2]`).
    Такие починки отдаём агенту: он видит смысл функции, а не только тесты.
    """
    before, after = _funcs(orig), _funcs(fixed)
    for name, fn in after.items():
        old = before.get(name)
        if old is None or ast.dump(old) == ast.dump(fn):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.BinOp) and type(n.op) in IDENTITY and isinstance(n.right, ast.Constant) \
                    and n.right.value == IDENTITY[type(n.op)] and not isinstance(n.right.value, bool):
                return f"{name}: identity {type(n.op).__name__} {n.right.value!r}"
        new_idx = set(_const_indices(fn)) - set(_const_indices(old))
        if new_idx:  # появился новый константный индекс - его подобрали под тест
            return f"{name}: constant index {_const_indices(old)} -> {_const_indices(fn)}"
    return None


def analyze(task, fast_path=True):
    p = Plan()
    targets, tests = find_files(task.workspace)
    ws = task.workspace
    p.notes.update(targets=[rel(t, ws) for t in targets], tests=[rel(t, ws) for t in tests])
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
            orig = tgt.read_text(encoding="utf-8")
            why = suspicious_patch(orig, fixed if fixed is not None else partial or orig)
            if why:  # похоже на подгонку под видимые тесты - чинить будет агент с исходного кода
                p.notes.setdefault("apr_suspect", {})[tgt.name] = why
                continue
            if fixed is not None:
                write_file(tgt, fixed + ("\n" if not fixed.endswith("\n") else ""))
                ok, out = run_tests(task, tests)
                if ok:
                    p.fast_done = True
                    p.fast_answer = "DONE"
                    return p
                write_file(tgt, orig)
            elif partial and info.get("applied"):
                p.notes.setdefault("partial", {})[rel(tgt, ws)] = partial
    size = sum(len(t.read_text(encoding="utf-8")) for t in targets + tests)
    src_tok = sum(approx_tokens(t.read_text(encoding="utf-8")) for t in targets)
    p.est_cost = (size / 8 + 150) * 0.1 + src_tok * 0.35 + 35
    p.p_success = 0.75
    return p


SYSTEM = ("You fix bugs in Python code so that the tests pass. Never change the tests. Call update_code "
          "with ONLY the corrected functions, each complete, without comments; it applies them and runs the "
          "tests. No text besides the tool call. If the tests fail, fix the remaining bugs the same way. If "
          "the tests contradict each other and cannot all pass, reply 'FINAL: IMPOSSIBLE'.")
QUERY = "{task}\n\n{files}\n\nCurrent test output:\n```\n{out}\n```"
MULTI = "\n\nSeveral source files: pass `file` to update_code."


def compact_test_output(out, limit=900):
    """Сжатый вывод unittest для промпта: по каждому упавшему тесту - заголовок и последние строки
    (место и текст ошибки), без трассировок и абсолютных путей. Вход модели тоже стоит денег."""
    out = re.sub(r'File "[^"]*/([^"/]+)"', r'File "\1"', out or "")
    summary = re.findall(r"^(?:Ran \d+ tests?.*|FAILED \(.*\)|OK)$", out, flags=re.MULTILINE)
    body_out = re.split(r"^-+\s*\nRan \d+ tests?", out, flags=re.MULTILINE)[0]
    blocks = re.split(r"^=+$", body_out, flags=re.MULTILINE)
    parts = []
    for b in blocks[1:]:
        lines = [x for x in b.strip().splitlines() if x.strip() and not set(x.strip()) <= {"-"}]
        if not lines:
            continue
        body = [x for x in lines[1:] if not x.startswith("Traceback")]
        parts.append("\n".join([lines[0]] + body[-3:]))
    text = "\n\n".join(parts + summary[-1:]) if parts else out.strip()
    return text if len(text) <= limit else text[:limit] + "\n..."


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


def solve_llm(task, llm, plan, strong=False):
    from .. import config
    from ..oj_agent import make_coding_tools
    if plan.notes.get("generic"):
        from .generic import solve_generic
        return solve_generic(task, llm, plan)
    targets, tests = find_files(task.workspace)
    ws = task.workspace
    originals = {rel(t, ws): t.read_text(encoding="utf-8") for t in targets}
    passed = False
    try:
        # стартуем с частичной APR-починки: меньше багов остаётся модели
        for name, src in plan.notes.get("partial", {}).items():
            write_file(task.workspace / name, src + "\n")
        ok, out = run_tests(task, tests)
        if ok:
            passed = True
            return "DONE"
        cmd = test_cmd(task, tests)
        cmd[0] = sandbox.PY
        task_text = strip_boilerplate(task.instruction)
        attempts = [(config.MODEL_FAST, 2500)]
        if config.ALLOW_REASONING and task.reward >= 2:
            attempts.append((config.MODEL_STRONG, config.MAX_OUT_STRONG + 2500))
        else:
            attempts.append((config.MODEL_FAST, 2500))
        multi = MULTI if len(targets) > 1 else ""
        for attempt, (model, max_out) in enumerate(attempts):
            # первая попытка - только упавшие тесты и нужные функции, вторая - файлы целиком
            files = failing_context(targets, tests, out) if attempt == 0 else None
            if files is None:
                files = "\n\n".join(f"File {rel(t, ws)}:\n```python\n{t.read_text(encoding='utf-8')}```"
                                    for t in targets + tests)
            tools, state = make_coding_tools(task.workspace, {rel(t, ws) for t in targets}, cmd)
            llm.agent(system_prompt=SYSTEM, query=QUERY.format(task=task_text, files=files, out=compact_test_output(out)) + multi,
                      tools=tools, model=model, max_iterations=3, max_tokens=max_out,
                      stop=lambda state=state: state["passed"])
            ok, out = run_tests(task, tests)  # проверяем сами, не веря агенту на слово
            if ok:
                passed = True
                return "DONE"
            if not llm.ledger.can_afford(plan.est_cost):
                break
        return None
    finally:
        if not passed:  # не оставляем испорченные файлы, в том числе при обрыве по бюджету
            for name, src in originals.items():
                write_file(task.workspace / name, src)
