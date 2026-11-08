"""Автоматический ремонт программ (APR) мутациями AST - без LLM.

Классический подход generate-and-validate: порождаем одиночные правки «типичных багов»
(перевёрнутый оператор сравнения, += вместо -=, off-by-one, sorted/reversed, min/max ...),
прогоняем тесты и жадно комбинируем правки, которые чинят тесты и ничего не ломают.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DRIVER = str(Path(__file__).with_name("apr_driver.py"))

B = ast
BIN_ALT = {
    B.Add: [B.Sub, B.Mult], B.Sub: [B.Add], B.Mult: [B.FloorDiv, B.Div, B.Add, B.Pow],
    B.Div: [B.FloorDiv, B.Mult], B.FloorDiv: [B.Div, B.Mod, B.Mult], B.Mod: [B.FloorDiv],
    B.Pow: [B.Mult],
}
CMP_ALT = {
    B.Lt: [B.Gt, B.LtE, B.GtE], B.Gt: [B.Lt, B.GtE, B.LtE], B.LtE: [B.GtE, B.Lt, B.Gt],
    B.GtE: [B.LtE, B.Gt, B.Lt], B.Eq: [B.NotEq], B.NotEq: [B.Eq], B.In: [B.NotIn],
    B.NotIn: [B.In], B.Is: [B.IsNot], B.IsNot: [B.Is],
}
NAME_ALT = {
    "sorted": ["reversed", "list"], "reversed": ["sorted", "list"], "min": ["max"],
    "max": ["min"], "any": ["all"], "all": ["any"], "True": ["False"], "False": ["True"],
    "len": ["sum"], "sum": ["len"],
}
ATTR_ALT = {
    "upper": ["lower", "capitalize", "title"], "lower": ["upper", "capitalize", "title"],
    "startswith": ["endswith"], "endswith": ["startswith"],
    "lstrip": ["rstrip", "strip"], "rstrip": ["lstrip", "strip"], "strip": ["lstrip", "rstrip"],
    "keys": ["values"], "values": ["keys"], "append": ["extend"], "find": ["rfind"],
    "rfind": ["find"], "index": ["rindex"], "pop": ["popleft"], "title": ["capitalize", "lower", "upper"],
    "capitalize": ["title", "lower", "upper"], "isupper": ["islower"], "islower": ["isupper"],
}
NON_COMMUTATIVE = (B.Sub, B.Div, B.FloorDiv, B.Mod, B.Pow)
STR_METHODS = ("lower", "upper", "split", "join", "replace", "capitalize", "title", "startswith", "endswith")
VALUE_WRAPS = ("abs", "sorted", "set", "list", "len", "reversed")
MAX_PAIRS = 3000


def _mutations(tree):
    """Список (prio, idx, kind, payload). idx - индекс узла в list(ast.walk(tree))."""
    nodes = list(ast.walk(tree))
    out = []
    funcs_locals = {}
    for fn in [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        names = {a.arg for a in fn.args.args}
        names |= {t.id for t in ast.walk(fn) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
        for t in ast.walk(fn):
            funcs_locals[id(t)] = names
    for i, n in enumerate(nodes):
        if isinstance(n, (ast.BinOp, ast.AugAssign)):
            for k, alt in enumerate(BIN_ALT.get(type(n.op), [])):
                out.append((1 + k, i, "binop", alt))
            if isinstance(n, ast.BinOp) and isinstance(n.op, NON_COMMUTATIVE):
                out.append((3, i, "swap_operands", None))
            if isinstance(n, ast.BinOp) and isinstance(n.right, ast.BinOp):
                out.append((5, i, "rotate", None))
        elif isinstance(n, ast.Compare):
            for j, op in enumerate(n.ops):
                for k, alt in enumerate(CMP_ALT.get(type(op), [])):
                    out.append((1 + k, i, "cmp", (j, alt)))
        elif isinstance(n, ast.BoolOp):
            out.append((1, i, "boolop", ast.Or if isinstance(n.op, ast.And) else ast.And))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.Not, ast.USub)):
            out.append((2, i, "unwrap", None))
        elif isinstance(n, ast.Constant) and type(n.value) is bool:
            out.append((1, i, "const", not n.value))
        elif isinstance(n, ast.Constant) and type(n.value) is int:
            v = n.value
            alts = [v + 1, v - 1] + ([1] if v == 0 else []) + ([0, -1] if v == 1 else []) + \
                   ([-1] if v == 0 else []) + ([v * 2, v // 2] if v > 2 else [])
            for k, a in enumerate(dict.fromkeys(a for a in alts if a != v)):
                out.append((2 + k, i, "const", a))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) <= 3:
            for k, a in enumerate([x for x in (" ", "", ",", "-", "_") if x != n.value]):
                out.append((4 + k, i, "const", a))
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            for k, a in enumerate(NAME_ALT.get(n.id, [])):
                out.append((1 + k, i, "name", a))
            # подмена переменной на другую локальную (return best -> return x)
            loc = funcs_locals.get(id(n))
            if loc and n.id in loc:
                for a in sorted(loc - {n.id}):
                    out.append((5, i, "name", a))
        elif isinstance(n, ast.Attribute):
            for k, a in enumerate(ATTR_ALT.get(n.attr, [])):
                out.append((2 + k, i, "attr", a))
        elif isinstance(n, ast.Slice):
            for fld in ("lower", "upper"):
                sub = getattr(n, fld)
                if sub is not None and not isinstance(sub, ast.Constant):
                    out.append((2, i, "slice", (fld, ast.Add)))
                    out.append((2, i, "slice", (fld, ast.Sub)))
            if n.step is None and (n.lower is None) != (n.upper is None):
                out.append((3, i, "slice_flip", None))
                out.append((4, i, "slice_flip_neg", None))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range" \
                and 1 <= len(n.args) <= 2:
            out.append((3, i, "range_shift", ast.Add))
            out.append((3, i, "range_shift", ast.Sub))
            if len(n.args) == 1:
                out.append((4, i, "range_from_one", None))
        if isinstance(n, ast.Call) and len(n.args) == 1 and not n.keywords:
            fname = n.func.id if isinstance(n.func, ast.Name) else \
                (n.func.attr if isinstance(n.func, ast.Attribute) else "")
            if fname in ("sorted", "reversed", "list", "abs", "set", "int", "round"):
                out.append((3, i, "unwrap_call", None))
            if fname in ("join", "sum", "list", "len", "max", "min", "tuple"):
                for k, w in enumerate(("reversed", "sorted")):
                    out.append((3 + k, i, "wrap_arg", w))
            if fname in ("sorted", "list", "len", "sum"):
                out.append((5, i, "wrap_arg", "set"))
            if fname == "len" and isinstance(n.args[0], ast.Name):
                out.append((5, i, "arg_method", "split"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in STR_METHODS \
                and n.func.attr != "join":
            for k, meth in enumerate(("strip", "lower")):
                if meth != n.func.attr:
                    out.append((5 + k, i, "insert_method", meth))
        if isinstance(n, (ast.Return, ast.Assign)) and isinstance(n.value, (ast.Name, ast.Call, ast.BinOp)) \
                and (isinstance(n, ast.Return) or (len(n.targets) == 1 and isinstance(n.targets[0], ast.Name))):
            for k, w in enumerate(VALUE_WRAPS):
                out.append((5 + k // 2, i, "wrap_value", w))
            if isinstance(n.value, ast.Name):
                out.append((5, i, "reverse_slice", None))
        if isinstance(n, ast.Return) and isinstance(n.value, (ast.Name, ast.Call, ast.Compare)):
            out.append((5, i, "negate_value", None))
        if isinstance(n, (ast.If, ast.While)):
            out.append((3, i, "negate_test", None))
        if isinstance(n, (ast.Break, ast.Continue)):
            out.append((3, i, "break_continue", None))
        if isinstance(n, (ast.For, ast.While)):
            if isinstance(n.body[-1], ast.Return):
                out.append((3, i, "dedent_return", None))
            for j in range(len(n.body) - 1):
                out.append((5, i, "swap_stmts", j))
            for j, st in enumerate(n.body):
                if isinstance(st, (ast.Assign, ast.AugAssign)) and len(n.body) > 1:
                    out.append((7, i, "delete_stmt", j))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and not any(isinstance(t, ast.Return) for t in ast.walk(n)):
            names = sorted({t.id for t in ast.walk(n) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)})
            for a in names:
                out.append((1, i, "add_return", a))
        if isinstance(n, (ast.For, ast.comprehension)):
            for k, w in enumerate(("reversed", "sorted")):
                out.append((4 + k, i, "wrap_iter", w))
        elif isinstance(n, ast.keyword) and isinstance(n.value, ast.Constant) \
                and type(n.value.value) is bool:
            out.append((1, i, "kwflip", None))
    return out


def _rotate(n):
    while isinstance(n.right, ast.BinOp):
        inner = n.right
        n.left, n.right, n.op = ast.BinOp(left=n.left, op=n.op, right=inner.left), inner.right, inner.op
    if isinstance(n.left, ast.BinOp):
        _rotate(n.left)


def _apply(src, muts):
    tree = ast.parse(src)
    nodes = list(ast.walk(tree))
    parent = {}
    for p in nodes:
        for fld, val in ast.iter_fields(p):
            if isinstance(val, ast.AST):
                parent[id(val)] = (p, fld, None)
            elif isinstance(val, list):
                for j, x in enumerate(val):
                    if isinstance(x, ast.AST):
                        parent[id(x)] = (p, fld, j)

    def replace(node, new):
        p, fld, j = parent[id(node)]
        if j is None:
            setattr(p, fld, new)
        else:  # индекс мог сдвинуться от предыдущих правок
            lst = getattr(p, fld)
            lst[next(k for k, x in enumerate(lst) if x is node)] = new

    def call(func, *args):
        return ast.Call(func=ast.Name(id=func, ctx=ast.Load()), args=list(args), keywords=[])

    for _, i, kind, pl in muts:
        n = nodes[i]
        if kind == "binop":
            n.op = pl()
        elif kind == "cmp":
            j, alt = pl
            n.ops[j] = alt()
        elif kind == "boolop":
            n.op = pl()
        elif kind == "unwrap":
            replace(n, n.operand)
        elif kind == "const":
            n.value = pl
        elif kind == "name":
            n.id = pl
        elif kind == "attr":
            n.attr = pl
        elif kind == "slice":
            fld, op = pl
            setattr(n, fld, ast.BinOp(left=getattr(n, fld), op=op(), right=ast.Constant(1)))
        elif kind == "slice_flip":
            n.lower, n.upper = n.upper, n.lower
        elif kind == "slice_flip_neg":
            n.lower, n.upper = n.upper, n.lower
            fld = "lower" if n.lower is not None else "upper"
            setattr(n, fld, ast.UnaryOp(op=ast.USub(), operand=getattr(n, fld)))
        elif kind == "range_shift":
            n.args[-1] = ast.BinOp(left=n.args[-1], op=pl(), right=ast.Constant(1))
        elif kind == "range_from_one":
            n.args = [ast.Constant(1), ast.BinOp(left=n.args[0], op=ast.Add(), right=ast.Constant(1))]
        elif kind == "swap_operands":
            n.left, n.right = n.right, n.left
        elif kind == "rotate":  # a - b * c -> (a - b) * c: забытые скобки
            _rotate(n)
        elif kind == "arg_method":
            n.args = [ast.Call(func=ast.Attribute(value=n.args[0], attr=pl, ctx=ast.Load()), args=[], keywords=[])]
        elif kind == "insert_method":
            n.func.value = ast.Call(func=ast.Attribute(value=n.func.value, attr=pl, ctx=ast.Load()), args=[], keywords=[])
        elif kind == "wrap_value":
            n.value = call(pl, n.value)
            if pl == "reversed":
                n.value = call("list", n.value)
        elif kind == "reverse_slice":
            n.value = ast.Subscript(value=n.value, ctx=ast.Load(),
                                    slice=ast.Slice(step=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))))
        elif kind == "negate_value":
            n.value = ast.UnaryOp(op=ast.Not(), operand=n.value)
        elif kind == "negate_test":
            n.test = ast.UnaryOp(op=ast.Not(), operand=n.test)
        elif kind == "break_continue":
            replace(n, ast.Continue() if isinstance(n, ast.Break) else ast.Break())
        elif kind == "dedent_return":
            st = n.body.pop()
            if not n.body:
                n.body.append(ast.Pass())
            lst = parent[id(n)]
            siblings = getattr(lst[0], lst[1])
            siblings.insert(next(k for k, x in enumerate(siblings) if x is n) + 1, st)
        elif kind == "swap_stmts":
            n.body[pl], n.body[pl + 1] = n.body[pl + 1], n.body[pl]
        elif kind == "delete_stmt":
            del n.body[pl]
        elif kind == "add_return":
            n.body.append(ast.Return(value=ast.Name(id=pl, ctx=ast.Load())))
        elif kind == "kwflip":
            n.value.value = not n.value.value
        elif kind == "unwrap_call":
            replace(n, n.args[0])
        elif kind == "wrap_arg":
            n.args = [ast.Call(func=ast.Name(id=pl, ctx=ast.Load()), args=[n.args[0]], keywords=[])]
        elif kind == "wrap_iter":
            n.iter = ast.Call(func=ast.Name(id=pl, ctx=ast.Load()), args=[n.iter], keywords=[])
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _functions(tree):
    """Индекс узла в list(ast.walk(tree)) -> функция, которой он принадлежит."""
    owner = {}
    ids = {id(n): i for i, n in enumerate(ast.walk(tree))}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for t in ast.walk(fn):
                owner[ids[id(t)]] = fn.name
    return owner


def _pair_search(src0, tree, cands, applied, ws, module, tests, cur_pass, deadline):
    """Один тест падает из-за двух ошибок сразу: одиночная правка его не чинит, нужна пара."""
    owner = _functions(tree)
    pool = [m for m in cands if m[0] <= 5 and m[1] in owner]
    pairs = [(a, b) for k, a in enumerate(pool) for b in pool[k + 1:]
             if a[1] != b[1] and owner[a[1]] == owner[b[1]]]
    pairs.sort(key=lambda ab: ab[0][0] + ab[1][0])
    for start in range(0, min(len(pairs), MAX_PAIRS), 200):
        if time.time() > deadline:
            return None
        batch, srcs = [], []
        for a, b in pairs[start:start + 200]:
            try:
                srcs.append(_apply(src0, applied + [a, b]))
                batch.append((a, b))
            except Exception:  # noqa: BLE001, S112 - мутант, который не собирается, просто пропускаем
                continue
        res = _evaluate(ws, module, tests, srcs, max(5, deadline - time.time()), deadline) or []
        for pair, r in zip(batch, res):
            ps = set(r["passed"])
            if r["err"] is None and ps > cur_pass:
                return pair, ps
    return None


def _evaluate(ws, module, tests, sources, timeout, deadline=None):
    fd, cfgp = tempfile.mkstemp(suffix=".json", dir=os.environ.get("TMPDIR", "/tmp"))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"ws": str(ws), "module": module, "tests": [str(t) for t in tests],
                   "sources": sources, "per_case_timeout": 2.0, "deadline": deadline}, f)
    try:
        p = subprocess.run([sys.executable, "-I", DRIVER, cfgp], capture_output=True, text=True, check=False,
                           timeout=timeout, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        return json.loads(p.stdout)
    except Exception:  # noqa: BLE001
        return None
    finally:
        os.unlink(cfgp)


def repair(ws: Path, target: Path, tests, max_rounds=6, time_budget=10):
    """Возвращает (fixed_src | None, partial_src, info). partial_src - лучшая частичная починка."""
    t0 = time.time()
    deadline = t0 + time_budget
    src0 = target.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src0)
    except SyntaxError:
        return None, src0, {"reason": "syntax"}
    module = target.stem
    cands = sorted(_mutations(tree), key=lambda m: m[0])
    base = _evaluate(ws, module, tests, [src0], 30)
    if not base:
        return None, src0, {"reason": "driver"}
    all_t = set(base[0]["all"])
    if not all_t:
        return None, src0, {"reason": "no_tests"}
    applied = []
    cur_pass = set(base[0]["passed"])
    info = {"mutants": len(cands), "rounds": 0, "start_pass": len(cur_pass), "total": len(all_t)}
    if cur_pass == all_t:
        return src0, src0, dict(info, reason="already_ok")
    used = set()
    for rnd in range(max_rounds):
        if time.time() - t0 > time_budget:
            break
        info["rounds"] = rnd + 1
        pool = [m for m in cands if m[1] not in used]
        srcs, ok = [], []
        for m in pool:
            try:
                srcs.append(_apply(src0, applied + [m]))
                ok.append(m)
            except Exception:  # noqa: BLE001, S110 - мутант, который не собирается, просто пропускаем
                pass
        res = _evaluate(ws, module, tests, srcs, max(5, time_budget), deadline)
        if not res:
            break
        good = []
        for m, r in zip(ok, res):
            ps = set(r["passed"])
            if r["err"] is None and ps >= cur_pass and ps > cur_pass:
                good.append((len(ps - cur_pass), -m[0], m, ps))
        if not good:
            found = _pair_search(src0, tree, pool, applied, ws, module, tests, cur_pass, deadline)
            if not found:
                break
            applied = applied + list(found[0])
            cur_pass = found[1]
            used |= {m[1] for m in applied}
            if cur_pass == all_t:
                break
            continue
        good.sort(key=lambda g: (g[0], g[1]), reverse=True)
        chosen, covered = [], set()
        for gain, _, m, ps in good:
            new = ps - cur_pass
            if new - covered and m[1] not in {c[1] for c in chosen}:
                chosen.append(m)
                covered |= new
        trial = applied + chosen
        r = _evaluate(ws, module, tests, [_apply(src0, trial)], 30)
        if r and r[0]["err"] is None and set(r[0]["passed"]) >= cur_pass | covered:
            applied = trial
            cur_pass = set(r[0]["passed"])
        else:
            applied = applied + [good[0][2]]
            cur_pass = good[0][3]
        used |= {m[1] for m in applied}
        if cur_pass == all_t:
            break
    partial = _apply(src0, applied) if applied else src0
    info.update(applied=len(applied), end_pass=len(cur_pass))
    return (partial if cur_pass == all_t else None), partial, info
