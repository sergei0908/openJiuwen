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
from pathlib import Path

DRIVER = str(Path(__file__).with_name("apr_driver.py"))

B = ast
BIN_ALT = {
    B.Add: [B.Sub], B.Sub: [B.Add], B.Mult: [B.FloorDiv, B.Div, B.Add, B.Pow],
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
}
ATTR_ALT = {
    "upper": ["lower"], "lower": ["upper"], "startswith": ["endswith"], "endswith": ["startswith"],
    "lstrip": ["rstrip", "strip"], "rstrip": ["lstrip", "strip"], "strip": ["lstrip", "rstrip"],
    "keys": ["values"], "values": ["keys"], "append": ["extend"], "find": ["rfind"],
    "rfind": ["find"], "index": ["rindex"], "pop": ["popleft"], "title": ["capitalize"],
    "capitalize": ["title"], "isupper": ["islower"], "islower": ["isupper"],
}


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
            alts = [v + 1, v - 1] + ([1] if v == 0 else []) + ([0] if v == 1 else []) + \
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
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range" \
                and len(n.args) == 1:
            out.append((3, i, "range_shift", None))
        if isinstance(n, ast.Call) and len(n.args) == 1 and not n.keywords:
            fname = n.func.id if isinstance(n.func, ast.Name) else \
                (n.func.attr if isinstance(n.func, ast.Attribute) else "")
            if fname in ("sorted", "reversed", "list", "abs", "set", "int", "round"):
                out.append((3, i, "unwrap_call", None))
            if fname in ("join", "sum", "list", "len", "max", "min", "tuple"):
                for k, w in enumerate(("reversed", "sorted")):
                    out.append((3 + k, i, "wrap_arg", w))
        if isinstance(n, (ast.For, ast.comprehension)):
            for k, w in enumerate(("reversed", "sorted")):
                out.append((4 + k, i, "wrap_iter", w))
        elif isinstance(n, ast.keyword) and isinstance(n.value, ast.Constant) \
                and type(n.value.value) is bool:
            out.append((1, i, "kwflip", None))
    return out


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
        else:
            getattr(p, fld)[j] = new

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
        elif kind == "range_shift":
            n.args = [ast.BinOp(left=n.args[0], op=ast.Add(), right=ast.Constant(1))]
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


def _evaluate(ws, module, tests, sources, timeout):
    fd, cfgp = tempfile.mkstemp(suffix=".json", dir=os.environ.get("TMPDIR", "/tmp"))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"ws": str(ws), "module": module, "tests": [str(t) for t in tests],
                   "sources": sources, "per_case_timeout": 2.0}, f)
    try:
        p = subprocess.run([sys.executable, "-I", DRIVER, cfgp], capture_output=True, text=True, check=False,
                           timeout=timeout, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        return json.loads(p.stdout)
    except Exception:  # noqa: BLE001
        return None
    finally:
        os.unlink(cfgp)


def repair(ws: Path, target: Path, tests, max_rounds=5, time_budget=60):
    """Возвращает (fixed_src | None, partial_src, info). partial_src - лучшая частичная починка."""
    import time
    t0 = time.time()
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
        res = _evaluate(ws, module, tests, srcs, max(30, len(srcs) * 0.2))
        if not res:
            break
        good = []
        for m, r in zip(ok, res):
            ps = set(r["passed"])
            if r["err"] is None and ps >= cur_pass and ps > cur_pass:
                good.append((len(ps - cur_pass), -m[0], m, ps))
        if not good:
            break
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
