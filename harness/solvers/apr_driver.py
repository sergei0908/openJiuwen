"""Запускается в отдельном процессе: прогоняет набор вариантов модуля через тесты in-process.

Вход: JSON {ws, module, tests: [paths], sources: [src, ...], per_case_timeout}
Выход (stdout): JSON [{"passed": [...], "all": [...], "err": str|null}, ...]
"""
import importlib.util
import inspect
import io
import json
import os
import signal
import sys
import time
import types
import unittest


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _ids(suite):
    out = []
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            out.extend(_ids(t))
        else:
            out.append(t)
    return out


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        cfg = json.load(f)
    ws, modname, tests = cfg["ws"], cfg["module"], cfg["tests"]
    limit = float(cfg.get("per_case_timeout", 2.0))
    os.chdir(ws)
    sys.path.insert(0, ws)
    sys.dont_write_bytecode = True
    signal.signal(signal.SIGALRM, _alarm)
    real_stdout, real_stderr = sys.stdout, sys.stderr
    results = []
    deadline = cfg.get("deadline")
    for src in cfg["sources"]:
        if deadline and time.time() > deadline:  # результатов меньше, чем вариантов: это допустимо
            break
        res = {"passed": [], "all": [], "err": None}
        sys.stdout = sys.stderr = io.StringIO()
        try:
            signal.setitimer(signal.ITIMER_REAL, limit)
            for k in list(sys.modules):
                if k == modname or k.startswith("_aprtest_"):
                    del sys.modules[k]
            mod = types.ModuleType(modname)
            mod.__file__ = os.path.join(ws, modname + ".py")
            exec(compile(src, mod.__file__, "exec"), mod.__dict__)  # noqa: S102 - запуск варианта кода задачи под тестами
            sys.modules[modname] = mod
            for i, tp in enumerate(tests):
                spec = importlib.util.spec_from_file_location(f"_aprtest_{i}", tp)
                tm = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tm)
                suite = unittest.defaultTestLoader.loadTestsFromModule(tm)
                for fname, fn in vars(tm).items():  # тесты в стиле pytest: функции test_* с assert
                    if fname.startswith("test") and inspect.isfunction(fn) and fn.__module__ == tm.__name__ \
                            and not inspect.signature(fn).parameters:
                        suite.addTest(unittest.FunctionTestCase(fn))
                cases = _ids(suite)
                r = unittest.TestResult()
                suite.run(r)
                bad = {t.id() for t, _ in r.failures + r.errors}
                bad |= {t.id() for t, _ in r.unexpectedSuccesses} if hasattr(r, "unexpectedSuccesses") else set()
                for c in cases:
                    res["all"].append(c.id())
                    if c.id() not in bad:
                        res["passed"].append(c.id())
        except BaseException as e:  # noqa: BLE001 - включая _Timeout и SystemExit
            res["err"] = type(e).__name__
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            sys.stdout, sys.stderr = real_stdout, real_stderr
        results.append(res)
    sys.stdout.write(json.dumps(results))


if __name__ == "__main__":
    main()
