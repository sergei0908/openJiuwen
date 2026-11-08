"""Локальная оценка: сверяет answer.txt с expected, для coding прогоняет тесты.

    python3 tools/score.py /tmp/train [--orig /tmp/train_orig]  # orig - копия до запуска (для тестов)
"""
import json
import subprocess
import sys
from pathlib import Path


def check(d: Path, j: dict):
    ans_p = d / "answer.txt"
    if not ans_p.is_file():
        return False, "no_answer"
    ans = ans_p.read_text().strip()
    exp, chk = str(j.get("expected", "")), j.get("check")
    if chk == "unittest":
        ws = d / "workspace"
        tests = sorted(ws.glob("test*.py"))
        p = subprocess.run([sys.executable, tests[0].name], cwd=ws, capture_output=True, text=True, timeout=60, check=False)
        return (p.returncode == 0 and ans == "DONE"), "tests_fail" if p.returncode else "ok"
    if chk == "exact_int":
        try:
            return int(float(ans)) == int(exp), ans
        except ValueError:
            return False, ans
    return ans.strip().lower() == exp.strip().lower(), ans


def main():
    root = Path(sys.argv[1])
    tot = got = 0.0
    by = {}
    wrong = []
    extra = {}
    for d in sorted(root.iterdir()):
        f = d / "task.json"
        if not f.is_file():
            continue
        j = json.loads(f.read_text())
        ok, info = check(d, j)
        if j.get("_infeasible"):
            ok = False
            if (d / "answer.txt").is_file():
                extra.setdefault(j.get("task_type"), []).append(d.name)
        rw = float(j.get("reward", 1))
        tot += 0 if j.get("_infeasible") else rw
        got += rw if ok else 0
        t = j.get("task_type")
        b = by.setdefault(t, [0, 0, 0])
        b[0] += 1
        b[1] += ok
        b[2] += 0 if j.get("_infeasible") else 1
        if not ok and not j.get("_infeasible"):
            wrong.append((d.name, j.get("expected"), info))
    print(f"score {got:.0f} / {tot:.0f} achievable")
    for t, (n, ok, feas) in sorted(by.items()):
        print(f"  {t:15s} solved {ok}/{feas} feasible ({n} total)")
    for w in wrong[:15]:
        print("  MISS", *w)
    for t, names in sorted(extra.items()):
        print(f"  WARNING answer.txt у невыполнимых задач ({t}): {len(names)}: {' '.join(names[:10])}")


if __name__ == "__main__":
    main()
