"""Сравнение моделей прокси на одном наборе задач: точность и расход по типам.

    python tools/compare_llms.py --tasks ~/burn-rate-data/m201 \
        --models qwen3.8-27b-noreason,qwen3.6-35b-a3b-noreason [--budget 30000]
    python tools/compare_llms.py --list            # какие модели отдаёт прокси

Каждая модель решает ВСЕ задачи сама (FAST_PATH=0; MODEL_FAST, MODEL_STRONG и запасные - все равны
ей, ALLOW_REASONING=0), на отдельной копии набора. Итог - таблица: баллы, токены, время, и по типам
«верно / расход на задачу». Ключ берётся из окружения (LLM_BASE_URL, LLM_API_KEY).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ("coding", "data_analysis", "math", "search")


def list_models():
    base, key = os.environ.get("LLM_BASE_URL", "").rstrip("/"), os.environ.get("LLM_API_KEY", "")
    req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    for m in data.get("data", []):
        print(m.get("id"))


def run_model(model, tasks, work, budget):
    d = work / re.sub(r"[^\w.-]", "_", model)
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(tasks, d)
    for a in d.rglob("answer.txt"):
        a.unlink()
    traj = work / f"{d.name}.jsonl"
    traj.unlink(missing_ok=True)
    env = dict(os.environ, FAST_PATH="0", ALLOW_REASONING="0", TRAJ_PATH=str(traj),
               MODEL_FAST=model, MODEL_STRONG=model, MODEL_ALT=model, MODEL_STRONG_ALT=model)
    if budget:
        env["BUDGET"] = str(budget)
    p = subprocess.run([sys.executable, str(ROOT / "run_agent.py"), "--tasks-dir", str(d)], env=env,
                       capture_output=True, text=True, check=False)
    m = re.findall(r"SUMMARY (\{.*\})", p.stdout + p.stderr)
    summary = json.loads(m[-1]) if m else {}
    s = subprocess.run([sys.executable, str(ROOT / "tools" / "score.py"), str(d)], capture_output=True,
                       text=True, check=False).stdout
    score = re.search(r"score (\d+) / (\d+)", s)
    by_type = {t: re.search(rf"{t}\s+solved (\d+)/(\d+)", s) for t in TYPES}
    cost = {t: [] for t in TYPES}
    if traj.exists():
        for line in traj.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("route") == "llm" and r.get("type") in cost:
                cost[r["type"]].append(float(r.get("cost") or 0))
    return {"model": model, "score": score.groups() if score else ("?", "?"),
            "tokens": summary.get("tokens_spent"), "secs": summary.get("elapsed_s"),
            "fail": summary.get("llm_fail"),
            "types": {t: ((by_type[t].groups() if by_type[t] else ("-", "-")),
                          sum(cost[t]) / len(cost[t]) if cost[t] else 0.0) for t in TYPES}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks")
    ap.add_argument("--models", default="qwen3.8-27b-noreason,qwen3.6-35b-a3b-noreason")
    ap.add_argument("--budget", type=int, default=0, help="предел токенов на прогон одной модели")
    ap.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "burnrate_llm_cmp"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if not os.environ.get("LLM_BASE_URL"):
        print("export LLM_BASE_URL and LLM_API_KEY first", file=sys.stderr)
        return 2
    if a.list:
        list_models()
        return 0
    if not a.tasks:
        ap.error("--tasks is required")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        print(f"... {model}", flush=True)
        rows.append(run_model(model, Path(a.tasks).expanduser(), work, a.budget))

    print(f"\n{'модель':28} {'баллы':>9} {'токены':>8} {'сек':>6} {'провал':>6}")
    for r in rows:
        print(f"{r['model']:28} {r['score'][0]:>4}/{r['score'][1]:<4} {r['tokens'] or 0:8.0f} "
              f"{r['secs'] or 0:6.0f} {r['fail'] or 0:6}")
    print(f"\n{'модель':28} " + " ".join(f"{t:>20}" for t in TYPES))
    print(f"{'':28} " + " ".join(f"{'верно  токены/задачу':>20}" for _ in TYPES))
    for r in rows:
        cells = []
        for t in TYPES:
            (ok, total), c = r["types"][t]
            cells.append(f"{ok + '/' + total:>9} {c:10.0f}")
        print(f"{r['model']:28} " + " ".join(f"{x:>20}" for x in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
