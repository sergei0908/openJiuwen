"""Большая проверка «как у жюри»: несколько наборов по 100 задач в обычном режиме.

    python tools/final_eval.py --out ~/burn-rate-data/final        # 8 наборов, ~25-40 тысяч токенов
    python tools/final_eval.py --out /tmp/f --sets 2                 # быстро, 2 набора

Каждый набор - 100 новых задач генератора с разной долей непривычных формулировок и сложности
(seed 501 и дальше - их не было ни в обучении оценщика, ни в других замерах). Харнесс работает ровно как
у жюри (FAST_PATH=1, BUDGET=100000), отдельно на каждом наборе. Итог:
  - баллы и доля от возможного, по типам задач, причины потерь (MISS);
  - токены: всего и на набор, сколько задач решено без модели;
  - точность оценщика: ошибка прогноза на задачу (MAE) против «среднего по типу» и ошибка прогноза
    суммарного расхода всего прогона.
Ключ берётся из окружения (LLM_BASE_URL, LLM_API_KEY).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ("coding", "data_analysis", "math", "search")
# (novel, spread): от «как примеры» до «почти всё непривычное»
PROFILES = [(0.0, 0.0), (0.3, 0.0), (0.3, 1.0), (0.5, 1.0), (0.8, 1.0), (0.5, 0.5), (1.0, 1.0), (0.3, 1.0)]


def run_set(i, seed, novel, spread, out):
    d = out / f"set_{seed}"
    if d.exists():
        shutil.rmtree(d)
    subprocess.run([sys.executable, str(ROOT / "tools" / "gen_tasks.py"), "--out", str(d), "--n", "100",
                    "--seed", str(seed), "--novel", str(novel), "--spread", str(spread), "--infeasible", "0.1"],
                   check=True, capture_output=True)
    traj = out / f"set_{seed}.jsonl"
    traj.unlink(missing_ok=True)
    env = dict(os.environ, FAST_PATH="1", BUDGET=os.environ.get("BUDGET", "100000"), TRAJ_PATH=str(traj))
    p = subprocess.run([sys.executable, str(ROOT / "run_agent.py"), "--tasks-dir", str(d)], env=env,
                       capture_output=True, text=True, check=False)
    (out / f"set_{seed}.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    m = re.findall(r"SUMMARY (\{.*\})", p.stdout + p.stderr)
    summary = json.loads(m[-1]) if m else {}
    s = subprocess.run([sys.executable, str(ROOT / "tools" / "score.py"), str(d)], capture_output=True, text=True,
                       check=False).stdout
    (out / f"set_{seed}.score").write_text(s, encoding="utf-8")
    sc = re.search(r"score (\d+) / (\d+)", s)
    types = {t: tuple(map(int, mm.groups())) for t in TYPES
             if (mm := re.search(rf"{t}\s+solved (\d+)/(\d+)", s))}
    misses = Counter()
    for line in s.splitlines():
        if line.strip().startswith("MISS"):
            parts = line.split()
            reason = parts[-1] if parts[-1] in ("no_answer", "overfit", "tests_fail") else "wrong"
            misses[(parts[1].rsplit("_", 1)[0], reason)] += 1
    recs = [json.loads(x) for x in traj.read_text(encoding="utf-8").splitlines()] if traj.exists() else []
    llm = [r for r in recs if r.get("route") == "llm" and r.get("cost")]
    return {"seed": seed, "novel": novel, "spread": spread,
            "score": (int(sc.group(1)), int(sc.group(2))) if sc else (0, 0),
            "tokens": float(summary.get("tokens_spent") or 0), "secs": float(summary.get("elapsed_s") or 0),
            "fast": summary.get("fast", 0), "llm": len(llm), "infeasible": summary.get("skipped_infeasible", 0),
            "types": types, "misses": misses,
            "pred": [(r["type"], float(r.get("pred_cost") or 0), float(r["cost"])) for r in llm]}


def mae(pairs):
    return sum(abs(p - f) for p, f in pairs) / len(pairs) if pairs else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", type=int, default=8)
    ap.add_argument("--seed", type=int, default=501)
    a = ap.parse_args()
    if not os.environ.get("LLM_BASE_URL"):
        print("export LLM_BASE_URL and LLM_API_KEY first", file=sys.stderr)
        return 2
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.sets):
        novel, spread = PROFILES[i % len(PROFILES)]
        seed = a.seed + i
        print(f"... набор {i + 1}/{a.sets}: seed {seed}, novel {novel}, spread {spread}", flush=True)
        rows.append(run_set(i, seed, novel, spread, out))

    print(f"\n{'seed':>5} {'novel':>5} {'spread':>6} {'баллы':>11} {'%':>6} {'токены':>8} {'без LLM':>8} "
          f"{'агент':>6} {'отсеяно':>8} {'сек':>5}")
    for r in rows:
        got, can = r["score"]
        print(f"{r['seed']:5d} {r['novel']:5.1f} {r['spread']:6.1f} {got:5d}/{can:<5d} {100 * got / max(can, 1):5.1f}% "
              f"{r['tokens']:8.0f} {r['fast']:8d} {r['llm']:6d} {r['infeasible']:8d} {r['secs']:5.0f}")
    got = sum(r["score"][0] for r in rows)
    can = sum(r["score"][1] for r in rows)
    tok = sum(r["tokens"] for r in rows)
    print(f"{'ВСЕГО':>18} {got:5d}/{can:<5d} {100 * got / max(can, 1):5.1f}% {tok:8.0f}  "
          f"(в среднем {tok / len(rows):.0f} токенов на 100 задач — {tok / len(rows) / 1000:.1f}% бюджета 100k)")

    print("\nпо типам задач (решено / решаемых):")
    for t in TYPES:
        s = sum(r["types"].get(t, (0, 0))[0] for r in rows)
        n = sum(r["types"].get(t, (0, 0))[1] for r in rows)
        print(f"  {t:14} {s:4d}/{n:<4d} {100 * s / max(n, 1):5.1f}%")
    misses = sum((r["misses"] for r in rows), Counter())
    if misses:
        print("\nпотери (тип, причина): " + ", ".join(f"{t} {why} ×{k}" for (t, why), k in misses.most_common()))

    pred = [x for r in rows for x in r["pred"]]
    if pred:
        by_type = defaultdict(list)
        for t, p, f in pred:
            by_type[t].append(f)
        mean = {t: sum(v) / len(v) for t, v in by_type.items()}
        ours = mae([(p, f) for _, p, f in pred])
        base = mae([(mean[t], f) for t, _, f in pred])
        print(f"\nоценщик: {len(pred)} задач через агента")
        print(f"  ошибка на задачу (MAE): наш прогноз {ours:.1f}, «среднее по типу» {base:.1f} "
              f"(среднее посчитано по этим же задачам — ему это на руку)")
        for t in TYPES:
            pt = [(p, f) for tt, p, f in pred if tt == t]
            if pt:
                print(f"  {t:14} n={len(pt):3d}  прогноз {sum(p for p, _ in pt) / len(pt):6.0f}  "
                      f"факт {sum(f for _, f in pt) / len(pt):6.0f}  MAE {mae(pt):5.1f}")
        sp, sf = sum(p for _, p, _ in pred), sum(f for _, _, f in pred)
        print(f"  суммарный расход агента: прогноз {sp:.0f}, факт {sf:.0f}, ошибка {100 * (sp - sf) / sf:+.1f}%")
    print(f"\nлоги, ответы и трассы: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
