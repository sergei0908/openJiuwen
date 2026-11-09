"""Кривая бюджета: сколько баллов набирает харнесс при разном бюджете и разных способах выбирать задачи.

    python tools/budget_sim.py traces/seed_101.jsonl --model harness/cost_model.json
    python tools/budget_sim.py traces/*.jsonl --budgets 500,1000,2000,5000 --png budget_curve.png

Ни одного токена: прогон «проигрывается» по трассе, где для каждой задачи уже известны награда,
фактический расход и успех. Способы выбора:
  random - случайный порядок (среднее по --shuffles перемешиваниям);
  reward - сначала задачи с большей наградой;
  est    - как в харнессе: «награда × вероятность успеха ÷ прогноз»; прогноз - модель из --model
           (или pred_cost из трассы), с онлайн-калибровкой, как в runner;
  oracle - знает настоящий расход и успех каждой задачи (верхняя граница для жадного выбора).
Правила как в runner: задачу берём, если её прогноз помещается в остаток бюджета; если фактический
расход больше остатка, задача обрывается без награды и бюджет кончается.

Для честной оценки трассы должны быть из seed, на которых модель НЕ обучалась.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.estimator import Estimator
from harness.solvers.base import Plan

STRATEGIES = ("random", "reward", "est", "oracle")


def load(paths):
    tasks = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("route") not in ("llm", "fast"):
                continue  # нерешаемые задачи отсеиваются в фазе A и не стоят ни токена
            tasks.append({"id": f"{p}:{r['task_id']}", "type": r["type"], "reward": float(r.get("reward", 1)),
                          "cost": float(r.get("cost") or 0), "ok": bool(r.get("correct", r.get("route") == "fast" or r.get("ok"))),
                          "pred_cost": float(r.get("pred_cost") or 300), "features": r.get("features") or {}})
    return tasks


def _load_model(path):
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw.get("types", raw)


def simulate(tasks, budget, strategy, model_path=None, seed=0):
    """Возвращает (баллы, потрачено). Бесплатные задачи (cost 0) засчитываются всегда, как в фазе A."""
    score = sum(t["reward"] for t in tasks if t["cost"] == 0 and t["ok"])
    queue = [t for t in tasks if t["cost"] > 0]
    est = Estimator()
    est.coef = _load_model(model_path)  # модель только из --model, не из harness/cost_model.json
    rank = {t["id"]: i for i, t in enumerate(random.Random(seed).sample(queue, len(queue)))}
    left = float(budget)

    def pred(t):
        if strategy == "oracle":
            return t["cost"], float(t["ok"])
        plan = Plan(est_cost=t["pred_cost"])
        plan.notes["features"] = t["features"]
        if t["features"].get("generic"):
            plan.notes["generic"] = True
        return est.predict(t["type"], plan)

    def priority(t, p, p_ok):
        if strategy == "reward":
            return t["reward"] - 1e-6 * p
        if strategy == "random":
            return -rank[t["id"]]
        return t["reward"] * p_ok / max(p, 1.0)

    while queue and left > 0:
        scored = [(priority(t, *pr), t, pr[0]) for t in queue for pr in [pred(t)]]
        scored.sort(key=lambda x: x[0], reverse=True)
        pick = next(((t, p) for _, t, p in scored if p <= left), None)
        if pick is None:
            break
        t, p = pick
        queue.remove(t)
        if t["cost"] > left:  # оборвалась на середине: бюджет кончился, награды нет
            left = 0
            break
        left -= t["cost"]
        est.observe(t["type"], p, t["cost"])
        if t["ok"]:
            score += t["reward"]
    return score, budget - left


def curve(tasks, budgets, model_path=None, shuffles=50):
    table = {}
    for s in STRATEGIES:
        row = []
        for b in budgets:
            if s == "random":
                runs = [simulate(tasks, b, s, model_path, seed=k)[0] for k in range(shuffles)]
                row.append(sum(runs) / len(runs))
            else:
                row.append(simulate(tasks, b, s, model_path)[0])
        table[s] = row
    return table


def plot(table, budgets, path, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    styles = {"oracle": ("#888888", "--"), "est": ("#1f6feb", "-"), "reward": ("#d29922", "-"),
              "random": ("#cf222e", ":")}
    names = {"oracle": "оракул (знает расход)", "est": "наш оценщик", "reward": "по баллам",
             "random": "случайно"}
    for s in ("oracle", "est", "reward", "random"):
        c, ls = styles[s]
        ax.plot(budgets, table[s], marker="o", color=c, linestyle=ls, label=names[s])
    ax.axhline(total, color="#cccccc", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("бюджет, токены")
    ax.set_ylabel("баллы")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--model", help="cost_model.json; без него est берёт pred_cost из трассы")
    ap.add_argument("--budgets", default="250,500,1000,2000,5000,10000")
    ap.add_argument("--shuffles", type=int, default=50)
    ap.add_argument("--png", help="сохранить график")
    a = ap.parse_args(argv)
    tasks = load(a.traces)
    if not tasks:
        print("в трассах нет задач", file=sys.stderr)
        return 1
    budgets = [int(x) for x in a.budgets.split(",")]
    table = curve(tasks, budgets, a.model, a.shuffles)
    total = sum(t["reward"] for t in tasks if t["ok"])
    need = sum(t["cost"] for t in tasks)
    print(f"задач: {len(tasks)}, баллов доступно: {total:.0f}, расход на всё: {need:.0f}")
    print(f"{'бюджет':>8} " + " ".join(f"{s:>8}" for s in STRATEGIES) + "   est/oracle")
    for i, b in enumerate(budgets):
        vals = [table[s][i] for s in STRATEGIES]
        share = table["est"][i] / table["oracle"][i] if table["oracle"][i] else 1.0
        print(f"{b:8d} " + " ".join(f"{v:8.1f}" for v in vals) + f"   {share:9.0%}")
    if a.png:
        plot(table, budgets, a.png, total)
        print("график:", a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
