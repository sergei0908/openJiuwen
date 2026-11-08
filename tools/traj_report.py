"""Отчёт по траекториям прогона: маршруты задач и точность прогноза расхода токенов.

    python tools/traj_report.py [/tmp/trajectories.jsonl]

Харнесс дописывает по строке JSON на задачу (см. TRAJ_PATH в harness/config.py), поэтому
файл копит несколько прогонов подряд. Чтобы смотреть один прогон, удаляйте файл перед запуском
или задайте свой путь: TRAJ_PATH=/tmp/run1.jsonl python run_agent.py ...
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROUTES = ("fast", "llm", "infeasible")


def load(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def report(rows):
    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("type") or "?"].append(r)

    print(f"Задач: {len(rows)}, потрачено токенов: {sum(r.get('cost', 0) for r in rows):.0f}\n")
    print(f"{'тип':15s} {'без LLM':>8s} {'LLM ok':>7s} {'LLM fail':>9s} {'отсеяно':>8s}")
    for t, rs in sorted(by_type.items()):
        n = defaultdict(int)
        for r in rs:
            route = r.get("route")
            if route == "llm":
                n["llm_ok" if r.get("ok") else "llm_fail"] += 1
            else:
                n[route] += 1
        print(f"{t:15s} {n['fast']:8d} {n['llm_ok']:7d} {n['llm_fail']:9d} {n['infeasible']:8d}")

    llm = [r for r in rows if r.get("route") == "llm" and "pred_cost" in r]
    if not llm:
        print("\nНет задач, решённых через LLM: прогноз сравнить не с чем.")
        return
    print("\nПрогноз расхода (только LLM-задачи)")
    print(f"{'тип':15s} {'n':>3s} {'прогноз':>8s} {'факт':>7s} {'MAE':>7s} {'факт/прогноз':>13s}")
    llm_by_type = defaultdict(list)
    for r in llm:
        llm_by_type[r["type"]].append(r)
    for t, rs in sorted(llm_by_type.items()) + [("ВСЕ", llm)]:
        pred = [r["pred_cost"] for r in rs]
        fact = [r["cost"] for r in rs]
        mae = mean([abs(p - f) for p, f in zip(pred, fact)])
        ratio = sum(fact) / sum(pred) if sum(pred) else 0.0
        print(f"{t:15s} {len(rs):3d} {mean(pred):8.0f} {mean(fact):7.0f} {mae:7.0f} {ratio:13.2f}")

    # базовый прогноз «среднее по типу» - с ним должна сравниваться любая модель оценщика
    base_err = []
    for rs in llm_by_type.values():
        avg = mean([r["cost"] for r in rs])
        base_err += [abs(r["cost"] - avg) for r in rs]
    print(f"\nMAE базового прогноза «среднее по типу» (на тех же данных): {mean(base_err):.0f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/trajectories.jsonl"
    if not Path(path).is_file():
        sys.exit(f"нет файла {path}: сначала запустите run_agent.py")
    report(load(path))


if __name__ == "__main__":
    main()
