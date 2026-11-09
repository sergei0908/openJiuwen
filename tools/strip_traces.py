"""Очищает трассы для обучения оценщика: только числа, без ответов модели.

    python tools/strip_traces.py raw/seed_101.jsonl --tasks-dir /tmp/tr_101 --group seed_101 \
        --out data/traces/seed_101.jsonl

Оставляет task_id, type, reward, route, pred_cost, cost, ok, calls (модель и числа токенов),
features, secs и добавляет:
  correct - ответ верный (сверка с expected через tools/score.py; нужен --tasks-dir после прогона);
  group   - имя группы для кросс-валидации (обычно seed генератора).
Ответы, тексты ошибок и всё остальное выбрасывает: правила NeuralDeep запрещают обучать модели на
ответах нейросети, а оценщику нужны только размеры задач и расход.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import check

KEEP = ("task_id", "type", "reward", "route", "pred_cost", "cost", "ok", "features", "secs")
ERR_KINDS = ("BudgetExhausted", "TaskBudgetExceeded")


def clean(rec, tasks_dir=None, group=None):
    out = {k: rec[k] for k in KEEP if k in rec}
    out["calls"] = [{k: c.get(k) for k in ("model", "prompt_tokens", "completion_tokens", "cost")}
                    for c in rec.get("calls") or []]
    err = rec.get("err") or ""
    if err.split(":")[0] in ERR_KINDS:
        out["err"] = err.split(":")[0]
    elif err:
        out["err"] = "other"
    if group:
        out["group"] = group
    if tasks_dir is not None:
        d = Path(tasks_dir) / rec["task_id"]
        tj = d / "task.json"
        if tj.is_file():
            j = json.loads(tj.read_text(encoding="utf-8"))
            out["correct"] = bool(check(d, j)[0]) if not j.get("_infeasible") else rec.get("route") == "infeasible"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks-dir")
    ap.add_argument("--group")
    a = ap.parse_args(argv)
    lines = []
    for line in Path(a.raw).read_text(encoding="utf-8").splitlines():
        try:
            lines.append(json.dumps(clean(json.loads(line), a.tasks_dir, a.group), ensure_ascii=False))
        except (ValueError, KeyError):
            continue
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"{len(lines)} записей -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
