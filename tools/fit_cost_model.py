"""Обучает оценщик расхода по трассам и пишет harness/cost_model.json.

    python tools/fit_cost_model.py data/traces/*.jsonl            # отчёт + запись модели
    python tools/fit_cost_model.py traces/*.jsonl --dry-run       # только отчёт

Берём из трасс задачи, решённые через LLM (route == "llm"), их признаки (features) и фактический
расход (cost). Для каждого типа сравниваем на кросс-валидации:
  - mean     - среднее по типу (базовый прогноз, его надо обогнать);
  - linear   - гребневая регрессия по признакам;
  - loglin   - то же по логарифму расхода (расход с длинным хвостом).
Кросс-валидация по группам: группа - файл трассы (обычно один seed генератора). Задачи одного seed
похожи друг на друга, и проверка на них же завысила бы точность. В модель идёт лучший вариант по
MAE, обученный на всех данных.

Только числа: ответы модели в трассах не используются.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "harness" / "cost_model.json"
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
MIN_ROWS = 8          # меньше задач типа - только среднее
FEATURE_COVERAGE = 0.8
SKIP_FEATURES = {"heur_cost"}  # эвристика решателя меняется вместе с кодом - не учимся на ней


def load(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("route") != "llm" or not isinstance(r.get("features"), dict):
                continue
            if r.get("err") in ("BudgetExhausted",) or r.get("cost") is None:
                continue  # задачу оборвали по общему бюджету - её расход не показателен
            if r["features"].get("generic"):
                continue  # универсальный путь оценивается эвристикой
            rows.append({"group": r.get("group") or str(p), "type": r["type"], "cost": float(r["cost"]),
                         "ok": bool(r.get("correct", r.get("ok"))), "f": r["features"]})
    return rows


def pick_features(rows):
    names = sorted({k for r in rows for k, v in r["f"].items() if isinstance(v, (int, float))} - SKIP_FEATURES)
    keep = []
    for n in names:
        vals = [r["f"].get(n) for r in rows]
        have = [v for v in vals if v is not None]
        if len(have) >= FEATURE_COVERAGE * len(rows) and len(set(have)) > 1:
            keep.append(n)
    return keep


def matrix(rows, names, fill):
    return np.array([[float(r["f"].get(n, fill[n])) for n in names] for r in rows], dtype=float)


def fit_ridge(X, y, lam):
    """Гребневая регрессия на стандартизованных признаках; возвращает (intercept, coef) в исходной шкале."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    coef = w / sd
    return float(ym - coef @ mu), coef


def predict(model, X):
    kind = model["kind"]
    if kind == "mean":
        return np.full(len(X), model["mean"])
    y = model["intercept"] + X @ model["coef"]
    if kind == "loglin":
        y = np.exp(np.minimum(y, 12.0))
    return np.maximum(y, 5.0)


def train(kind, X, y, lam):
    if kind == "mean":
        return {"kind": kind, "mean": float(y.mean())}
    target = np.log(np.maximum(y, 1.0)) if kind == "loglin" else y
    b, w = fit_ridge(X, target, lam)
    return {"kind": kind, "intercept": b, "coef": w}


def folds(rows):
    groups = sorted({r["group"] for r in rows})
    if len(groups) >= 3:
        return [[i for i, r in enumerate(rows) if r["group"] == g] for g in groups]
    # групп мало - обычные 5 частей (оценка будет оптимистичной, отчёт об этом пишет)
    idx = np.random.default_rng(0).permutation(len(rows))
    return [sorted(idx[k::5].tolist()) for k in range(5)]


def cv_mae(kind, rows, X, y, lam):
    errs = []
    for test in folds(rows):
        test_set = set(test)
        tr = [i for i in range(len(rows)) if i not in test_set]
        if not tr or not test:
            continue
        m = train(kind, X[tr], y[tr], lam)
        errs.extend(np.abs(predict(m, X[test]) - y[test]).tolist())
    return float(np.mean(errs)) if errs else math.inf


def fit_type(rows):
    y = np.array([r["cost"] for r in rows])
    names = pick_features(rows)
    fill = {n: float(np.mean([r["f"][n] for r in rows if r["f"].get(n) is not None])) for n in names}
    X = matrix(rows, names, fill) if names else np.zeros((len(rows), 0))
    results = [("mean", None, cv_mae("mean", rows, X, y, None))]
    if names and len(rows) >= MIN_ROWS:
        for kind in ("linear", "loglin"):
            for lam in LAMBDAS:
                results.append((kind, lam, cv_mae(kind, rows, X, y, lam)))
    kind, lam, mae = min(results, key=lambda r: r[2])
    m = train(kind, X, y, lam)
    resid = np.abs(predict(m, X) - y)
    n_ok = sum(r["ok"] for r in rows)
    out = {"kind": kind, "n": len(rows), "cv_mae": round(mae, 1),
           "cv_mae_mean": round(results[0][2], 1), "mean": round(float(y.mean()), 1),
           "p_success": round((n_ok + 1) / (len(rows) + 2), 3),
           "err_q90": round(float(np.quantile(resid, 0.9)), 1),
           "clip": [5.0, round(float(y.max()) * 3, 1)]}
    if kind != "mean":
        out.update({"intercept": m["intercept"], "coef": dict(zip(names, m["coef"].tolist(), strict=True)),
                    "fill": fill, "log": kind == "loglin", "lambda": lam})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows = load(a.traces)
    if not rows:
        print("нет LLM-задач с признаками в трассах", file=sys.stderr)
        return 1
    model = {"version": 2, "trained_on": len(rows), "types": {}}
    print(f"{'тип':14} {'n':>5} {'модель':>8} {'MAE':>7} {'MAE среднего':>13} {'p_success':>10}")
    for t in sorted({r["type"] for r in rows}):
        m = fit_type([r for r in rows if r["type"] == t])
        model["types"][t] = m
        print(f"{t:14} {m['n']:5d} {m['kind']:>8} {m['cv_mae']:7.1f} {m['cv_mae_mean']:13.1f} {m['p_success']:10.2f}")
    if len({r["group"] for r in rows}) < 3:
        print("внимание: меньше 3 файлов трасс - кросс-валидация не по группам, MAE оптимистичен")
    if not a.dry_run:
        Path(a.out).write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")
        print("записано:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
