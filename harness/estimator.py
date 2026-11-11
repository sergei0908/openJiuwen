"""Оценка расхода токенов до и во время выполнения.

До запуска: модель из harness/cost_model.json, обученная tools/fit_cost_model.py на наших трассах
(организаторы: «агент не успеет набрать статистику, надо нагенерить трасс самим и на них учиться»).
Для каждого типа задачи - линейная модель по признакам из harness/features.py (возможно, по
логарифму расхода). Если модели нет или для задачи нет признаков - эвристика решателя plan.est_cost.

Во время запуска: мягкая онлайн-калибровка - множитель (факт + k) / (прогноз + k) по уже решённым
задачам того же типа. Статистики одного прогона мало, поэтому k велико и поправка небольшая.
"""
import json
import math
import threading
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("cost_model.json")
CALIB_K = 8  # во сколько «прогнозов» весит априорная оценка при онлайн-калибровке
MIN_COST = 5.0


def load_model(path=MODEL_PATH):
    """Модель по типам. Поддерживает старый формат {type: {a, b, p_success}}."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    types = raw.get("types", raw) if isinstance(raw, dict) else {}
    return {t: m for t, m in types.items() if isinstance(m, dict)}


def linear_predict(m, feats):
    """Прогноз одной модели типа; None, если модель к задаче неприменима."""
    coef = m.get("coef")
    if coef is None:
        if "a" in m and "prompt_tokens" in feats:  # старый формат
            return m["a"] + m.get("b", 0) * feats["prompt_tokens"]
        return m.get("mean")
    fill = m.get("fill", {})
    y = m.get("intercept", 0.0)
    for name, w in coef.items():
        v = feats.get(name, fill.get(name))
        if v is None:
            return m.get("mean")
        y += w * float(v)
    if m.get("log"):
        y = math.exp(min(y, 12.0))
    y *= m.get("scale", 1.0)  # поправка на сдвиг цен после смены промптов (медиана факт/прогноз)
    lo, hi = m.get("clip", (MIN_COST, None))
    y = max(lo or MIN_COST, y)
    return min(hi, y) if hi else y


class Estimator:
    def __init__(self, path=MODEL_PATH):
        self.coef = load_model(path)
        self.obs = {}  # type -> [sum_pred, sum_fact, n]
        self.lock = threading.Lock()

    def base(self, task_type, plan):
        m = self.coef.get(task_type)
        feats = plan.notes.get("features") or {}
        if m and not plan.notes.get("generic"):
            y = linear_predict(m, feats)
            if y is not None:
                return float(y)
        return float(plan.est_cost)

    def predict(self, task_type, plan):
        base = self.base(task_type, plan)
        m = self.coef.get(task_type) or {}
        p_ok = m.get("p_success", plan.p_success) if not plan.notes.get("generic") else plan.p_success
        with self.lock:
            sp, sf, n = self.obs.get(task_type, (0.0, 0.0, 0))
        k = CALIB_K * base
        mult = (sf + k) / (sp + k) if n else 1.0
        return max(1.0, base * mult), p_ok

    def observe(self, task_type, predicted, actual):
        with self.lock:
            sp, sf, n = self.obs.get(task_type, (0.0, 0.0, 0))
            self.obs[task_type] = (sp + predicted, sf + actual, n + 1)
