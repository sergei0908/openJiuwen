"""Оценка расхода токенов до и во время выполнения.

До запуска: est = a_type + b_type * prompt_tokens (коэффициенты из cost_model.json,
обученного tools/fit_cost_model.py на собранных траекториях; иначе - эвристика решателя).
Во время запуска: онлайн-калибровка - множитель = (факт + k) / (прогноз + k) по уже
решённым задачам того же типа, чтобы следующие прогнозы подстраивались под реальную модель.
"""
import json
import threading
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("cost_model.json")


class Estimator:
    def __init__(self):
        self.coef = {}
        if MODEL_PATH.is_file():
            try:
                self.coef = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
            except ValueError:
                self.coef = {}
        self.obs = {}  # type -> [sum_pred, sum_fact, n]
        self.lock = threading.Lock()

    def predict(self, task_type, plan):
        base = plan.est_cost
        c = self.coef.get(task_type)
        if c and "prompt_tokens" in plan.notes:
            base = c["a"] + c["b"] * plan.notes["prompt_tokens"]
        p_ok = (self.coef.get(task_type) or {}).get("p_success", plan.p_success)
        with self.lock:
            sp, sf, n = self.obs.get(task_type, (0.0, 0.0, 0))
        k = 3 * base  # сглаживание: первые наблюдения двигают оценку мягко
        mult = (sf + k) / (sp + k) if n else 1.0
        return max(1.0, base * mult), p_ok

    def observe(self, task_type, predicted, actual):
        with self.lock:
            sp, sf, n = self.obs.get(task_type, (0.0, 0.0, 0))
            self.obs[task_type] = (sp + predicted, sf + actual, n + 1)
