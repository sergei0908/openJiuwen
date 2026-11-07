"""Настройки харнесса. Всё переопределяется переменными окружения."""
import os


def _env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


# Модели: по умолчанию дешёвая без рассуждений; reasoning - только для эскалации.
MODEL_FAST = _env("MODEL_FAST", "qwen3.8-27b-noreason")
MODEL_STRONG = _env("MODEL_STRONG", "qwen3.8-27b")
MODEL_ALT = _env("MODEL_ALT", "qwen3.6-35b-a3b-noreason")

# Решатели без LLM (разбор шаблонных задач, автоматический ремонт кода мутациями).
FAST_PATH = _env("FAST_PATH", "1") == "1"
# Разрешать эскалацию на reasoning-модель (дорого: рассуждения считаются как выходные токены).
ALLOW_REASONING = _env("ALLOW_REASONING", "1") == "1"

PARALLEL = int(_env("PARALLEL", "5"))          # регламент: до 5 сессий одновременно
TIME_LIMIT_S = float(_env("TIME_LIMIT_S", "3300"))  # 1 час минус запас
MAX_OUT_FAST = int(_env("MAX_OUT_FAST", "1200"))
MAX_OUT_STRONG = int(_env("MAX_OUT_STRONG", "3000"))
LLM_TIMEOUT_S = float(_env("LLM_TIMEOUT_S", "120"))

# Бюджет в «выходных токенах»: cost = out + in/10. Если не передан - считаем неизвестным.
INPUT_WEIGHT = 0.1
RESERVE_FRAC = float(_env("RESERVE_FRAC", "0.03"))

TRAJ_PATH = _env("TRAJ_PATH", "/tmp/trajectories.jsonl")
