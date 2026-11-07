"""Точка входа харнесса команды Burn Rate.

    python run_agent.py --tasks-dir /data/tasks [--out /path/solutions.json]

Переменные окружения: LLM_BASE_URL, LLM_API_KEY (прокси организаторов), TASKS, OUT,
TOKEN_BUDGET (если бюджет известен), FAST_PATH=0/1 и др. (см. harness/config.py).
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.dont_write_bytecode = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    ap.add_argument("--out", default=os.environ.get("OUT"))
    a = ap.parse_args()
    tasks_dir = Path(a.tasks_dir).resolve()

    # Корень образа read-only: openJiuwen пишет logs/ в текущий каталог - уходим в /tmp.
    work = Path(tempfile.gettempdir()) / "burnrate"
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    os.environ.setdefault("TMPDIR", str(work))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    from harness.runner import Runner
    Runner(tasks_dir, a.out).run()


if __name__ == "__main__":
    main()
