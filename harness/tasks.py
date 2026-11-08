"""Загрузка задач и запись ответов."""
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

_lock = threading.Lock()


@dataclass
class Task:
    task_id: str
    dir: Path
    task_type: str
    instruction: str
    reward: float
    raw: dict = field(default_factory=dict)

    @property
    def workspace(self) -> Path:
        ws = self.dir / "workspace"
        return ws if ws.is_dir() else self.dir


def load_tasks(tasks_dir: Path):
    out = []
    for d in sorted(tasks_dir.iterdir()):
        f = d / "task.json"
        if not d.is_dir() or not f.is_file():
            continue
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001, S112 - битый task.json пропускаем
            continue
        try:
            reward = float(j.get("reward", 1) or 0)
        except (TypeError, ValueError):
            reward = 1.0
        out.append(Task(task_id=str(j.get("task_id", d.name)), dir=d,
                        task_type=str(j.get("task_type", "")).lower(),
                        instruction=str(j.get("instruction", "")), reward=reward, raw=j))
    return out


def find_budget(tasks_dir: Path):
    """Ищет бюджет токенов: env -> файлы в каталоге задач. None, если неизвестен."""
    for k in ("BUDGET", "TOKEN_BUDGET", "LLM_BUDGET", "OUTPUT_TOKEN_BUDGET"):  # организаторы: BUDGET
        v = os.environ.get(k)
        if v:
            try:
                return float(v)
            except ValueError:
                pass
    for name in ("budget.json", "budget.txt", "budget", "config.json", "tasks.json", "meta.json"):
        p = tasks_dir / name
        if p.is_file():
            try:
                txt = p.read_text(encoding="utf-8").strip()
                try:
                    j = json.loads(txt)
                except ValueError:
                    return float(txt)
                if isinstance(j, (int, float)):
                    return float(j)
                if isinstance(j, dict):
                    for k in ("budget", "token_budget", "output_token_budget", "tokens"):
                        if k in j:
                            return float(j[k])
            except Exception:  # noqa: BLE001, S110 - нечитаемый файл бюджета пропускаем
                pass
    # Если BUDGET не передан, работаем с бюджетом организаторов: 100k выходных токенов.
    # DEFAULT_BUDGET=0 - считать бюджет неизвестным.
    v = float(os.environ.get("DEFAULT_BUDGET", "100000"))
    return v if v > 0 else None


def can_write(task: Task) -> bool:
    """Контейнер работает без capabilities: root не пишет в чужой каталог без прав на запись."""
    probe = task.dir / ".write_probe"
    try:
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


class AnswerWriter:
    """Пишет answer.txt сразу после решения + общий файл решений (если задан)."""

    def __init__(self, out_path=None):
        self.out_path = Path(out_path) if out_path else None
        self.solutions = {}

    def write(self, task: Task, answer: str):
        answer = str(answer).strip()
        with _lock:
            targets = {task.dir / "answer.txt", task.workspace / "answer.txt"}
            for p in targets:
                try:
                    p.write_text(answer + "\n", encoding="utf-8")
                except OSError as e:
                    print(f"ERROR cannot write {p}: {e}", file=sys.stderr, flush=True)
            self.solutions[task.task_id] = answer
            if self.out_path:
                try:
                    p = self.out_path
                    if p.is_dir() or str(p).endswith("/"):
                        p = p / "solutions.json"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix(p.suffix + ".tmp")
                    tmp.write_text(json.dumps(self.solutions, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
                    os.replace(tmp, p)
                except OSError:
                    pass
