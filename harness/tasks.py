"""Загрузка задач и запись ответов."""
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
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


def can_write(directory: Path) -> bool:
    """Контейнер работает без capabilities: root не пишет в чужой каталог без прав на запись."""
    probe = directory / ".write_probe"
    try:
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def write_file(path: Path, text: str):
    """Пишет файл. Если сам файл закрыт от записи, а каталог нет, заменяет файл через переименование."""
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as first:
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            raise first from None


_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4}
# Только просьба об ответе («Round to 2 decimal places», «answer with one decimal»), а не описание данных
# («The price column has 2 decimal places»): иначе целый ответ 12 превратился бы в 12.00.
DECIMALS_RE = re.compile(r"\b(\d|one|two|three|four)\s+decimals?\b", re.IGNORECASE)
REQUEST_RE = re.compile(r"\b(?:round|rounded|rounding|answer|give|report|express|show|write|precision|accurate)\b",
                        re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def match_decimals(instruction, answer):
    """«Round to 2 decimal places» и ответ-число с лишними знаками: округляем (27.505 -> 27.51).

    Нули в конце не дописываем и убираем: 25.4 остаётся 25.4, а не 25.40. Так выглядит round(x, 2) в
    Python - скорее всего, так посчитан и эталон. Если знаков не больше, чем просили, в условии разное
    число знаков или ответ не одно число, ответ остаётся как есть (кроме нулей в конце)."""
    asks = [x for x in re.split(r"(?<=[.?!])\s+", instruction or "") if REQUEST_RE.search(x)]
    counts = {int(n) if n.isdigit() else _WORD_NUM[n.lower()] for x in asks for n in DECIMALS_RE.findall(x)}
    if len(counts) != 1 or not NUMBER_RE.fullmatch(answer):
        return answer
    places = counts.pop()
    if len(answer.partition(".")[2]) <= places:
        # знаков не больше, чем просили: только убираем нули в конце (модель пишет «25.40»)
        if "." in answer:
            answer = answer.rstrip("0").rstrip(".")
        return "0" if answer in ("-0", "") else answer
    try:
        out = str(Decimal(answer).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return answer
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return "0" if out in ("-0", "") else out


class AnswerWriter:
    """Пишет answer.txt рядом с task.json сразу после решения + общий файл решений (если задан).

    Для невыполнимой задачи файл не создаётся (требование организаторов). Ошибка записи не должна
    пройти незаметно: иначе в логе будут баллы, а на деле ни одного ответа. Если каталог задачи закрыт,
    ответ дублируется в OUT и в лог.
    """

    def __init__(self, out_path=None):
        self.out_path = Path(out_path) if out_path else None
        self.solutions = {}

    def write(self, task: Task, answer: str) -> bool:
        """True, если ответ сохранён хотя бы в одном месте."""
        answer = match_decimals(task.instruction, str(answer).strip())
        with _lock:
            in_task_dir = False
            try:
                write_file(task.dir / "answer.txt", answer + "\n")
                in_task_dir = True
            except OSError as e:
                print(f"ERROR cannot write answer for {task.task_id}: {e}", file=sys.stderr, flush=True)
            self.solutions[task.task_id] = answer
            in_out = self._write_out(task, answer, in_task_dir)
            if not in_task_dir:  # последняя страховка: ответ останется хотя бы в логе запуска
                print(f"ANSWER {task.task_id} {answer}", flush=True)
            return in_task_dir or in_out

    def _write_out(self, task, answer, in_task_dir):
        if not self.out_path:
            return False
        try:
            p = self.out_path
            as_dir = p.is_dir() or str(p).endswith("/")
            if as_dir:
                p = p / "solutions.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(self.solutions, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, p)
            if as_dir and not in_task_dir:
                d = self.out_path / task.dir.name
                d.mkdir(parents=True, exist_ok=True)
                (d / "answer.txt").write_text(answer + "\n", encoding="utf-8")
            return True
        except OSError as e:
            print(f"ERROR cannot write {self.out_path}: {e}", file=sys.stderr, flush=True)
            return False
