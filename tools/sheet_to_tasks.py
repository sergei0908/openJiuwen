"""Переводит таблицу задач Маши (Google-таблица, скачанная как CSV) в папки задач нашего формата.

    python tools/sheet_to_tasks.py --sheet maria.csv --files ~/Downloads/maria_files \
        --out ~/burn-rate-data/maria --set рабочий

Столбцы таблицы (заголовки можно по-русски или по-английски):
  id, type, reward, instruction, expected, файлы (files), что проверяю (note), набор (set)
expected: ответ; для coding - DONE; для нерешаемых - IMPOSSIBLE.
Столбец «файлы», через запятую:
  sales_17.csv                 - CSV из папки --files, кладётся под тем же именем;
  sales_17.csv -> sales.csv    - то же, но под другим именем (как в тексте задачи);
  kb: kb_09                    - база знаний: лист kb_09, скачанный как kb_09.csv со столбцами
                                 «имя файла», «текст»; файлы ложатся в workspace/kb/;
  copy: coding_0003            - взять код и тесты из examples/tasks/coding_0003/workspace
                                 (или из папки --files/coding_0003, если такая есть).
Хвост «Work only inside the current directory… answer.txt» добавляется, если его нет в тексте.
"""
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAIL = (" Work only inside the current directory. When you are done, write ONLY {what} to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
TYPES = {"math", "search", "data_analysis", "coding"}
ALIASES = {"файлы": "files", "что проверяю": "note", "набор": "set", "тип": "type", "награда": "reward",
           "задача": "instruction", "ответ": "expected"}
SETS = {"рабочий": "work", "work": "work", "отложенный": "holdout", "holdout": "holdout"}


def norm_header(h):
    h = (h or "").strip().lower()
    return ALIASES.get(h, h)


def read_sheet(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    head = [norm_header(h) for h in rows[0]]
    return [dict(zip(head, (c.strip() for c in r), strict=False)) for r in rows[1:] if any(c.strip() for c in r)]


def place_files(spec, files_dir: Path, ws: Path):
    for item in [x.strip() for x in (spec or "").split(",") if x.strip() and x.strip() not in ("—", "-")]:
        if item.lower().startswith("kb:"):
            sheet = files_dir / (item.split(":", 1)[1].strip() + ".csv")
            kb = ws / "kb"
            kb.mkdir(parents=True, exist_ok=True)
            for r in read_sheet(sheet):
                vals = list(r.values())
                name, text = (vals + ["", ""])[:2]
                if name:
                    (kb / Path(name).name).write_text(text.replace("\\n", "\n") + "\n", encoding="utf-8")
        elif item.lower().startswith("copy:"):
            name = item.split(":", 1)[1].strip()
            src = files_dir / name
            if not src.is_dir():
                src = ROOT / "examples" / "tasks" / name / "workspace"
            if not src.is_dir():
                raise FileNotFoundError(f"нет папки для copy: {name}")
            shutil.copytree(src, ws, dirs_exist_ok=True)
        else:
            src_name, _, dst_name = (s.strip() for s in item.partition("->"))
            src = files_dir / src_name
            if not src.is_file():
                raise FileNotFoundError(f"нет файла {src}")
            ws.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, ws / (dst_name or src_name))


def build_task(row, files_dir: Path, out: Path):
    tid, ttype = row.get("id", ""), row.get("type", "").strip().lower()
    if not re.fullmatch(r"[\w.-]+", tid):
        raise ValueError(f"плохой id: {tid!r}")
    if ttype not in TYPES:
        raise ValueError(f"{tid}: тип {ttype!r}, нужен один из {sorted(TYPES)}")
    exp = row.get("expected", "").strip()
    infeasible = exp.upper() == "IMPOSSIBLE"
    instr = row.get("instruction", "").strip()
    if "answer.txt" not in instr:
        instr += TAIL.format(what="the word DONE" if ttype == "coding" else "the final answer")
    if ttype == "coding":
        check, expected = "unittest", "tests_pass"
    elif infeasible:
        check, expected = "exact_str", "N/A"
    else:
        check = "exact_int" if re.fullmatch(r"-?\d+", exp) else "exact_str"
        expected = exp
    d = out / tid
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    j = {"task_id": tid, "task_type": ttype, "instruction": instr,
         "reward": float(row.get("reward") or 1), "expected": expected, "check": check}
    if infeasible:
        j["_infeasible"] = True
    if row.get("note"):
        j["_note"] = row["note"]
    place_files(row.get("files"), files_dir, d / "workspace")
    (d / "task.json").write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--files", default=".", help="папка с CSV и листами kb_*.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="all", help="рабочий / отложенный / all")
    a = ap.parse_args(argv)
    want = SETS.get(a.set.lower(), "all")
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    made, errors = 0, []
    for row in read_sheet(a.sheet):
        if want != "all" and SETS.get(row.get("set", "").lower(), "work") != want:
            continue
        try:
            build_task(row, Path(a.files).expanduser(), out)
            made += 1
        except (ValueError, OSError) as e:
            errors.append(str(e))
    for e in errors:
        print("ОШИБКА:", e, file=sys.stderr)
    print(f"создано задач: {made} в {out}" + (f", с ошибками: {len(errors)}" if errors else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
