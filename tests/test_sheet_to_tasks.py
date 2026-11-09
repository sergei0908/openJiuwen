"""Перевод таблицы задач Маши в папки задач."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from sheet_to_tasks import main


def test_sheet_is_converted(tmp_path):
    files = tmp_path / "files"
    files.mkdir()
    (files / "sales_17.csv").write_text("region,units\neast,5\nwest,7\n", encoding="utf-8")
    (files / "kb_09.csv").write_text("имя файла,текст\nperson_anna_lee.md,# Anna Lee\\n\\nEmployer: X Corp\n"
                                     "company_x_corp.md,# X Corp\\n\\nHeadquarters: Kazan\n", encoding="utf-8")
    sheet = tmp_path / "maria.csv"
    sheet.write_text(
        "id,type,reward,instruction,expected,файлы,что проверяю,набор\n"
        "maria_01,math,3,A shop has 450 apples. It sells 138. How many are left?,312,—,лишние числа,рабочий\n"
        "maria_09,search,2,Where is the main office of Anna Lee's employer?,Kazan,kb: kb_09,синоним,рабочий\n"
        "maria_17,data_analysis,3,How many units were sold in the east?,5,sales_17.csv -> sales.csv,слова,рабочий\n"
        "maria_20,coding,2,The tests fail. Repair mylib.py.,DONE,copy: coding_0003,формулировка,рабочий\n"
        "maria_21,math,1,It ships some boxes. How many are left?,IMPOSSIBLE,—,нет данных,отложенный\n",
        encoding="utf-8")
    out = tmp_path / "tasks"
    assert main(["--sheet", str(sheet), "--files", str(files), "--out", str(out), "--set", "рабочий"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["maria_01", "maria_09", "maria_17", "maria_20"]
    j = json.loads((out / "maria_01" / "task.json").read_text(encoding="utf-8"))
    assert j["check"] == "exact_int" and j["expected"] == "312" and "answer.txt" in j["instruction"]
    assert (out / "maria_09" / "workspace" / "kb" / "company_x_corp.md").read_text().startswith("# X Corp\n\n")
    assert (out / "maria_17" / "workspace" / "sales.csv").is_file()
    assert (out / "maria_20" / "workspace" / "mylib.py").is_file()
    j = json.loads((out / "maria_20" / "task.json").read_text(encoding="utf-8"))
    assert j["check"] == "unittest" and "the word DONE" in j["instruction"]

    held = tmp_path / "held"
    assert main(["--sheet", str(sheet), "--files", str(files), "--out", str(held), "--set", "отложенный"]) == 0
    j = json.loads((held / "maria_21" / "task.json").read_text(encoding="utf-8"))
    assert j["_infeasible"] is True and j["expected"] == "N/A"


def test_bad_rows_are_reported(tmp_path, capsys):
    sheet = tmp_path / "s.csv"
    sheet.write_text("id,type,reward,instruction,expected,files\nx1,poetry,1,Write a poem,ok,\n"
                     "x2,data_analysis,1,Sum?,3,missing.csv\n", encoding="utf-8")
    assert main(["--sheet", str(sheet), "--out", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert "poetry" in err and "missing.csv" in err
