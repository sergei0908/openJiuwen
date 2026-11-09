"""Варианты оформления задач: те же задачи, что в examples/tasks, но в другом формате.

    python tools/make_variants.py --out /tmp/variants

Проверяет устойчивость харнесса к тому, как жюри может оформить задачи иначе, чем в примерах:
  coding  - тесты в стиле pytest (функции с assert), тесты в подпапке tests/, модуль не mylib.py;
  data    - CSV с разделителем «;», файл в подпапке data/, вопрос со средним и округлением;
  search  - база знаний не в kb/, а в docs/; одна большая база в одном файле.
Правильный ответ каждой задачи сохраняется (expected), score.py проверяет как обычно.
"""
import argparse
import csv
import io
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "tasks"


def load(tid):
    return json.loads((EX / tid / "task.json").read_text(encoding="utf-8"))


def clone(tid, new_id, out):
    d = out / new_id
    shutil.copytree(EX / tid, d)
    j = load(tid)
    j["task_id"] = new_id
    return d, j


def save(d, j):
    (d / "task.json").write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")


def to_pytest(src):
    """unittest.TestCase -> функции pytest с assert."""
    out = ["import mylib as m", ""]
    for name, body in re.findall(r"def (test_\w+)\(self\):\n((?:        .*\n)+)", src):
        out.append(f"def {name}():")
        for line in body.splitlines():
            line = line.strip()
            mm = re.fullmatch(r"self\.assertEqual\((.+), (.+)\)", line)
            out.append(f"    assert {mm.group(1)} == {mm.group(2)}" if mm else f"    {line}")
        out.append("")
    return "\n".join(out)


def coding(out):
    # 1. pytest-стиль
    d, j = clone("coding_0003", "coding_v_pytest", out)
    ws = d / "workspace"
    (ws / "test_mylib.py").write_text(to_pytest((ws / "test_mylib.py").read_text()), encoding="utf-8")
    j["instruction"] = j["instruction"].replace("`python3 test_mylib.py`", "`python3 -m pytest -q`")
    save(d, j)
    # 2. тесты в подпапке tests/
    d, j = clone("coding_0003", "coding_v_subdir", out)
    ws = d / "workspace"
    (ws / "tests").mkdir()
    shutil.move(ws / "test_mylib.py", ws / "tests" / "test_mylib.py")
    (ws / "tests" / "__init__.py").write_text("", encoding="utf-8")
    save(d, j)
    # 3. модуль с другим именем
    d, j = clone("coding_0007", "coding_v_rename", out)
    ws = d / "workspace"
    shutil.move(ws / "mylib.py", ws / "textutils.py")
    t = ws / "test_mylib.py"
    t.write_text(t.read_text().replace("import mylib as m", "import textutils as m"), encoding="utf-8")
    shutil.move(t, ws / "test_textutils.py")
    j["instruction"] = j["instruction"].replace("mylib.py", "textutils.py").replace("test_mylib", "test_textutils")
    save(d, j)


def data(out):
    # 1. разделитель ;
    d, j = clone("data_analysis_0002", "data_v_semicolon", out)
    p = d / "workspace" / "sales.csv"
    rows = list(csv.reader(io.StringIO(p.read_text())))
    p.write_text("\n".join(";".join(r) for r in rows) + "\n", encoding="utf-8")
    j["instruction"] = j["instruction"].replace("has columns order_id,region,product,month,units,price",
                                                "uses ';' as the separator and has columns "
                                                "order_id;region;product;month;units;price")
    save(d, j)
    # 2. файл в подпапке data/
    d, j = clone("data_analysis_0002", "data_v_subdir", out)
    ws = d / "workspace"
    (ws / "data").mkdir()
    shutil.move(ws / "sales.csv", ws / "data" / "sales.csv")
    j["instruction"] = j["instruction"].replace("The file sales.csv", "The file data/sales.csv")
    save(d, j)
    # 3. среднее с округлением до 2 знаков
    d, j = clone("data_analysis_0002", "data_v_average", out)
    rows = list(csv.DictReader(io.StringIO((d / "workspace" / "sales.csv").read_text())))
    vals = [int(r["units"]) for r in rows if r["region"] == "north"]
    j["instruction"] = re.sub(r"How many units .*? inclusive\?",
                              "What is the average number of units per order in the 'north' region? "
                              "Round to 2 decimal places.", j["instruction"])
    j["expected"], j["check"] = f"{sum(vals) / len(vals):.2f}", "exact_str"
    save(d, j)


def search(out):
    # 1. база в docs/
    d, j = clone("search_0001", "search_v_docs", out)
    ws = d / "workspace"
    shutil.move(ws / "kb", ws / "docs")
    j["instruction"] = j["instruction"].replace("kb/", "docs/")
    save(d, j)
    # 2. вся база в одном файле
    d, j = clone("search_0005", "search_v_onefile", out)
    ws = d / "workspace"
    parts = [p.read_text(encoding="utf-8").strip() for p in sorted((ws / "kb").rglob("*.md"))]
    shutil.rmtree(ws / "kb")
    (ws / "knowledge_base.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    j["instruction"] = re.sub(r"The directory kb/ contains a small knowledge base of markdown files\.",
                              "The file knowledge_base.md contains a small knowledge base.", j["instruction"])
    save(d, j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    coding(out)
    data(out)
    search(out)
    print(f"variants: {len(list(out.iterdir()))} -> {out}")


if __name__ == "__main__":
    main()
