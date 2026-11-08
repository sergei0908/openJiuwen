"""Генератор тренировочных задач в формате организаторов (+ невыполнимые, + новые формулировки).

    python3 tools/gen_tasks.py --out /tmp/train --n 100 --seed 1 [--infeasible 0.2] [--novel 0.15]

novel - доля задач с формулировками, которых нет в примерах (проверка LLM-пути и устойчивости).
"""
import argparse
import json
import random
import shutil
from pathlib import Path

FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira",
         "Maya", "Nikolai", "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya"]
LAST = ["Smith", "Novak", "Sato", "Muller", "Garcia", "Haddad", "Ivanova", "Kowalski", "Petrov", "Wang"]
COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio",
             "Fjord Data", "Granite AI", "Iris Optics", "Juniper Foods", "Helix Motors"]
CITIES = ["Osaka", "Leeds", "Lyon", "Kazan", "Porto", "Austin", "Gdansk", "Busan", "Tartu", "Cork"]
PRODUCTS = ["widget", "gadget", "gizmo", "sprocket"]
REGIONS = ["north", "south", "east", "west"]
TAIL = (" Work only inside the current directory. When you are done, write ONLY {what} to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


def save(root, tid, ttype, instr, reward, expected, check, files=None, infeasible=False):
    d = root / tid
    d.mkdir(parents=True)
    what = "the word DONE" if ttype == "coding" else "the final answer"
    j = {"task_id": tid, "task_type": ttype, "instruction": instr + TAIL.format(what=what),
         "reward": reward, "expected": expected, "check": check}
    if infeasible:
        j["_infeasible"] = True  # только для локальной оценки; в настоящих задачах этого поля нет
    (d / "task.json").write_text(json.dumps(j, indent=2))
    if files:
        ws = d / "workspace"
        ws.mkdir()
        for name, content in files.items():
            p = ws / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)


def gen_math(r, novel):
    s = r.randint(500, 2000)
    sents = [f"A warehouse starts with {s} boxes."]
    for _ in range(r.randint(2, 5)):
        k = r.choice(["ship", "recv", "split", "double"] + (["sell", "triple"] if novel else []))
        if k == "ship":
            a, b = r.randint(5, 60), r.randint(1, 5)
            s -= a * b
            sents.append(f"It ships {a} boxes per day for {b} days.")
        elif k == "recv":
            a, b = r.randint(1, 4), r.randint(20, 120)
            s += a * b
            sents.append(f"It receives {a} deliveries of {b} boxes each.")
        elif k == "split":
            a = r.randint(2, 4)
            s //= a
            sents.append(f"Then the stock is split equally into {a} warehouses and it keeps one share (rounded down).")
        elif k == "double":
            s *= 2
            sents.append("Then the stock is doubled by merging with a sister warehouse.")
        elif k == "sell":
            a = r.randint(10, 90)
            s -= a
            sents.append(f"Then a customer buys {a} boxes from it.")
        elif k == "triple":
            s *= 3
            sents.append("Then its stock grows threefold after an acquisition.")
    instr = " ".join(sents) + " How many boxes does it have at the end? Compute it with the bash tool (python3 -c)."
    return instr, str(s), "exact_int", None, False


def gen_kb(r, n_people=26, n_comp=10):
    comps = r.sample(COMPANIES, n_comp)
    comp = {c: {"Headquarters": r.choice(CITIES), "Founded": str(r.randint(1950, 2015))} for c in comps}
    names = set()
    while len(names) < n_people:
        names.add(f"{r.choice(FIRST)} {r.choice(LAST)}")
    names = sorted(names)
    ppl = {n: {"Born": str(r.randint(1950, 2000)), "Employer": r.choice(comps),
               "Mentor": f"{r.choice(FIRST)} {r.choice(LAST)}"} for n in names}
    for n in names:  # часть наставников - люди из базы
        if r.random() < 0.6:
            ppl[n]["Mentor"] = r.choice([m for m in names if m != n])
    files = {}
    for c, f in comp.items():
        files[f"kb/company_{c.lower().replace(' ', '_')}.md"] = f"# {c}\n\nHeadquarters: {f['Headquarters']}\nFounded: {f['Founded']}\n"
    for n, f in ppl.items():
        files[f"kb/person_{n.lower().replace(' ', '_')}.md"] = f"# {n}\n\nBorn: {f['Born']}\nEmployer: {f['Employer']}\nMentor: {f['Mentor']}\n"
    return ppl, comp, files


def gen_search(r, novel, infeasible):
    ppl, comp, files = gen_kb(r)
    names = sorted(ppl)
    p = r.choice(names)
    if infeasible:
        while True:
            ghost = f"{r.choice(FIRST)} {r.choice(LAST)}"
            if ghost not in ppl:
                break
        q = r.choice([f"In which year was {ghost} born?",
                      f"In which city is the headquarters of {ghost}'s employer?"])
        exp = "N/A"
    else:
        kinds = ["born", "hq"] + (["founded", "mentor_born", "mentor"] if novel else [])
        k = r.choice(kinds)
        if k == "born":
            q, exp = f"In which year was {p} born?", ppl[p]["Born"]
        elif k == "hq":
            q, exp = f"In which city is the headquarters of {p}'s employer?", comp[ppl[p]["Employer"]]["Headquarters"]
        elif k == "founded":
            q, exp = f"When was the company that employs {p} founded?", comp[ppl[p]["Employer"]]["Founded"]
        elif k == "mentor":
            q, exp = f"Who is the mentor of {p}?", ppl[p]["Mentor"]
        else:
            m = [x for x in names if ppl[x]["Mentor"] in ppl]
            p = r.choice(m)
            q, exp = f"What is the birth year of the person who mentors {p}?", ppl[ppl[p]["Mentor"]]["Born"]
    instr = ("The directory kb/ contains a small knowledge base of markdown files. " + q +
             " Use the file tools or bash (ls, grep, cat) to find it.")
    return instr, exp, "exact_str", files, infeasible


def gen_data(r, novel, infeasible):
    n = r.choice([20, 50, 100, 300])
    rows = []
    for i in range(1, n + 1):
        rows.append((i, r.choice(REGIONS), r.choice(PRODUCTS), r.randint(1, 12), r.randint(1, 40), r.randint(10, 99)))
    csv = "order_id,region,product,month,units,price\n" + "\n".join(",".join(map(str, x)) for x in rows) + "\n"
    a = r.randint(1, 11)
    b = r.randint(a, min(12, a + 2))
    prod = r.choice(PRODUCTS)
    bad_q = None
    if infeasible:  # данных для точного ответа нет: товар, регион или месяцы не существуют
        kind = r.choice(["product", "region", "months", "reversed"])
        if kind == "product":
            prod = r.choice(["doohickey", "thingamajig", "whatsit"])
        elif kind == "region":
            bad_q = f"How many units of product '{prod}' in region 'atlantis' were sold in months {a} through {b} inclusive?"
        elif kind == "months":
            bad_q = f"How many units of product '{prod}' were sold in months {r.randint(13, 20)} through {r.randint(21, 30)} inclusive?"
        else:
            bad_q = f"How many units of product '{prod}' were sold in months {b + 1} through {a} inclusive?"
    if bad_q:
        q, exp = bad_q, "N/A"
    elif novel and not infeasible:
        reg = r.choice(REGIONS)
        q = f"What is the total revenue (units times price) of product '{prod}' in region '{reg}' for months {a} to {b}?"
        exp = sum(x[4] * x[5] for x in rows if x[2] == prod and x[1] == reg and a <= x[3] <= b)
    else:
        q = f"How many units of product '{prod}' were sold in months {a} through {b} inclusive?"
        exp = sum(x[4] for x in rows if x[2] == prod and a <= x[3] <= b)
    instr = (f"The file sales.csv has columns order_id,region,product,month,units,price ({n} rows). "
             + q + " Use bash with python3 to compute it.")
    return instr, str(exp), "exact_int", {"sales.csv": csv}, infeasible


FUNCS = {
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                      ["def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
                       "def reverse_words(s):\n    return ' '.join(s.split())\n"],
                      "self.assertEqual(m.reverse_words('c a b'), 'b a c')"),
    "total": ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
              ["def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
               "def total(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s\n"],
              "self.assertEqual(m.total([1, 2, 3]), 6)"),
    "maximum": ("def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                ["def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                 "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return x\n"],
                "self.assertEqual(m.maximum([3, 9, 2]), 9)"),
    "average": ("def average(xs):\n    return sum(xs) / len(xs)\n",
                ["def average(xs):\n    return sum(xs) / (len(xs) + 1)\n",
                 "def average(xs):\n    return sum(xs) // len(xs) + 1\n"],
                "self.assertEqual(m.average([2, 4, 6]), 4)"),
    "first_n": ("def first_n(xs, n):\n    return xs[:n]\n",
                ["def first_n(xs, n):\n    return xs[:n - 1]\n", "def first_n(xs, n):\n    return xs[n:]\n"],
                "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])"),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
                   ["def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n"],
                   "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)"),
    "is_palindrome": ("def is_palindrome(s):\n    t = s.lower()\n    return t == t[::-1]\n",
                      ["def is_palindrome(s):\n    t = s.upper()\n    return t != t[::-1]\n"],
                      "self.assertTrue(m.is_palindrome('Abba'))\n        self.assertFalse(m.is_palindrome('abc'))"),
}


def gen_coding(r, novel, infeasible):
    names = r.sample(sorted(FUNCS), r.randint(4, 6))
    n_bugs = r.randint(1, 3)
    buggy = set(r.sample(names, n_bugs))
    src, tests = [], []
    for nm in names:
        good, bads, t = FUNCS[nm]
        src.append(r.choice(bads) if nm in buggy else good)
        tests.append(f"    def test_{nm}(self):\n        {t}\n")
    if infeasible:  # противоречивые тесты: починить нельзя
        tests.append("    def test_contradiction(self):\n        self.assertEqual(m.total([1, 2, 3]), 7)\n")
        if "total" not in names:
            src.append(FUNCS["total"][0])
    test = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n" + "\n".join(tests) + \
           "\n\nif __name__ == '__main__':\n    unittest.main()\n"
    instr = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in "
             "mylib.py (do not edit the tests) until all tests pass.")
    return instr, "tests_pass", "unittest", {"mylib.py": "\n\n".join(src), "test_mylib.py": test}, infeasible


GEN = {"math": gen_math, "search": gen_search, "data_analysis": gen_data, "coding": gen_coding}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infeasible", type=float, default=0.2)
    ap.add_argument("--novel", type=float, default=0.0)
    a = ap.parse_args()
    r = random.Random(a.seed)
    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for i in range(a.n):
        t = r.choice(sorted(GEN))
        inf = r.random() < a.infeasible and t != "math"
        novel = r.random() < a.novel
        if t == "math":
            instr, exp, chk, files, inf = gen_math(r, novel)
        else:
            instr, exp, chk, files, inf = GEN[t](r, novel, inf)
        save(root, f"{t}_{i:04d}", t, instr, r.randint(1, 5), exp, chk, files, inf)
    print(f"generated {a.n} tasks in {root}")


if __name__ == "__main__":
    main()
