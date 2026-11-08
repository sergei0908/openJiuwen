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
    "minimum": ("def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                ["def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                 "def minimum(xs):\n    best = xs[-1]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"],
                "self.assertEqual(m.minimum([3, 1, 2]), 1)"),
    "product": ("def product(xs):\n    r = 1\n    for x in xs:\n        r *= x\n    return r\n",
                ["def product(xs):\n    r = 0\n    for x in xs:\n        r *= x\n    return r\n",
                 "def product(xs):\n    r = 1\n    for x in xs:\n        r += x\n    return r\n"],
                "self.assertEqual(m.product([2, 3, 4]), 24)"),
    "factorial": ("def factorial(n):\n    r = 1\n    for i in range(1, n + 1):\n        r *= i\n    return r\n",
                  ["def factorial(n):\n    r = 1\n    for i in range(1, n):\n        r *= i\n    return r\n",
                   "def factorial(n):\n    r = 1\n    for i in range(n):\n        r *= i\n    return r\n",
                   "def factorial(n):\n    r = 0\n    for i in range(1, n + 1):\n        r += i\n    return r\n"],
                  "self.assertEqual(m.factorial(5), 120)"),
    "fibonacci": ("def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
                  ["def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return b\n",
                   "def fibonacci(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
                   "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a - b\n    return a\n"],
                  "self.assertEqual(m.fibonacci(7), 13)"),
    "is_prime": ("def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n",
                 ["def is_prime(n):\n    if n < 2:\n        return True\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n",
                  "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i != 0:\n            return False\n    return True\n",
                  "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(1, n):\n        if n % i == 0:\n            return False\n    return True\n"],
                 "self.assertTrue(m.is_prime(7))\n        self.assertFalse(m.is_prime(9))\n        self.assertFalse(m.is_prime(1))"),
    "last_n": ("def last_n(xs, n):\n    return xs[-n:]\n",
               ["def last_n(xs, n):\n    return xs[n:]\n", "def last_n(xs, n):\n    return xs[:-n]\n",
                "def last_n(xs, n):\n    return xs[:n]\n"],
               "self.assertEqual(m.last_n([5, 6, 7, 8, 9], 2), [8, 9])"),
    "unique": ("def unique(xs):\n    return sorted(set(xs))\n",
               ["def unique(xs):\n    return sorted(xs)\n", "def unique(xs):\n    return list(set(xs))[::-1]\n"],
               "self.assertEqual(m.unique([3, 1, 3, 2, 1]), [1, 2, 3])"),
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              ["def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n", "def clamp(x, lo, hi):\n    return max(lo, max(x, hi))\n"],
              "self.assertEqual(m.clamp(5, 0, 3), 3)\n        self.assertEqual(m.clamp(-1, 0, 3), 0)\n        self.assertEqual(m.clamp(2, 0, 3), 2)"),
    "capitalize_words": ("def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
                         ["def capitalize_words(s):\n    return ' '.join(w.upper() for w in s.split())\n",
                          "def capitalize_words(s):\n    return ' '.join(w.lower() for w in s.split())\n",
                          "def capitalize_words(s):\n    return ''.join(w.capitalize() for w in s.split())\n"],
                         "self.assertEqual(m.capitalize_words('hello big world'), 'Hello Big World')"),
    "sum_even": ("def sum_even(xs):\n    return sum(x for x in xs if x % 2 == 0)\n",
                 ["def sum_even(xs):\n    return sum(x for x in xs if x % 2 == 1)\n",
                  "def sum_even(xs):\n    return sum(x for x in xs if x % 2 != 0)\n",
                  "def sum_even(xs):\n    return sum(x for x in xs if x % 3 == 0)\n"],
                 "self.assertEqual(m.sum_even([1, 2, 3, 4, 9]), 6)"),
    "abs_diff": ("def abs_diff(a, b):\n    return abs(a - b)\n",
                 ["def abs_diff(a, b):\n    return a - b\n", "def abs_diff(a, b):\n    return abs(a + b)\n"],
                 "self.assertEqual(m.abs_diff(2, 5), 3)\n        self.assertEqual(m.abs_diff(5, 2), 3)"),
    "index_of": ("def index_of(xs, t):\n    for i, x in enumerate(xs):\n        if x == t:\n            return i\n    return -1\n",
                 ["def index_of(xs, t):\n    for i, x in enumerate(xs):\n        if x != t:\n            return i\n    return -1\n",
                  "def index_of(xs, t):\n    for i, x in enumerate(xs):\n        if x == t:\n            return i + 1\n    return -1\n",
                  "def index_of(xs, t):\n    for i, x in enumerate(xs):\n        if x == t:\n            return i\n    return 0\n"],
                 "self.assertEqual(m.index_of([4, 5, 6], 6), 2)\n        self.assertEqual(m.index_of([4, 5], 9), -1)"),
    "word_count": ("def word_count(s):\n    return len(s.split())\n",
                   ["def word_count(s):\n    return len(s)\n", "def word_count(s):\n    return len(s.split()) + 1\n"],
                   "self.assertEqual(m.word_count('a bb ccc'), 3)"),
    "is_sorted": ("def is_sorted(xs):\n    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))\n",
                  ["def is_sorted(xs):\n    return all(xs[i] >= xs[i + 1] for i in range(len(xs) - 1))\n",
                   "def is_sorted(xs):\n    return any(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))\n",
                   "def is_sorted(xs):\n    return all(xs[i] <= xs[i + 1] for i in range(len(xs)))\n"],
                  "self.assertTrue(m.is_sorted([1, 2, 2, 5]))\n        self.assertFalse(m.is_sorted([1, 5, 2]))"),
    "running_sum": ("def running_sum(xs):\n    out = []\n    s = 0\n    for x in xs:\n        s += x\n        out.append(s)\n    return out\n",
                    ["def running_sum(xs):\n    out = []\n    s = 0\n    for x in xs:\n        s += x\n        out.append(x)\n    return out\n",
                     "def running_sum(xs):\n    out = []\n    s = 0\n    for x in xs:\n        s -= x\n        out.append(s)\n    return out\n"],
                    "self.assertEqual(m.running_sum([1, 2, 3]), [1, 3, 6])"),
    "normalize": ("def normalize(s):\n    return s.strip().lower()\n",
                  ["def normalize(s):\n    return s.lower()\n", "def normalize(s):\n    return s.strip().upper()\n",
                   "def normalize(s):\n    return s.lstrip().lower()\n"],
                  "self.assertEqual(m.normalize('  HeLLo '), 'hello')"),
    "median": ("def median(xs):\n    ys = sorted(xs)\n    return ys[len(ys) // 2]\n",
               ["def median(xs):\n    ys = xs\n    return ys[len(ys) // 2]\n",
                "def median(xs):\n    ys = sorted(xs)\n    return ys[len(ys) // 2 + 1]\n",
                "def median(xs):\n    ys = sorted(xs)\n    return ys[0]\n"],
               "self.assertEqual(m.median([9, 1, 5]), 5)"),
    "to_celsius": ("def to_celsius(f):\n    return (f - 32) * 5 / 9\n",
                   ["def to_celsius(f):\n    return (f + 32) * 5 / 9\n", "def to_celsius(f):\n    return (f - 32) * 9 / 5\n",
                    "def to_celsius(f):\n    return f - 32 * 5 / 9\n"],
                   "self.assertEqual(m.to_celsius(212), 100)"),
    "power": ("def power(a, b):\n    return a ** b\n",
              ["def power(a, b):\n    return a * b\n", "def power(a, b):\n    return b ** a\n"],
              "self.assertEqual(m.power(2, 5), 32)"),
    "reverse_string": ("def reverse_string(s):\n    return s[::-1]\n",
                       ["def reverse_string(s):\n    return s[::1]\n", "def reverse_string(s):\n    return s\n"],
                       "self.assertEqual(m.reverse_string('abc'), 'cba')"),
    "count_vowels": ("def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
                     ["def count_vowels(s):\n    return sum(1 for c in s.lower() if c not in 'aeiou')\n",
                      "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n"],
                     "self.assertEqual(m.count_vowels('EduCation'), 5)"),
    "merge_dicts": ("def merge_dicts(a, b):\n    r = dict(a)\n    r.update(b)\n    return r\n",
                    ["def merge_dicts(a, b):\n    r = dict(b)\n    r.update(a)\n    return r\n",
                     "def merge_dicts(a, b):\n    r = dict(a)\n    r.update(a)\n    return r\n"],
                    "self.assertEqual(m.merge_dicts({'x': 1, 'y': 2}, {'y': 3}), {'x': 1, 'y': 3})"),
    "missing_return_total": ("def missing_return_total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
                             ["def missing_return_total(xs):\n    s = 0\n    for x in xs:\n        s += x\n"],
                             "self.assertEqual(m.missing_return_total([1, 2, 3]), 6)"),
    "not_empty": ("def not_empty(xs):\n    return len(xs) > 0\n",
                  ["def not_empty(xs):\n    return len(xs) == 0\n", "def not_empty(xs):\n    return len(xs) > 1\n"],
                  "self.assertTrue(m.not_empty([1]))\n        self.assertFalse(m.not_empty([]))"),
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
    if infeasible:  # противоречивые тесты: один вызов с двумя разными ответами, починить нельзя
        tests.append("    def test_contradiction(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n"
                     "        self.assertEqual(m.total([1, 2, 3]), 7)\n")
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
