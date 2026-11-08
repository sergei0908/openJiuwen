"""Вопросы к базе знаний kb/*.md.

Бесплатно: парсим kb в словарь сущностей и разрешаем цепочки вида
"X's employer's headquarters". Если имени нет в базе - задача невыполнима, токены не тратим.
LLM-путь: в промпт идут только файлы упомянутых сущностей и их связей (2 шага по графу).
"""
import re
from pathlib import Path

from .base import (
    ANSWER_RULES,
    Plan,
    approx_tokens,
    clean_answer,
    extract_question,
    is_unsolvable,
)

REL = {"employer": "employer", "company": "employer", "mentor": "mentor",
       "boss": "mentor", "advisor": "mentor"}

# (регэксп вопроса с группой s - субъект) -> поле итоговой сущности ("" = сама сущность)
TEMPLATES = [
    (r"In which year was (?P<s>.+?) born\?", "born"),
    (r"When was (?P<s>.+?) born\?", "born"),
    (r"In which year was (?P<s>.+?) founded\?", "founded"),
    (r"When was (?P<s>.+?) founded\?", "founded"),
    (r"In which city is the headquarters of (?P<s>.+?)\?", "headquarters"),
    (r"In which city is (?P<s>.+?) headquartered\?", "headquarters"),
    (r"Where is the headquarters of (?P<s>.+?)\?", "headquarters"),
    (r"Where is (?P<s>.+?) headquartered\?", "headquarters"),
    (r"Who mentors (?P<s>.+?)\?", "mentor"),
    (r"Who is the mentor of (?P<s>.+?)\?", "mentor"),
    (r"Who is (?P<s>.+?)\?", ""),
    (r"(?:Which company|Where) does (?P<s>.+?) work(?: for| at)?\?", "employer"),
    (r"Which company employs (?P<s>.+?)\?", "employer"),
    (r"What is the name of (?P<s>.+?)\?", ""),
]


def load_kb(ws: Path):
    kb = ws / "kb"
    if not kb.is_dir():
        return None, {}
    ents, files = {}, {}
    for f in sorted(kb.rglob("*.md")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^#\s*(.+)$", txt, re.MULTILINE)
        if not m:
            continue
        name = m.group(1).strip()
        fields = {}
        for k, v in re.findall(r"^([A-Za-z _-]+):\s*(.+?)\s*$", txt, re.MULTILINE):
            fields[k.strip().lower()] = v.strip()
        ents[name.lower()] = {"name": name, "fields": fields}
        files[name.lower()] = txt.strip()
    return ents, files


def resolve_subject(s, ents):
    """'Sergei Garcia's employer' -> сущность Fjord Data. None если цепочка рвётся."""
    s = s.strip().strip("?").strip()
    s = re.sub(r"^the\s+", "", s, flags=re.IGNORECASE)
    parts = re.split(r"['’]s\s+", s)
    base = parts[0].strip()
    cur = ents.get(base.lower())
    if cur is None:
        return None, "missing:" + base
    for rel in parts[1:]:
        rel = rel.strip().lower()
        key = REL.get(rel)
        if key is None:
            return None, "unknown_rel:" + rel
        val = cur["fields"].get(key)
        if not val:
            return None, "missing_field:" + key
        nxt = ents.get(val.lower())
        if nxt is None:
            # значение поля известно, но файла связанной сущности нет
            return {"name": val, "fields": {}}, "dangling"
        cur = nxt
    return cur, "ok"


def fast_solve(question, ents):
    for pat, attr in TEMPLATES:
        m = re.fullmatch(pat, question.strip(), flags=re.IGNORECASE)
        if not m:
            continue
        ent, status = resolve_subject(m.group("s"), ents)
        if ent is None:
            if status.startswith(("missing:", "missing_field")):
                return "IMPOSSIBLE"
            continue
        if attr == "":
            # «Who is X's mentor?» - ответ = имя; «Who is X?» без связи - не наш случай
            if "'" not in m.group("s") and "’" not in m.group("s"):
                continue
            return ent["name"]
        val = ent["fields"].get(attr)
        if val:
            return val
        # поле нужно, а его нет: у сущности нет поля или файла связанной сущности нет в базе
        return "IMPOSSIBLE" if status in ("ok", "dangling") else None
    return None


def mentioned(question, ents):
    q = question.lower()
    hits = [k for k in ents if k in q]
    # убираем вложенные совпадения ("kira" внутри "kira novak")
    return [h for h in hits if not any(h != o and h in o for o in hits)]


def gather_context(question, ents, files, hops=2):
    seen, frontier = [], mentioned(question, ents)
    for _ in range(hops + 1):
        nxt = []
        for k in frontier:
            if k in seen:
                continue
            seen.append(k)
            for v in ents[k]["fields"].values():
                if v.lower() in ents and v.lower() not in seen:
                    nxt.append(v.lower())
        frontier = nxt
    return "\n\n".join(files[k] for k in seen)


def analyze(task, fast_path=True):
    p = Plan()
    ents, files = load_kb(task.workspace)
    q = extract_question(task.instruction)
    p.notes["question"] = q
    if ents is None:
        p.est_cost = 1500
        p.p_success = 0.5
        p.notes["generic"] = True
        return p
    if fast_path:
        a = fast_solve(q, ents)
        if a == "IMPOSSIBLE":
            p.feasible = False
            return p
        if a is not None:
            p.fast_answer = a
            return p
    if not mentioned(q, ents):
        p.feasible = False  # ни одна сущность из вопроса не найдена в базе
        return p
    p.context = gather_context(q, ents, files)
    p.est_cost = (approx_tokens(p.context) + 120) * 0.1 + 25
    p.p_success = 0.85
    return p


PROMPT = ("Knowledge base excerpt:\n{ctx}\n\nQuestion: {q}\n"
          "Answer using only the excerpt. " + ANSWER_RULES)


def solve_llm(task, llm, plan, strong=False):
    from .. import config
    if plan.notes.get("generic"):
        from .generic import solve_generic
        return solve_generic(task, llm, plan)
    msgs = [{"role": "user", "content": PROMPT.format(ctx=plan.context, q=plan.notes["question"])}]
    text, _cost, _meta = llm.chat(msgs, model=config.MODEL_STRONG if strong else config.MODEL_FAST,
                                max_tokens=config.MAX_OUT_STRONG if strong else 40)
    ans = clean_answer(text.strip().splitlines()[-1] if text.strip() else "")
    return None if is_unsolvable(ans) else ans
