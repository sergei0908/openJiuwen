"""Вопросы к базе знаний kb/*.md.

Бесплатно: парсим kb в словарь сущностей и разрешаем цепочки вида
"X's employer's headquarters". Если имени нет в базе - задача невыполнима, токены не тратим.
LLM-путь: в промпт идут только файлы упомянутых сущностей и их связей (2 шага по графу).
"""
import re
from pathlib import Path

from .base import (
    FINAL_RULES,
    Plan,
    agent_answer,
    approx_tokens,
    extract_question,
    strip_boilerplate,
)

REL = {"employer": "employer", "company": "employer", "mentor": "mentor",
       "boss": "mentor", "advisor": "mentor"}

# (регэксп вопроса с группой s - субъект) -> поле итоговой сущности ("" = сама сущность)
TEMPLATES = [
    (r"In which year was (?P<s>.+?) born\?", "born"),
    (r"When was (?P<s>.+?) born\?", "born"),
    (r"What is the birth year of (?P<s>.+?)\?", "born"),
    (r"What year was (?P<s>.+?) born\?", "born"),
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


KB_EXT = ("*.md", "*.txt", "*.markdown")


def kb_files(ws: Path, instruction=""):
    """Файлы базы знаний: kb/, иначе папка или файл, названные в условии, иначе все .md/.txt задачи."""
    def listed(d):
        return sorted(p for ext in KB_EXT for p in d.rglob(ext) if p.is_file()
                      and not any(part.startswith(".") for part in p.relative_to(ws).parts))
    if (ws / "kb").is_dir():
        return listed(ws / "kb")
    for name in re.findall(r"([\w.-]+(?:/[\w.-]+)*)/?", instruction):
        p = ws / name
        if name and p.is_dir() and p != ws:
            return listed(p)
        if p.is_file() and p.suffix in (".md", ".txt", ".markdown"):
            return [p]
    return listed(ws)


# «Born: 1984», «- Born: 1984», «* Born: 1984», «**Born:** 1984», «**Born**: 1984»
FIELD_RE = re.compile(r"^\s*(?:[-*•]\s+)?(?:\*\*|__)?([A-Za-z][A-Za-z _-]*?)(?:\*\*|__)?\s*:\s*(?:\*\*|__)?\s*(.+?)\s*$",
                      re.MULTILINE)
# «Arktika Labs, Inc.» - в вопросах и в полях других сущностей обычно просто «Arktika Labs»
CORP_SUFFIX = re.compile(r",?\s+(?:Inc|Ltd|LLC|Corp|Co|GmbH|PLC|AG|S\.A)\.?$", re.IGNORECASE)


def load_kb(ws: Path, instruction=""):
    """Сущности базы: заголовок «# Имя» и поля «Ключ: значение» (в том числе пунктом списка и жирным).
    В одном файле может быть много сущностей подряд - делим по заголовкам первого уровня. Факты,
    записанные предложениями, полями не станут - такие вопросы решает модель по файлу сущности."""
    ents, files = {}, {}
    for f in kb_files(ws, instruction):
        txt = f.read_text(encoding="utf-8", errors="replace")
        for part in re.split(r"(?m)^(?=#\s)", txt):
            m = re.search(r"^#\s*(.+)$", part, re.MULTILINE)
            if not m:
                continue
            name = m.group(1).strip()
            fields = {}
            for k, v in FIELD_RE.findall(part):
                fields[k.strip().lower()] = v.strip().strip("*_").strip()
            ent = {"name": name, "fields": fields}
            ents[name.lower()] = ent
            files[name.lower()] = part.strip()
            alias = CORP_SUFFIX.sub("", name).strip()
            if alias and alias != name and alias.lower() not in ents:
                ents[alias.lower()] = ent
                files[alias.lower()] = part.strip()
    if not ents:
        return None, {}  # базу не нашли или не поняли - универсальный путь, а не «невыполнимо»
    return ents, files


# Одно поле может называться по-разному: «HQ», «Established», «Birth year».
FIELD_SYN = {
    "born": ("born", "birth year", "year of birth", "date of birth", "birth date", "birthday"),
    "founded": ("founded", "established", "year founded", "founding year", "founded in"),
    "headquarters": ("headquarters", "hq", "head office", "headquartered in", "based in"),
    "employer": ("employer", "works at", "works for", "employed by", "company"),
    "mentor": ("mentor", "mentored by", "advisor", "adviser"),
}


def get_field(ent, key):
    """Значение поля по его имени или синониму; None, если такого поля у сущности нет."""
    for k in FIELD_SYN.get(key, (key,)):
        v = ent["fields"].get(k)
        if v:
            return v
    return None


OF_CHAIN = re.compile(r"^(?:the\s+)?(?P<rel>\w+)\s+of\s+(?P<rest>.+)$", re.IGNORECASE)


def to_possessive(s):
    """'the employer of the mentor of X' -> "X's mentor's employer" (только для известных связей)."""
    s = s.strip()
    m = re.match(r"^(?:the\s+)?(?:company|organi[sz]ation|firm) (?:that|which) employs (?P<rest>.+)$", s, re.IGNORECASE)
    if m:
        return f"{to_possessive(m.group('rest'))}'s employer"
    m = re.match(r"^(?:the\s+)?(?:person|one) (?:who|that) mentors (?P<rest>.+)$", s, re.IGNORECASE)
    if m:
        return f"{to_possessive(m.group('rest'))}'s mentor"
    m = OF_CHAIN.match(s)
    if m and m.group("rel").lower() in REL:
        return f"{to_possessive(m.group('rest'))}'s {m.group('rel').lower()}"
    return re.sub(r"^the\s+", "", s, flags=re.IGNORECASE)


def resolve_subject(s, ents):
    """'Sergei Garcia's employer' -> сущность Fjord Data. None если цепочка рвётся."""
    s = s.strip().strip("?").strip()
    s = to_possessive(s)
    parts = re.split(r"['’]s\s+", s)
    base = parts[0].strip()
    cur = ents.get(base.lower())
    if cur is None:
        return None, "missing:" + base
    for rel in parts[1:]:
        rel = rel.strip().lower()
        # поле с таким именем есть («Boss: …») - берём его, а не синоним (boss -> mentor)
        key = rel if rel in cur["fields"] else REL.get(rel)
        if key is None:
            return None, "unknown_rel:" + rel
        val = cur["fields"].get(key) or get_field(cur, key)
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
            if status.startswith("missing:") and mentioned(question, ents):
                # кто-то из базы в вопросе есть - значит, мы не поняли формулировку, а не данных нет
                return None
            if status.startswith("missing:"):
                return "IMPOSSIBLE"  # самого человека или компании в базе нет
            if status.startswith("missing_field"):
                # сущность есть, а поля под знакомым именем нет - факт может быть записан иначе
                # («was born in 1984»); решает модель по файлу сущности, а не «невыполнимо»
                return None
            continue
        if attr == "":
            # «Who is X's mentor?» - ответ = имя; «Who is X?» без связи - не наш случай
            if "'" not in m.group("s") and "’" not in m.group("s"):
                continue
            return ent["name"]
        val = get_field(ent, attr)
        if val:
            return field_answer(attr, val, question)
        # файла связанной сущности нет в базе - узнать её поля неоткуда
        return "IMPOSSIBLE" if status == "dangling" else None
    return None


YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def field_answer(attr, val, question=""):
    """Ответ из значения поля. Год из полной даты («12 March 1984» -> 1984) - только если спрашивают
    год: на «When was X born?» эталон может быть полной датой, такое решает модель. Город и страну
    вместе («Osaka, Japan») тоже отдаём модели: какая часть нужна, решает вопрос."""
    if attr in ("born", "founded"):
        if YEAR.fullmatch(val.strip()):
            return val.strip()
        years = YEAR.findall(val)
        return years[0] if len(years) == 1 and "year" in question.lower() else None
    if attr == "headquarters" and "," in val:
        return None
    return val


def mentioned(question, ents):
    q = question.lower()
    hits = [k for k in ents if k in q]
    # убираем вложенные совпадения ("kira" внутри "kira novak")
    return [h for h in hits if not any(h != o and h in o for o in hits)]


NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:[ -][A-Z][a-z]+)+)\b")
NOT_NAMES = {"in", "the", "which", "what", "who", "where", "when", "how", "use", "is", "does", "did", "a", "an"}
KB_INTRO = re.compile(r"(?:The directory \S+ contains[^.]*\.|The file \S+ contains[^.]*\.|Use the file tools[^.]*\.)",
                      re.IGNORECASE)


def task_text(instruction):
    """Условие целиком, без вступления про базу и общего хвоста: вопрос может опираться на соседние
    предложения («Rosa Sato works for a firm. In which city is that firm based?»)."""
    return KB_INTRO.sub("", strip_boilerplate(instruction)).strip()


def unknown_names(text, ents):
    """Имена собственные из текста, которых нет в базе («Victor Kraus», когда в базе Viktor Krasnov).
    Известным считается и то, что встречается в значениях полей: город «New York» - не неизвестный
    человек, а место из базы."""
    values = [v.lower() for e in ents.values() for v in e["fields"].values()]
    out = []
    for m in NAME_RE.finditer(text):
        words = m.group(1).split()
        while words and words[0].lower() in NOT_NAMES:
            words = words[1:]
        name = " ".join(words).lower()
        if len(words) >= 2 and name not in ents and not any(name in v for v in values):
            out.append(" ".join(words))
    return out


def gather_context(question, ents, files, hops=2):
    start = mentioned(question, ents)
    # обратные связи: «Who works at Polar Bit?» - нужны люди, у которых Employer: Polar Bit
    # (только для вопросов «кто…» / «сколько людей…», иначе контекст разрастается зря)
    back = []
    if re.search(r"\b(?:who|whom|which (?:people|person|employees?)|how many (?:people|persons|employees))\b",
                 question, re.IGNORECASE):
        back = [k for k, e in ents.items() if k not in start
                and any(v.lower() in start for v in e["fields"].values())]
    seen, frontier = [], start + back
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
    return "\n\n".join(dict.fromkeys(files[k] for k in seen))  # у компании может быть два ключа (с «Inc.»)


def analyze(task, fast_path=True):
    p = Plan()
    ents, files = load_kb(task.workspace, task.instruction)
    q = extract_question(task.instruction)
    full = task_text(task.instruction)
    p.notes["question"] = q
    p.notes["task_text"] = full
    if ents is None:
        p.est_cost = 1500
        p.p_success = 0.5
        p.notes["generic"] = True
        return p
    if fast_path:
        a = fast_solve(q, ents)
        if a == "IMPOSSIBLE" and not mentioned(q, ents) and mentioned(full, ents):
            a = None  # вопрос ссылается на соседнее предложение («that person») - разбор его не понял
        if a == "IMPOSSIBLE":
            p.feasible = False
            return p
        if a is not None:
            p.fast_answer = a
            return p
    if not mentioned(full, ents):
        if unknown_names(full, ents):
            p.feasible = False  # в условии названо имя, которого в базе нет
            return p
        # никого не назвали («сколько людей работает в компаниях из Казани?») - нужна вся база
        p.context = "\n\n".join(dict.fromkeys(files.values()))[:8000]
    else:
        p.context = gather_context(full, ents, files)
    p.est_cost = (approx_tokens(p.context) + 120) * 0.1 + 25
    p.p_success = 0.85
    return p


SYSTEM = ("You answer a question about a small knowledge base. Use only the excerpt given in the "
          "message; follow links between people and companies. Do not write out the steps: reply with "
          "the single line 'FINAL: <answer>'. " + FINAL_RULES)
MULTI_HOP = (" This question needs several links. If you must check them, keep each step to a few words "
             "on one line; the last line must be 'FINAL: <answer>'.")


def chain_len(question):
    """Сколько связей пройти: «the employer of X's mentor» -> 2."""
    q = question or ""
    return len(re.findall(r"'s\b|’s\b", q)) + len(re.findall(r"\b(?:of|that employs|who mentors)\b", q)) - \
        int(bool(re.search(r"\b(?:headquarters|year|city|name) of\b", q)))


def solve_llm(task, llm, plan, strong=False):
    if plan.notes.get("generic"):
        from .generic import solve_generic
        return solve_generic(task, llm, plan)
    q = plan.notes["question"]
    query = f"Knowledge base excerpt:\n{plan.context}\n\nQuestion: {plan.notes.get('task_text') or q}"
    # Длинные цепочки модель любит расписывать по шагам и при 60 токенах обрывается до «FINAL:»
    # (замер 9.11: все 17 ошибок на 120 задачах - вопросы «headquarters of the employer of X's mentor»).
    # Им даём запас на короткие шаги; простым вопросам хватает 60.
    multi = chain_len(q) >= 2
    # контекст уже собран харнессом - инструменты не нужны, хватает одного шага
    return agent_answer(llm, system_prompt=SYSTEM + (MULTI_HOP if multi else ""), query=query, tools=[],
                        strong=strong, max_tokens=220 if multi else 60, max_iterations=1)
