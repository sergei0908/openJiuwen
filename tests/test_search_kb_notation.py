"""База знаний, записанная не строго «Ключ: значение»: ложного «невыполнимо» быть не должно."""
import pytest

from harness.solvers.search_solver import analyze, field_answer, load_kb, unknown_names
from harness.tasks import Task

PERSON = {
    "kv": "Born: 1984\nEmployer: Arktika Labs\nMentor: Hiro Petrov\n",
    "bullets": "- Born: 1984\n- Employer: Arktika Labs\n- Mentor: Hiro Petrov\n",
    "stars": "* Born: 1984\n* Employer: Arktika Labs\n* Mentor: Hiro Petrov\n",
    "bold": "**Born:** 1984\n\n**Employer:** Arktika Labs\n\n**Mentor:** Hiro Petrov\n",
    "bold_key": "**Born**: 1984\n**Employer**: Arktika Labs\n**Mentor**: Hiro Petrov\n",
    "synonyms": "Birth year: 1984\nWorks at: Arktika Labs\nMentored by: Hiro Petrov\n",
}
COMPANY = {
    "kv": "Headquarters: Osaka\nFounded: 1982\n",
    "bullets": "- Headquarters: Osaka\n- Founded: 1982\n",
    "stars": "* Headquarters: Osaka\n* Founded: 1982\n",
    "bold": "**Headquarters:** Osaka\n\n**Founded:** 1982\n",
    "bold_key": "**Headquarters**: Osaka\n**Founded**: 1982\n",
    "synonyms": "HQ: Osaka\nEstablished: 1982\n",
}


def make(tmp_path, question, person, company, company_title="Arktika Labs", extra=None):
    kb = tmp_path / "w" / "kb"
    kb.mkdir(parents=True)
    (kb / "rosa_sato.md").write_text(f"# Rosa Sato\n\n{person}")
    (kb / "hiro_petrov.md").write_text("# Hiro Petrov\n\nBorn: 1961\nEmployer: Volga Data\n")
    (kb / "arktika_labs.md").write_text(f"# {company_title}\n\n{company}")
    (kb / "volga_data.md").write_text("# Volga Data\n\nHeadquarters: New York\n")
    for name, text in (extra or {}).items():
        (kb / name).write_text(text)
    instr = "The directory kb/ contains a small knowledge base of markdown files. " + question + \
        " Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory."
    return Task("search_x", tmp_path / "w", "search", instr, 1.0)


@pytest.mark.parametrize("style", sorted(PERSON))
@pytest.mark.parametrize("question,expected", [
    ("In which year was Rosa Sato born?", "1984"),
    ("Which company employs Rosa Sato?", "Arktika Labs"),
    ("In which city is the headquarters of Rosa Sato's employer?", "Osaka"),
    ("When was the company that employs Rosa Sato founded?", "1982"),
])
def test_other_notations_are_solved(tmp_path, style, question, expected):
    p = analyze(make(tmp_path, question, PERSON[style], COMPANY[style]))
    assert p.feasible is not False
    assert p.fast_answer == expected


def test_facts_in_sentences_go_to_the_model(tmp_path):
    p = analyze(make(tmp_path, "In which year was Rosa Sato born?",
                     "Rosa Sato was born in 1984. She works at Arktika Labs.\n", COMPANY["kv"]))
    assert p.feasible is not False and p.fast_answer is None
    assert "was born in 1984" in p.context


def test_corporate_suffix_in_title(tmp_path):
    p = analyze(make(tmp_path, "In which city is Arktika Labs headquartered?", PERSON["kv"], COMPANY["kv"],
                     company_title="Arktika Labs, Inc."))
    assert p.fast_answer == "Osaka"


def test_field_named_like_the_relation_wins_over_synonym(tmp_path):
    person = PERSON["kv"] + "Boss: Olga Novak\n"
    p = analyze(make(tmp_path, "Who is Rosa Sato's boss?", person, COMPANY["kv"],
                     extra={"olga_novak.md": "# Olga Novak\n\nBorn: 1975\n"}))
    assert p.fast_answer == "Olga Novak"
    p = analyze(make(tmp_path / "b", "Who is Rosa Sato's boss?", PERSON["kv"], COMPANY["kv"]))
    assert p.fast_answer == "Hiro Petrov"   # поля Boss нет - как раньше, синоним mentor


def test_full_date_gives_year_and_city_country_goes_to_model():
    assert field_answer("born", "12 March 1984") == "1984"
    assert field_answer("born", "1984") == "1984"
    assert field_answer("born", "between 1984 and 1986") is None
    assert field_answer("headquarters", "Osaka, Japan") is None
    assert field_answer("headquarters", "Osaka") == "Osaka"


def test_city_from_kb_is_not_an_unknown_name(tmp_path):
    t = make(tmp_path, "Which company is headquartered in New York?", PERSON["kv"], COMPANY["kv"])
    p = analyze(t)
    assert p.feasible is not False
    assert "Volga Data" in p.context


def test_unknown_person_is_still_infeasible(tmp_path):
    p = analyze(make(tmp_path, "In which year was Victor Kraus born?", PERSON["kv"], COMPANY["kv"]))
    assert p.feasible is False
    ents, _ = load_kb(tmp_path / "w")
    assert unknown_names("Where does Victor Kraus work?", ents) == ["Victor Kraus"]


def test_dangling_link_is_still_infeasible(tmp_path):
    person = "Born: 1984\nEmployer: Ghost Corp\n"     # файла Ghost Corp в базе нет
    p = analyze(make(tmp_path, "In which city is the headquarters of Rosa Sato's employer?", person,
                     COMPANY["kv"]))
    assert p.feasible is False
