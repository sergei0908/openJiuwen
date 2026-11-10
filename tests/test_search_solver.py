"""Тесты решателя вопросов к базе знаний kb/*.md."""
import pytest

from harness.solvers.search_solver import fast_solve, gather_context, load_kb


def _person(name, born, employer, mentor):
    return f"# {name}\n\nBorn: {born}\nEmployer: {employer}\nMentor: {mentor}\n"


def _company(name, hq, founded):
    return f"# {name}\n\nHeadquarters: {hq}\nFounded: {founded}\n"


@pytest.fixture
def kb(tmp_path):
    """Маленькая база: два похожих имени (Irina Smith / Irina Smithson) и одна компания без файла."""
    d = tmp_path / "kb"
    d.mkdir()
    files = {
        "person_irina_smith.md": _person("Irina Smith", 1984, "Ember Bio", "Boris Novak"),
        "person_irina_smithson.md": _person("Irina Smithson", 1990, "Fjord Data", "Irina Smith"),
        "person_boris_novak.md": _person("Boris Novak", 1961, "Fjord Data", "Chen Wang"),
        "person_olga_petrov.md": _person("Olga Petrov", 1975, "Ghost Corp", "Boris Novak"),
        "company_ember_bio.md": _company("Ember Bio", "Osaka", 1999),
        "company_fjord_data.md": _company("Fjord Data", "Tartu", 2004),
    }
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    ents, _ = load_kb(tmp_path)
    return ents


def test_birth_year(kb):
    assert fast_solve("In which year was Boris Novak born?", kb) == "1961"


def test_employer_headquarters_chain(kb):
    q = "In which city is the headquarters of Irina Smith's employer?"
    assert fast_solve(q, kb) == "Osaka"


def test_false_match_similar_names(kb):
    # "Irina Smith" - подстрока "Irina Smithson", но ответ должен быть про нужного человека
    assert fast_solve("In which year was Irina Smithson born?", kb) == "1990"
    assert fast_solve("In which year was Irina Smith born?", kb) == "1984"


def test_mentor_chain(kb):
    # наставник Irina Smithson - Irina Smith, её работодатель - Ember Bio
    q = "In which city is the headquarters of Irina Smithson's mentor's employer?"
    assert fast_solve(q, kb) == "Osaka"


def test_of_chain_is_understood(kb):
    # «X of Y» вместо притяжательной формы, в том числе вперемешку
    q = "In which city is the headquarters of the employer of Irina Smithson's mentor?"
    assert fast_solve(q, kb) == "Osaka"
    q = "In which city is the headquarters of the employer of the mentor of Irina Smithson?"
    assert fast_solve(q, kb) == "Osaka"
    assert fast_solve("In which year was the mentor of Irina Smith born?", kb) == "1961"


def test_relative_clauses(kb):
    assert fast_solve("When was the company that employs Irina Smith founded?", kb) == "1999"
    assert fast_solve("What is the birth year of the person who mentors Irina Smith?", kb) == "1961"


def test_not_understood_phrase_with_known_person_is_not_impossible(kb):
    # формулировку не поняли, но человек из базы в вопросе есть - отдаём LLM, а не выбрасываем задачу
    q = "In which year was the closest colleague of Boris Novak born?"
    assert fast_solve(q, kb) is None


def test_missing_person_is_impossible(kb):
    # человека нет в базе - задача невыполнима, токены не тратим
    assert fast_solve("In which year was Maya Sato born?", kb) == "IMPOSSIBLE"


def test_company_without_file_is_not_guessed(kb):
    # у Olga Petrov работодатель Ghost Corp, но файла компании нет - точно ответить нельзя
    q = "In which city is the headquarters of Olga Petrov's employer?"
    assert fast_solve(q, kb) == "IMPOSSIBLE"


def test_unknown_question_goes_to_llm(kb):
    assert fast_solve("How many patents does Boris Novak hold?", kb) is None


def test_context_includes_linked_entities(tmp_path, kb):
    # для LLM-пути в контекст попадают человек и связанные с ним сущности, а не вся база
    _, files = load_kb(tmp_path)
    ctx = gather_context("Where does Boris Novak work?", kb, files)
    assert "# Boris Novak" in ctx
    assert "# Fjord Data" in ctx
    assert "# Olga Petrov" not in ctx


def test_chain_len():
    from harness.solvers.search_solver import chain_len
    assert chain_len("In which city is the headquarters of the employer of Rosa Ivanova's mentor?") == 2
    assert chain_len("Where is the headquarters of Irina Smith's employer?") == 1
    assert chain_len("In which year was Irina Smith born?") == 0


def test_multi_hop_question_gets_room_to_finish():
    # длинной цепочке нужен запас токенов, иначе ответ обрывается до «FINAL:»
    from harness.solvers.base import Plan
    from harness.solvers.search_solver import solve_llm

    class FakeLLM:
        def __init__(self):
            self.kw = []

        def agent(self, **kw):
            self.kw.append(kw)
            return "FINAL: Kazan"

    llm = FakeLLM()
    for q in ("In which city is the headquarters of the employer of Rosa Ivanova's mentor?",
              "In which year was Irina Smith born?"):
        p = Plan()
        p.context, p.notes["question"] = "# Rosa Ivanova", q
        assert solve_llm(None, llm, p) == "Kazan"
    assert [k["max_tokens"] for k in llm.kw] == [220, 60]
    assert "several links" in llm.kw[0]["system_prompt"]
    assert "several links" not in llm.kw[1]["system_prompt"]
