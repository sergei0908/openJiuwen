"""Ответ агента без «FINAL:»: обрывок рассуждений в answer.txt не пишем."""
from harness.solvers.base import agent_answer, plausible_answer


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.queries = list(replies), []

    def agent(self, *, system_prompt, query, tools, model=None, max_iterations=4, max_tokens=800, stop=None):
        self.queries.append(query)
        return self.replies.pop(0)


def ask(replies, tools=()):
    llm = FakeLLM(replies)
    return agent_answer(llm, system_prompt="s", query="q", tools=list(tools)), llm


def test_final_marker_is_used():
    assert ask(["thinking\nFINAL: Osaka"])[0] == "Osaka"


def test_plain_short_last_line_is_still_accepted():
    assert ask(["Osaka"])[0] == "Osaka"


def test_cut_reasoning_triggers_one_cheap_retry():
    ans, llm = ask(["1. **Find Sergei**\n2. **Identify the employer of Sergei:", "FINAL: Kazan"])
    assert ans == "Kazan"
    assert len(llm.queries) == 2 and "exactly one line" in llm.queries[1]


def test_retry_without_final_gives_no_answer():
    ans, _ = ask(["2. **Identify the employer of Sergei:", "Let me think about Sergei's employer..."])
    assert ans is None


def test_with_tools_no_retry_and_no_garbage():
    ans, llm = ask(["Step 1: read the CSV file and filter the rows by region:"], tools=["python"])
    assert ans is None and len(llm.queries) == 1


def test_plausible_answer():
    for good in ("1983", "Osaka", "Fjord Data", "27.50", "-3"):
        assert plausible_answer(good), good
    long_text = "The answer depends on many different factors in the data"
    for bad in ("2. **Identify the employer of Sergei:", "- Sergei works at", long_text, ""):
        assert not plausible_answer(bad), bad
