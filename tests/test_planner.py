"""
Tests for agent/planner.py: query generation must not hardcode a stale
past year, and must anchor on the real current date.
"""

from datetime import datetime, timezone

from agent.planner import planner_node
from tests.conftest import FakeLLM


def test_planner_prompt_includes_real_current_date(monkeypatch):
    captured_prompts = []

    class CapturingLLM(FakeLLM):
        def invoke(self, prompt):
            captured_prompts.append(prompt)
            return super().invoke(prompt)

    fake = CapturingLLM()
    monkeypatch.setattr("agent.planner.get_llm", lambda *a, **k: fake)

    planner_node({"goal": "Create a weekly AI agent newsletter.", "search_attempt": 0})

    assert captured_prompts, "planner did not call the LLM"
    current_year = str(datetime.now(timezone.utc).year)
    assert current_year in captured_prompts[0]


def test_planner_prompt_forbids_hardcoded_past_year(monkeypatch):
    captured_prompts = []

    class CapturingLLM(FakeLLM):
        def invoke(self, prompt):
            captured_prompts.append(prompt)
            return super().invoke(prompt)

    fake = CapturingLLM()
    monkeypatch.setattr("agent.planner.get_llm", lambda *a, **k: fake)

    planner_node({"goal": "Create a weekly AI agent newsletter.", "search_attempt": 0})

    prompt_lower = captured_prompts[0].lower()
    assert "do not hardcode or assume any specific past year" in prompt_lower


def test_planner_fallback_queries_contain_no_stale_year():
    """Even the safe fallback plan (used when LLM JSON parsing fails)
    must not reference a specific past year."""

    import agent.planner as planner_module

    class BrokenLLM:
        def invoke(self, prompt):
            class R:
                content = "not valid json"
            return R()

    import tools.llm_utils

    original = tools.llm_utils.get_llm
    tools.llm_utils.get_llm = lambda *a, **k: BrokenLLM()
    planner_module.get_llm = lambda *a, **k: BrokenLLM()
    try:
        result = planner_node({"goal": "Create a weekly AI agent newsletter.", "search_attempt": 0})
    finally:
        tools.llm_utils.get_llm = original
        planner_module.get_llm = original

    for query in result["search_queries"]:
        assert "2023" not in query
        assert "2024" not in query


def test_planner_does_not_crash_when_llm_call_itself_raises(monkeypatch, capsys):
    """Regression test: a Groq rate-limit/network failure during the
    LLM *call* itself (not just bad JSON) must not crash the planner --
    it must fall back to the safe default query plan."""

    class RateLimitedLLM:
        def invoke(self, prompt):
            raise Exception("Error code: 429 - rate_limit_exceeded")

    monkeypatch.setattr("agent.planner.get_llm", lambda *a, **k: RateLimitedLLM())

    result = planner_node({"goal": "Create a weekly AI agent newsletter.", "search_attempt": 0})

    assert result["search_queries"]  # fell back to the safe default plan, did not crash
    for query in result["search_queries"]:
        assert "2023" not in query
        assert "2024" not in query

    captured = capsys.readouterr()
    assert "rate limit" in captured.out.lower()
