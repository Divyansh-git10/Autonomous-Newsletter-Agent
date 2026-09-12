"""
Tests for tools/summarizer.py and agent/summarize.py: content
availability must be checked BEFORE the LLM is ever invoked.
"""

from agent.grounding import needs_fallback
from agent.summarize import summarize_node
from tests.conftest import make_article
from tools.summarizer import summarize_article_structured


def _with_content(article, content, status="success", available=True):
    return {
        **article,
        "content": content,
        "content_available": available,
        "extraction_status": status,
        "is_title_only": not available,
    }


def test_needs_fallback_true_when_content_missing():
    article = make_article(verified=True)
    article["content_available"] = False
    article["is_title_only"] = True
    assert needs_fallback(article) is True


def test_needs_fallback_true_when_url_unresolved_even_with_text():
    article = make_article(verified=False)
    article["content"] = "some text " * 50
    article["content_available"] = True
    article["extraction_status"] = "success"
    # Still unresolved -- must not be trusted.
    assert needs_fallback(article) is True


def test_needs_fallback_false_for_verified_long_content():
    article = _with_content(make_article(verified=True), "Real content. " * 30)
    assert needs_fallback(article) is False


def test_summarizer_bypasses_llm_when_content_unavailable(monkeypatch, fake_llm):
    article = make_article(verified=False)
    article["content_available"] = False
    article["is_title_only"] = True

    result = summarize_article_structured(article)

    assert result["grounding_status"] == "fallback"
    assert result["content_available"] is False
    assert fake_llm.calls == []  # LLM must never be invoked
    assert "no key takeaways available" in result["key_takeaways"][0].lower()


def test_summarizer_invokes_llm_when_content_available(fake_llm):
    article = _with_content(make_article(verified=True), "Real detailed content. " * 30)

    result = summarize_article_structured(article)

    assert result["grounding_status"] == "grounded"
    assert result["content_available"] is True
    assert len(fake_llm.calls) == 1
    assert result["what_happened"]
    assert result["key_takeaways"]


def test_summarize_node_reports_correct_stats(fake_llm):
    grounded_article = _with_content(make_article(title="Grounded", verified=True, idx=1), "Real content. " * 30)
    fallback_article = make_article(title="Fallback", verified=False, idx=2)
    fallback_article["content_available"] = False
    fallback_article["is_title_only"] = True

    result = summarize_node({"raw_articles": [grounded_article, fallback_article]})

    stats = result["summary_stats"]
    assert stats["llm_grounded"] == 1
    assert stats["deterministic_fallback"] == 1

    ranked = {a["title"]: a for a in result["ranked_articles"]}
    assert ranked["Grounded"]["grounding_status"] == "grounded"
    assert ranked["Fallback"]["grounding_status"] == "fallback"


def test_summarizer_rate_limit_error_falls_back_without_being_grounded(monkeypatch, capsys):
    """A Groq 429/token-limit failure must fall back deterministically
    and must never be counted as a successful LLM-grounded summary.
    Since the article's content WAS successfully extracted (only the
    LLM call failed), content_available must remain True and the
    wording must say the content was available -- never "only the
    title was available"."""

    article = _with_content(make_article(verified=True), "Real detailed content. " * 30)

    class RateLimitedLLM:
        def invoke(self, prompt):
            raise Exception("Error code: 429 - {'error': 'rate_limit_exceeded'}")

    monkeypatch.setattr("tools.summarizer.get_llm", lambda *a, **k: RateLimitedLLM())

    result = summarize_article_structured(article)

    assert result["grounding_status"] == "error_fallback"
    assert result["content_available"] is True
    assert "successfully retrieved" in result["what_happened"].lower()
    assert "only the title" not in result["what_happened"].lower()

    captured = capsys.readouterr()
    assert "rate limit" in captured.out.lower()


def test_summarizer_generic_llm_exception_reports_content_was_available(monkeypatch):
    """A non-rate-limit LLM failure (e.g. a malformed/unparseable
    response) must use the same "content available, LLM failed"
    wording, distinct from the true no-content fallback."""

    article = _with_content(make_article(verified=True), "Real detailed content. " * 30)

    class BrokenLLM:
        def invoke(self, prompt):
            return type("R", (), {"content": "not valid json at all"})()

    monkeypatch.setattr("tools.summarizer.get_llm", lambda *a, **k: BrokenLLM())

    result = summarize_article_structured(article)

    assert result["grounding_status"] == "error_fallback"
    assert result["content_available"] is True
    assert "no details were provided in the source beyond the title" not in result["what_happened"].lower()
