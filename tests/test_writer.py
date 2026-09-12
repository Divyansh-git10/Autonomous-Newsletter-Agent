"""
Tests for agent/writer.py: deterministic rendering of title-only
articles, correct source labeling, and Editor's Takeaway grounding.
"""

from agent.grounding import (
    DETERMINISTIC_EDITORS_TAKEAWAY,
    build_fallback_summary,
    build_llm_failed_summary,
)
from agent.writer import build_editors_takeaway, render_article_section, write_newsletter
from tests.conftest import make_article


def _grounded(article, idx=1):
    return {
        **article,
        "content_available": True,
        "grounding_status": "grounded",
        "what_happened": f"Something specific happened, item {idx}.",
        "key_takeaways": [f"Point {idx}a", f"Point {idx}b"],
        "why_it_matters": "It matters for AI agents.",
        "source_limitations": "None noted.",
    }


def _title_only(article):
    return {**article, **build_fallback_summary(article), "content_available": False}


def _llm_failed_with_content(article):
    return {**article, **build_llm_failed_summary(article)}


def test_title_only_article_uses_exact_fallback_wording():
    article = _title_only(make_article(verified=False))
    section = render_article_section(1, article)

    assert "No details were provided in the source beyond the title." in section
    assert "No key takeaways available because the article content could not be retrieved." in section
    assert "no factual claims were" in section.lower()


def test_google_news_reference_not_labeled_as_original_article():
    article = _title_only(make_article(verified=False))
    section = render_article_section(1, article)

    assert "Google News reference" in section
    assert "Verified publisher article" not in section
    assert "original article" not in section.lower()


def test_verified_article_labeled_as_verified_publisher():
    article = _grounded(make_article(verified=True))
    section = render_article_section(1, article)

    assert "Verified publisher article" in section
    assert "Google News reference" not in section


def test_bing_news_unresolved_article_labeled_as_bing_reference():
    article = _title_only(make_article(verified=False, provider="bing_news"))
    section = render_article_section(1, article)

    assert "Bing News reference" in section
    assert "Google News reference" not in section
    assert "Verified publisher article" not in section


def test_unknown_provider_unresolved_article_labeled_generically():
    article = _title_only(make_article(verified=False, provider="some_future_provider"))
    section = render_article_section(1, article)

    assert "Unverified reference" in section


def test_grounded_article_never_gets_fallback_wording():
    article = _grounded(make_article(verified=True))
    section = render_article_section(1, article)

    assert "No details were provided in the source beyond the title." not in section
    assert "Something specific happened, item 1." in section


def test_llm_failed_article_distinguished_from_no_content_article():
    """Regression test: an article whose content WAS extracted but
    whose LLM summarization call failed must not be described as
    title-only/no-content -- it must clearly say the content was
    available but summarization could not be completed."""

    article = _llm_failed_with_content(make_article(verified=True))
    section = render_article_section(1, article)

    assert "No details were provided in the source beyond the title." not in section
    assert "successfully retrieved" in section.lower()
    assert "summar" in section.lower() and "could not be" in section.lower()


def test_editors_takeaway_deterministic_when_mostly_title_only(fake_llm):
    articles = [
        _title_only(make_article(title="A", verified=False, idx=1)),
        _title_only(make_article(title="B", verified=False, idx=2)),
        _grounded(make_article(title="C", verified=True, idx=3), idx=3),
    ]

    takeaway = build_editors_takeaway("goal", articles)

    assert takeaway == DETERMINISTIC_EDITORS_TAKEAWAY
    # The LLM must never even be asked to synthesize from mostly-unsupported data.
    assert fake_llm.calls == []


def test_editors_takeaway_llm_synthesis_when_majority_grounded(fake_llm):
    articles = [
        _grounded(make_article(title="A", verified=True, idx=1), idx=1),
        _grounded(make_article(title="B", verified=True, idx=2), idx=2),
        _title_only(make_article(title="C", verified=False, idx=3)),
    ]

    takeaway = build_editors_takeaway("goal", articles)

    assert takeaway != DETERMINISTIC_EDITORS_TAKEAWAY
    assert len(fake_llm.calls) == 1


def test_write_newsletter_full_document_structure(fake_llm):
    articles = [_grounded(make_article(title="A", verified=True, idx=1), idx=1)]
    newsletter = write_newsletter("goal", articles)

    assert newsletter.startswith("# Weekly AI Agent Insights")
    assert "## 1. A" in newsletter
    assert "## Editor's Takeaway" in newsletter
