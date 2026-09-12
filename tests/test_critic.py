"""
Tests for agent/critic.py and the grounding validators it relies on.
"""

from agent.critic import critique_newsletter
from agent.grounding import build_fallback_summary, validate_newsletter_grounding
from tests.conftest import FakeLLM, make_article


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


def test_validator_flags_unsupported_claim_in_title_only_article():
    article = _title_only(make_article(title="Salesforce News", verified=False))
    newsletter = (
        "# Weekly AI Agent Insights\n\n"
        "## 1. Salesforce News\n\n"
        "**What happened:**\n"
        "Salesforce announced a major new AI agent platform today.\n\n"
        "**Key takeaways:**\n- It was announced today.\n\n"
        "**Why it matters:**\nBig news.\n\n"
        "**Source limitations:**\nNone.\n\n"
        "**Source:** [Google News reference](https://news.google.com/rss/articles/x)\n\n"
        "## Editor's Takeaway\n\nSomething.\n"
    )

    issues = validate_newsletter_grounding(newsletter, [article])
    assert any("factual-claim language" in issue for issue in issues)


def test_validator_flags_mislabeled_source():
    article = _title_only(make_article(title="Some Story", verified=False))
    newsletter = (
        "# Weekly AI Agent Insights\n\n"
        "## 1. Some Story\n\n"
        "**What happened:**\nNo details were provided in the source beyond the title.\n\n"
        "**Key takeaways:**\n- No key takeaways available because the article content could not be retrieved.\n\n"
        "**Why it matters:**\nUnclear.\n\n"
        "**Source limitations:**\nUnavailable.\n\n"
        "**Source:** [Verified publisher article](https://news.google.com/rss/articles/x)\n\n"
        "## Editor's Takeaway\n\nSomething.\n"
    )

    issues = validate_newsletter_grounding(newsletter, [article])
    assert any("labeled as a verified/original article" in issue for issue in issues)


def test_validator_passes_clean_grounded_newsletter():
    article = _grounded(make_article(title="Grounded Story", verified=True))
    newsletter = (
        "# Weekly AI Agent Insights\n\n"
        "## 1. Grounded Story\n\n"
        "**What happened:**\nSomething specific happened, item 1.\n\n"
        "**Key takeaways:**\n- Point 1a\n- Point 1b\n\n"
        "**Why it matters:**\nIt matters for AI agents.\n\n"
        "**Source limitations:**\nNone noted.\n\n"
        "**Source:** [Verified publisher article](https://example.com/article-1)\n\n"
        "## Editor's Takeaway\n\nThe verified article shows progress in agent tooling.\n"
    )

    issues = validate_newsletter_grounding(newsletter, [article])
    assert issues == []


def test_validator_flags_unsupported_editors_takeaway_generalization():
    articles = [
        _title_only(make_article(title="A", verified=False, idx=1)),
        _title_only(make_article(title="B", verified=False, idx=2)),
    ]
    newsletter = (
        "# Weekly AI Agent Insights\n\n"
        "## 1. A\n\nNo details were provided in the source beyond the title.\n"
        "**Key takeaways:**\n- No key takeaways available because the article content could not be retrieved.\n\n"
        "## 2. B\n\nNo details were provided in the source beyond the title.\n"
        "**Key takeaways:**\n- No key takeaways available because the article content could not be retrieved.\n\n"
        "## Editor's Takeaway\n\n"
        "This week's articles show a clear trend toward healthcare regulation in AI.\n"
    )

    issues = validate_newsletter_grounding(newsletter, articles)
    assert any("generalizes a trend" in issue for issue in issues)


def test_critic_triggers_revision_for_unsupported_claim(monkeypatch):
    article = _title_only(make_article(title="Salesforce News", verified=False))
    newsletter = (
        "# Weekly AI Agent Insights\n\n"
        "## 1. Salesforce News\n\n"
        "**What happened:**\nSalesforce announced a major new AI agent platform today.\n\n"
        "**Key takeaways:**\n- It was announced today.\n\n"
        "**Why it matters:**\nBig news.\n\n"
        "**Source limitations:**\nNone.\n\n"
        "**Source:** [Google News reference](https://news.google.com/rss/articles/x)\n\n"
        "## Editor's Takeaway\n\nSomething.\n"
    )

    fake = FakeLLM(critic_needs_revision=False)  # LLM itself thinks it's fine
    monkeypatch.setattr("agent.critic.get_llm", lambda *a, **k: fake)

    result = critique_newsletter(goal="goal", newsletter=newsletter, articles=[article])

    # Deterministic grounding checks must override the LLM's own opinion.
    assert result["needs_revision"] is True
    assert result["quality_score"] <= 40


def test_critic_passes_valid_grounded_summary(monkeypatch):
    article = _grounded(make_article(title="Grounded Story", verified=True))
    newsletter = (
        "# Weekly AI Agent Insights\n\n"
        "## 1. Grounded Story\n\n"
        "**What happened:**\nSomething specific happened, item 1, with enough "
        "detail in this sentence to make the section substantive rather than a "
        "one-liner, since the critic also runs a minimum-length structural check.\n\n"
        "**Key takeaways:**\n- Point 1a with a bit more explanatory detail\n"
        "- Point 1b with a bit more explanatory detail\n\n"
        "**Why it matters:**\nIt matters for AI agents because it demonstrates "
        "continued investment in agent tooling and multi-step planning frameworks "
        "that developers can build on.\n\n"
        "**Source limitations:**\nNone noted.\n\n"
        "**Source:** [Verified publisher article](https://example.com/article-1)\n\n"
        "## Editor's Takeaway\n\nThe verified article shows continued progress in "
        "agent tooling this week, with practical implications for developers "
        "building multi-step autonomous workflows.\n"
    )

    fake = FakeLLM(critic_needs_revision=False)
    monkeypatch.setattr("agent.critic.get_llm", lambda *a, **k: fake)

    result = critique_newsletter(goal="goal", newsletter=newsletter, articles=[article])

    assert result["needs_revision"] is False
    assert result["quality_score"] > 40
