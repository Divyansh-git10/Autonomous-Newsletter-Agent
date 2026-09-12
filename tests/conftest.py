"""
Shared test fixtures. Nothing here makes a real network or LLM call.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


_SAMPLE_STRUCTURED_SUMMARY = (
    '{"what_happened": "The article describes a new AI agent framework release.", '
    '"key_takeaways": ["The framework supports multi-step planning.", '
    '"It integrates with existing tool APIs."], '
    '"why_it_matters": "It expands what AI agents can automate.", '
    '"source_limitations": "None noted."}'
)


class FakeLLM:
    """
    Returns a canned, valid response based on which prompt is being
    asked, so planner / summarizer / editor's-takeaway / critic can all
    be exercised end-to-end offline.
    """

    def __init__(self, critic_needs_revision: bool = False, critic_issues=None):
        self.critic_needs_revision = critic_needs_revision
        self.critic_issues = critic_issues or []
        self.calls: list[str] = []

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        self.calls.append(text)

        if "planning component" in text:
            return FakeResponse('{"search_queries": ["ai agent news", "ai agent safety"]}')

        if "strict but practical editor" in text:
            import json

            payload = {
                "needs_revision": self.critic_needs_revision,
                "issues": self.critic_issues,
                "suggestions": [],
                "quality_score": 40 if self.critic_needs_revision else 90,
            }
            return FakeResponse(json.dumps(payload))

        if "Editor's Takeaway" in text and "Verified articles:" in text:
            return FakeResponse(
                "The verified articles this week show continued progress on "
                "AI agent tooling and multi-step planning frameworks."
            )

        # Falls through to the structured per-article summarizer prompt.
        return FakeResponse(_SAMPLE_STRUCTURED_SUMMARY)


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    for module_path in [
        "tools.llm_utils",
        "agent.planner",
        "agent.writer",
        "agent.critic",
        "tools.summarizer",
    ]:
        monkeypatch.setattr(f"{module_path}.get_llm", lambda *a, **k: fake, raising=False)
    return fake


def make_article(
    title="Example AI Agent Story",
    verified=True,
    content_long=True,
    idx=1,
    provider="google_news",
):
    """Builds an article dict shaped like tools.news_search.news_search_tool's output."""

    rss_url = f"https://news.google.com/rss/articles/fake-token-{idx}?oc=5"
    resolved_url = f"https://example.com/article-{idx}" if verified else ""

    article = {
        "title": title,
        "source": "Example Source",
        "source_url": "https://example.com",
        "published": "Mon, 01 Sep 2026 00:00:00 GMT",
        "published_ts": 1798761600.0 + idx,
        "summary": f"Article titled '{title}' published by Example Source.",
        "provider": provider,
        "rss_url": rss_url,
        "url": resolved_url if verified else rss_url,
        "resolved_url": resolved_url,
        "url_resolution_status": "verified_direct" if verified else "unresolved",
        "url_resolution_method": "redirect" if verified else "none",
    }
    return article


@pytest.fixture
def fake_articles():
    return [make_article(title="First AI Agent Story", verified=True, idx=1),
            make_article(title="Second AI Agent Story", verified=True, idx=2)]


@pytest.fixture
def mock_pipeline_io(monkeypatch, fake_articles):
    """Mocks the two network-touching tools used by the graph nodes."""

    def fake_news_search_tool(query, max_results=10):
        return [dict(a) for a in fake_articles]

    def fake_extract_article_text(url):
        return {
            "content": (
                "This is a long, realistic article body describing a new AI "
                "agent framework release in detail. " * 5
            ),
            "content_available": True,
            "extraction_status": "success",
            "source_url": url,
        }

    monkeypatch.setattr("agent.research.news_search_tool", fake_news_search_tool)
    monkeypatch.setattr("agent.content.extract_article_text", fake_extract_article_text)
