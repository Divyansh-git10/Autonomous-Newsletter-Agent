"""
Unit tests for llm_utils and the news_search RSS-fetching path
(URL resolution itself is covered in tests/test_url_resolution.py,
extraction in tests/test_extraction.py).
"""

import pytest

from tools.llm_utils import get_llm, response_to_text


def test_get_llm_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError):
        get_llm()


def test_get_llm_succeeds_with_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    assert get_llm() is not None


def test_get_llm_uses_default_model_when_env_var_unset(monkeypatch):
    from tools.llm_utils import DEFAULT_GROQ_MODEL

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    llm = get_llm()
    assert llm.model_name == DEFAULT_GROQ_MODEL


def test_get_llm_honors_groq_model_env_var_override(monkeypatch):
    """The LLM model is configurable via GROQ_MODEL -- this is a
    genuine, tested override, not just documentation."""

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    llm = get_llm()
    assert llm.model_name == "llama-3.3-70b-versatile"


def test_response_to_text_handles_plain_string():
    class FakeResponse:
        content = "hello world"

    assert response_to_text(FakeResponse()) == "hello world"


def test_response_to_text_handles_list_content():
    class FakeResponse:
        content = [{"text": "part one"}, {"text": "part two"}]

    assert response_to_text(FakeResponse()) == "part one\npart two"


def test_news_search_tool_returns_resolved_articles_from_mocked_feed(monkeypatch):
    from tools import news_search

    fake_entries = [
        {
            "title": "Example AI Agent Story",
            "link": "https://news.google.com/rss/articles/fake?oc=5",
            "published": "Mon, 01 Sep 2026 00:00:00 GMT",
            "published_parsed": (2026, 9, 1, 0, 0, 0, 0, 0, 0),
            "summary": "<p>Some summary text.</p>",
            "source": {"title": "Example Source", "href": "https://example.com"},
        }
    ]

    class FakeFeed:
        entries = fake_entries

    class FakeHttpResponse:
        content = b"<xml></xml>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(news_search.httpx, "get", lambda *a, **k: FakeHttpResponse())
    monkeypatch.setattr(news_search.feedparser, "parse", lambda *a, **k: FakeFeed())
    monkeypatch.setattr(
        news_search,
        "resolve_article_url",
        lambda **k: {"resolved_url": "https://example.com/real-article", "status": "verified_direct", "method": "redirect"},
    )

    articles = news_search.news_search_tool("ai agents", max_results=5, providers=["google_news"])

    assert len(articles) == 1
    article = articles[0]
    assert article["title"] == "Example AI Agent Story"
    assert article["provider"] == "google_news"
    assert article["rss_url"] == "https://news.google.com/rss/articles/fake?oc=5"
    assert article["resolved_url"] == "https://example.com/real-article"
    assert article["url"] == "https://example.com/real-article"
    assert article["url_resolution_status"] == "verified_direct"
    assert article["published_ts"] > 0


def test_news_search_tool_keeps_google_news_url_when_unresolved(monkeypatch):
    from tools import news_search

    fake_entries = [
        {
            "title": "Unresolved Story",
            "link": "https://news.google.com/rss/articles/fake2?oc=5",
            "published": "Mon, 01 Sep 2026 00:00:00 GMT",
            "published_parsed": (2026, 9, 1, 0, 0, 0, 0, 0, 0),
            "summary": "<p>Some summary text.</p>",
            "source": {"title": "Example Source", "href": "https://example.com"},
        }
    ]

    class FakeFeed:
        entries = fake_entries

    class FakeHttpResponse:
        content = b"<xml></xml>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(news_search.httpx, "get", lambda *a, **k: FakeHttpResponse())
    monkeypatch.setattr(news_search.feedparser, "parse", lambda *a, **k: FakeFeed())
    monkeypatch.setattr(
        news_search,
        "resolve_article_url",
        lambda **k: {"resolved_url": "", "status": "unresolved", "method": "none"},
    )

    articles = news_search.news_search_tool("ai agents", max_results=5, providers=["google_news"])

    assert len(articles) == 1
    article = articles[0]
    assert article["url_resolution_status"] == "unresolved"
    assert article["resolved_url"] == ""
    # The original Google News URL must be retained as the reference link.
    assert article["url"] == "https://news.google.com/rss/articles/fake2?oc=5"


def test_is_rate_limit_error_detects_common_markers():
    from tools.llm_utils import is_rate_limit_error

    assert is_rate_limit_error(Exception("Error code: 429 - rate limit exceeded")) is True
    assert is_rate_limit_error(Exception("You have exceeded your daily token quota")) is True
    assert is_rate_limit_error(Exception("Rate limit reached for tokens per minute (TPM)")) is True


def test_is_rate_limit_error_does_not_flag_unrelated_errors():
    from tools.llm_utils import is_rate_limit_error

    assert is_rate_limit_error(Exception("Connection refused")) is False
    assert is_rate_limit_error(ValueError("invalid JSON in response")) is False


def test_get_llm_accepts_custom_max_tokens(monkeypatch):
    from tools.llm_utils import get_llm

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    llm = get_llm(max_tokens=500)
    assert llm.max_tokens == 500
