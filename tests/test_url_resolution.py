"""
Tests for the Google News URL resolution pipeline in
tools/news_search.py. All HTTP calls are mocked.
"""

from tools import news_search


def test_is_google_news_wrapper_detected():
    assert news_search.is_google_news_wrapper(
        "https://news.google.com/rss/articles/abc?oc=5"
    )
    assert not news_search.is_google_news_wrapper("https://example.com/article")


def test_validate_candidate_url_accepts_direct_url():
    assert news_search.validate_candidate_url("https://example.com/article") is True


def test_validate_candidate_url_rejects_wrapper_and_search_engines():
    assert news_search.validate_candidate_url("https://news.google.com/rss/articles/x") is False
    assert news_search.validate_candidate_url("https://duckduckgo.com/html/?q=x") is False
    assert news_search.validate_candidate_url("https://www.bing.com/search?q=x") is False


def test_validate_candidate_url_rejects_malformed():
    assert news_search.validate_candidate_url("") is False
    assert news_search.validate_candidate_url("not-a-url") is False
    assert news_search.validate_candidate_url("ftp://example.com/file") is False


def test_resolve_article_url_accepts_direct_rss_link(monkeypatch):
    """Some RSS providers already give a direct (non-Google-News) link."""

    result = news_search.resolve_article_url(
        google_news_url="https://example.com/already-direct",
        title="Some Title",
        source="Example",
    )
    assert result["status"] == "verified_direct"
    assert result["method"] == "direct_rss_link"
    assert result["resolved_url"] == "https://example.com/already-direct"


def test_resolve_article_url_via_redirect(monkeypatch):
    class FakeResponse:
        url = "https://example.com/real-article"
        text = "<html></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(news_search.httpx, "get", lambda *a, **k: FakeResponse())

    result = news_search.resolve_article_url(
        google_news_url="https://news.google.com/rss/articles/token?oc=5",
        title="Some Title",
        source="Example",
    )
    assert result["status"] == "verified_direct"
    assert result["method"] == "redirect"
    assert result["resolved_url"] == "https://example.com/real-article"


def test_resolve_article_url_falls_back_to_secondary_search(monkeypatch):
    """Redirect stays on Google's own domain, but the secondary search succeeds."""

    class FakeRedirectResponse:
        url = "https://news.google.com/rss/articles/token?oc=5"
        text = "<html><body>no canonical link here</body></html>"

        def raise_for_status(self):
            return None

    class FakeSearchResponse:
        text = '<a class="result__a" href="https://example.com/found-article">link</a>'

        def raise_for_status(self):
            return None

    call_log = []

    def fake_get(url, *args, **kwargs):
        call_log.append(url)
        if "duckduckgo" in url:
            return FakeSearchResponse()
        return FakeRedirectResponse()

    monkeypatch.setattr(news_search.httpx, "get", fake_get)

    result = news_search.resolve_article_url(
        google_news_url="https://news.google.com/rss/articles/token?oc=5",
        title="Some Title",
        source="Example",
    )
    assert result["status"] == "verified_direct"
    assert result["method"] == "secondary_search"
    assert result["resolved_url"] == "https://example.com/found-article"


def test_resolve_article_url_all_strategies_fail_safely(monkeypatch):
    """When redirect resolution and secondary search both fail, the
    original Google News URL is preserved as unresolved -- never a
    fabricated URL."""

    def fake_get(*args, **kwargs):
        raise Exception("403 Forbidden")

    monkeypatch.setattr(news_search.httpx, "get", fake_get)

    result = news_search.resolve_article_url(
        google_news_url="https://news.google.com/rss/articles/token?oc=5",
        title="Some Title",
        source="Example",
    )
    assert result["status"] == "unresolved"
    assert result["resolved_url"] == ""
    assert result["method"] == "none"


def test_secondary_search_rejects_invalid_candidates(monkeypatch):
    """A candidate that is itself a search-engine page must be rejected."""

    class FakeSearchResponse:
        text = '<a class="result__a" href="https://duckduckgo.com/y.js?u=x">bad link</a>'

        def raise_for_status(self):
            return None

    monkeypatch.setattr(news_search.httpx, "get", lambda *a, **k: FakeSearchResponse())

    candidate = news_search.search_original_article_url(title="Some Title", source="Example")
    assert candidate == ""


# --------------------------------------------------------------------
# Multi-provider architecture (direct_rss, bing_news, orchestration)
# --------------------------------------------------------------------

_SAMPLE_RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Fake Feed</title>
<item>
  <title>New AI Agent Framework Ships With Multi-Step Planning</title>
  <link>https://example-publisher.com/ai-agent-framework</link>
  <description>A detailed look at the new AI agent framework release.</description>
  <pubDate>Mon, 01 Sep 2026 00:00:00 GMT</pubDate>
</item>
<item>
  <title>Totally Unrelated Sports Recap</title>
  <link>https://example-publisher.com/sports-recap</link>
  <description>A recap of last night's game.</description>
  <pubDate>Mon, 01 Sep 2026 00:00:00 GMT</pubDate>
</item>
</channel></rss>"""


class _FakeFeedResponse:
    def __init__(self, content: str):
        self.content = content.encode("utf-8")

    def raise_for_status(self):
        return None


def test_query_keywords_strips_stopwords():
    keywords = news_search._query_keywords("the latest AI agent news")
    assert "agent" in keywords
    assert "the" not in keywords
    assert "latest" not in keywords
    assert "news" not in keywords


def test_search_direct_rss_filters_by_query_and_returns_verified_direct(monkeypatch):
    monkeypatch.setattr(
        news_search.httpx,
        "get",
        lambda *a, **k: _FakeFeedResponse(_SAMPLE_RSS_FEED),
    )

    results = news_search._search_direct_rss("AI agent framework", max_results=10)

    assert len(results) >= 1
    titles = [r["title"] for r in results]
    assert any("AI Agent Framework" in t for t in titles)
    assert all("Sports Recap" not in t for t in titles)
    for article in results:
        assert article["url_resolution_status"] == "verified_direct"
        assert article["url_resolution_method"] == "direct_rss_feed"
        assert article["provider"] == "direct_rss"
        assert news_search.validate_candidate_url(article["url"])


def test_search_direct_rss_returns_empty_when_feeds_fail(monkeypatch):
    def fake_get(*args, **kwargs):
        raise Exception("network blocked")

    monkeypatch.setattr(news_search.httpx, "get", fake_get)

    results = news_search._search_direct_rss("AI agent framework", max_results=10)
    assert results == []


def test_resolve_bing_url_reads_plain_query_param():
    link = (
        "https://www.bing.com/news/apiclick.aspx?ID=abc123"
        "&url=https%3A%2F%2Fexample.com%2Freal-article"
    )
    result = news_search._resolve_bing_url(link)

    assert result["status"] == "verified_direct"
    assert result["method"] == "bing_query_param"
    assert result["resolved_url"] == "https://example.com/real-article"


def test_resolve_bing_url_accepts_already_direct_link():
    result = news_search._resolve_bing_url("https://example.com/already-direct")
    assert result["status"] == "verified_direct"
    assert result["method"] == "direct_rss_link"


def test_resolve_bing_url_falls_back_to_redirect(monkeypatch):
    class FakeResponse:
        url = "https://example.com/resolved-via-redirect"
        text = "<html></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(news_search.httpx, "get", lambda *a, **k: FakeResponse())

    link = "https://www.bing.com/news/apiclick.aspx?ID=abc123"  # no url= param
    result = news_search._resolve_bing_url(link)

    assert result["status"] == "verified_direct"
    assert result["method"] == "redirect"
    assert result["resolved_url"] == "https://example.com/resolved-via-redirect"


def test_resolve_bing_url_unresolved_when_nothing_works(monkeypatch):
    def fake_get(*args, **kwargs):
        raise Exception("blocked")

    monkeypatch.setattr(news_search.httpx, "get", fake_get)

    link = "https://www.bing.com/news/apiclick.aspx?ID=abc123"
    result = news_search._resolve_bing_url(link)

    assert result["status"] == "unresolved"
    assert result["resolved_url"] == ""


def test_get_configured_providers_defaults(monkeypatch):
    monkeypatch.delenv("NEWS_SEARCH_PROVIDERS", raising=False)
    assert news_search.get_configured_providers() == ["direct_rss", "google_news", "bing_news"]


def test_get_configured_providers_reads_env_var(monkeypatch):
    monkeypatch.setenv("NEWS_SEARCH_PROVIDERS", "bing_news, direct_rss")
    assert news_search.get_configured_providers() == ["bing_news", "direct_rss"]


def test_get_configured_providers_falls_back_on_garbage_env_var(monkeypatch):
    monkeypatch.setenv("NEWS_SEARCH_PROVIDERS", "not_a_real_provider, also_fake")
    assert news_search.get_configured_providers() == ["direct_rss", "google_news", "bing_news"]


def test_news_search_tool_tags_provider_and_dedupes(monkeypatch):
    def fake_direct_rss(query, max_results):
        return [
            {
                "title": "A",
                "source": "S",
                "source_url": "https://s.com",
                "published": "",
                "published_ts": 2.0,
                "summary": "s",
                "provider": "direct_rss",
                "rss_url": "https://example.com/shared-url",
                "url": "https://example.com/shared-url",
                "resolved_url": "https://example.com/shared-url",
                "url_resolution_status": "verified_direct",
                "url_resolution_method": "direct_rss_feed",
            }
        ]

    def fake_google_news(query, max_results):
        # Same URL as direct_rss already found -- must be deduped, not
        # counted twice.
        return [
            {
                "title": "A duplicate",
                "source": "S",
                "source_url": "https://s.com",
                "published": "",
                "published_ts": 1.0,
                "summary": "s",
                "provider": "google_news",
                "rss_url": "https://example.com/shared-url",
                "url": "https://example.com/shared-url",
                "resolved_url": "https://example.com/shared-url",
                "url_resolution_status": "verified_direct",
                "url_resolution_method": "redirect",
            },
            {
                "title": "B",
                "source": "S",
                "source_url": "https://s.com",
                "published": "",
                "published_ts": 1.0,
                "summary": "s",
                "provider": "google_news",
                "rss_url": "https://news.google.com/rss/articles/x",
                "url": "https://news.google.com/rss/articles/x",
                "resolved_url": "",
                "url_resolution_status": "unresolved",
                "url_resolution_method": "none",
            },
        ]

    monkeypatch.setitem(news_search._PROVIDER_FUNCS, "direct_rss", fake_direct_rss)
    monkeypatch.setitem(news_search._PROVIDER_FUNCS, "google_news", fake_google_news)
    monkeypatch.setitem(news_search._PROVIDER_FUNCS, "bing_news", lambda q, m: [])

    results = news_search.news_search_tool(
        "test query", max_results=10, providers=["direct_rss", "google_news", "bing_news"]
    )

    urls = [a["url"] for a in results]
    assert urls.count("https://example.com/shared-url") == 1
    assert any(a["provider"] == "direct_rss" for a in results)
    assert any(a["provider"] == "google_news" and a["title"] == "B" for a in results)


def test_news_search_tool_stops_once_enough_verified_direct(monkeypatch):
    call_log = []

    def fake_direct_rss(query, max_results):
        call_log.append("direct_rss")
        return [
            {
                "title": f"Article {i}",
                "source": "S",
                "source_url": "",
                "published": "",
                "published_ts": float(i),
                "summary": "s",
                "provider": "direct_rss",
                "rss_url": f"https://example.com/a{i}",
                "url": f"https://example.com/a{i}",
                "resolved_url": f"https://example.com/a{i}",
                "url_resolution_status": "verified_direct",
                "url_resolution_method": "direct_rss_feed",
            }
            for i in range(3)
        ]

    def fake_google_news(query, max_results):
        call_log.append("google_news")
        return []

    monkeypatch.setitem(news_search._PROVIDER_FUNCS, "direct_rss", fake_direct_rss)
    monkeypatch.setitem(news_search._PROVIDER_FUNCS, "google_news", fake_google_news)

    results = news_search.news_search_tool(
        "test query", max_results=3, providers=["direct_rss", "google_news"]
    )

    assert len(results) == 3
    assert "google_news" not in call_log  # short-circuited once max_results was met


def test_news_search_tool_returns_empty_when_all_providers_blocked(monkeypatch):
    def blocked(*args, **kwargs):
        raise Exception("blocked")

    monkeypatch.setitem(news_search._PROVIDER_FUNCS, "direct_rss", blocked)
    monkeypatch.setitem(news_search._PROVIDER_FUNCS, "google_news", blocked)
    monkeypatch.setitem(news_search._PROVIDER_FUNCS, "bing_news", blocked)

    results = news_search.news_search_tool(
        "test query", max_results=5, providers=["direct_rss", "google_news", "bing_news"]
    )

    assert results == []


def test_news_search_tool_skips_unknown_provider_name(monkeypatch, capsys):
    monkeypatch.setitem(news_search._PROVIDER_FUNCS, "direct_rss", lambda q, m: [])

    results = news_search.news_search_tool(
        "test query", max_results=5, providers=["not_a_real_provider", "direct_rss"]
    )

    assert results == []
    captured = capsys.readouterr()
    assert "Unknown provider" in captured.out
