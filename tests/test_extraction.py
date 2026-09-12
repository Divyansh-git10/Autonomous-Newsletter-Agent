"""
Tests for tools/article_extractor.py and the extraction-gating logic
in agent/content.py.
"""

import httpx

from agent.content import content_enrichment_node
from tests.conftest import make_article
from tools import article_extractor


def test_wrapper_url_is_rejected_by_extractor_itself():
    """Defense in depth: even if a wrapper URL reaches the extractor
    directly, it must refuse rather than scrape the interstitial page."""

    result = article_extractor.extract_article_text(
        "https://news.google.com/rss/articles/fake?oc=5"
    )
    assert result["content_available"] is False
    assert result["extraction_status"] == "skipped_invalid_url"


def test_direct_article_extraction_succeeds(monkeypatch):
    class FakeResponse:
        text = "<html><body>" + ("Real article content. " * 40) + "</body></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(article_extractor.httpx, "get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(
        article_extractor.trafilatura,
        "extract",
        lambda *a, **k: "Real article content. " * 40,
    )

    result = article_extractor.extract_article_text("https://example.com/article")
    assert result["extraction_status"] == "success"
    assert result["content_available"] is True
    assert len(result["content"]) > 200


def test_http_403_is_handled_gracefully(monkeypatch):
    def raise_403(*args, **kwargs):
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError("403 Forbidden", request=request, response=response)

    monkeypatch.setattr(article_extractor.httpx, "get", raise_403)

    result = article_extractor.extract_article_text("https://example.com/blocked")
    assert result["content_available"] is False
    assert result["extraction_status"] == "failed_http_error"


def test_timeout_is_handled_gracefully(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(article_extractor.httpx, "get", raise_timeout)

    result = article_extractor.extract_article_text("https://example.com/slow")
    assert result["content_available"] is False
    assert result["extraction_status"] == "failed_timeout"


def test_empty_extraction_marked_unavailable(monkeypatch):
    class FakeResponse:
        text = "<html><body></body></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(article_extractor.httpx, "get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(article_extractor.trafilatura, "extract", lambda *a, **k: "")

    result = article_extractor.extract_article_text("https://example.com/empty")
    assert result["content_available"] is False
    assert result["extraction_status"] == "failed_empty"
    assert result["content"] == ""


def test_content_enrichment_skips_extraction_for_unresolved_articles(monkeypatch):
    """A Google News wrapper URL (unresolved) must never be passed to
    the extractor at all."""

    calls = []

    def fake_extract(url):
        calls.append(url)
        return {"content": "x" * 300, "content_available": True, "extraction_status": "success", "source_url": url}

    monkeypatch.setattr("agent.content.extract_article_text", fake_extract)

    unresolved_article = make_article(title="Unresolved Story", verified=False, idx=1)
    verified_article = make_article(title="Verified Story", verified=True, idx=2)

    result = content_enrichment_node({"raw_articles": [unresolved_article, verified_article]})

    # Only the verified article's resolved_url should ever reach the extractor.
    assert calls == [verified_article["resolved_url"]]

    enriched_unresolved = result["raw_articles"][0]
    assert enriched_unresolved["extraction_status"] == "skipped_wrapper_url"
    assert enriched_unresolved["content_available"] is False
    assert enriched_unresolved["is_title_only"] is True

    enriched_verified = result["raw_articles"][1]
    assert enriched_verified["extraction_status"] == "success"
    assert enriched_verified["content_available"] is True
    assert enriched_verified["is_title_only"] is False


def test_recall_mode_fallback_used_when_precision_mode_underextracts(monkeypatch):
    """Some publisher templates (AWS's blog template among them) cause
    trafilatura's precision mode to under-extract. The recall-mode retry
    (Tier 2) should be used instead when it returns more content."""

    class FakeResponse:
        text = "<html><body>" + ("Full article body text here. " * 40) + "</body></html>"

        def raise_for_status(self):
            return None

    def fake_extract(html, **kwargs):
        if kwargs.get("favor_recall"):
            return "Full article body text here. " * 40
        return "Too short."

    monkeypatch.setattr(article_extractor.httpx, "get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(article_extractor.trafilatura, "extract", fake_extract)

    result = article_extractor.extract_article_text("https://aws.amazon.com/blogs/machine-learning/example")
    assert result["extraction_status"] == "success"
    assert result["content_available"] is True
    assert "Full article body text here." in result["content"]


def test_content_container_fallback_used_when_trafilatura_fails_entirely(monkeypatch):
    """When both trafilatura modes return nothing (e.g. an unusual page
    template), Tier 3 should still recover real content from a common
    <article>/<main> container rather than falling straight to a noisy
    full-page strip or giving up."""

    article_html = (
        "<html><body><nav>Skip nav links</nav>"
        "<article>" + ("This is the real AWS blog post content. " * 20) + "</article>"
        "<footer>Copyright footer text</footer></body></html>"
    )

    class FakeResponse:
        text = article_html

        def raise_for_status(self):
            return None

    monkeypatch.setattr(article_extractor.httpx, "get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(article_extractor.trafilatura, "extract", lambda *a, **k: "")

    result = article_extractor.extract_article_text("https://aws.amazon.com/blogs/machine-learning/example")
    assert result["extraction_status"] == "success"
    assert result["content_available"] is True
    assert "real AWS blog post content" in result["content"]
    assert "Skip nav links" not in result["content"]
    assert "Copyright footer text" not in result["content"]


def test_waf_challenge_page_is_detected_as_blocked(monkeypatch):
    """A CloudFront/WAF-style bot-challenge page should be recognized as
    blocked rather than treated as real (if odd) article content."""

    challenge_text = "Request unsuccessful. Incapsula incident ID: Reference ID: 12.34abc56"

    class FakeResponse:
        text = "<html><body>" + (challenge_text + " ") * 10 + "</body></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(article_extractor.httpx, "get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(article_extractor.trafilatura, "extract", lambda *a, **k: challenge_text * 10)

    result = article_extractor.extract_article_text("https://example.com/blocked-by-waf")
    assert result["content_available"] is False
    assert result["extraction_status"] == "failed_blocked"
    assert result["content"] == ""
