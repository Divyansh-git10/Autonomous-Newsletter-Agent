"""
Article Extraction tool.

Extracts readable article text from a *verified direct* publisher URL
and reports back structured status metadata instead of a bare string,
so callers can tell "no content" apart from "didn't even try" apart
from "tried and it was blocked". An empty extraction is never silently
treated as valid content.
"""

import httpx
import trafilatura
from bs4 import BeautifulSoup

MIN_CONTENT_LENGTH = 200

# Cheap heuristics for pages that "extracted" text but that text is a
# block/paywall/consent page rather than the article itself. Extended
# to also cover common WAF/CDN bot-challenge pages (e.g. AWS's own
# CloudFront/WAF, Cloudflare, Akamai) that a plain httpx GET without a
# real browser fingerprint can trigger on some publisher sites.
_BLOCKED_PAGE_MARKERS = [
    "enable javascript",
    "please enable cookies",
    "access denied",
    "are you a robot",
    "captcha",
    "subscribe to continue reading",
    "subscribe now to continue",
    "sign in to continue reading",
    "this content is not available in your region",
    "request unsuccessful",
    "reference id:",
    "checking your browser",
    "javascript is not available",
    "verify you are a human",
    "unusual traffic",
    "attention required",
]

# Content containers commonly used by real article templates, tried in
# order before falling back to stripping the whole page. This is a
# second-tier fallback for sites (AWS's blog template among them) where
# trafilatura's precision-favoring heuristics can under-extract due to
# heavy sidebar/related-post/comment boilerplate around the actual
# article body.
_CONTENT_CONTAINER_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    "#content",
    ".content",
    "#main-content",
    ".post-content",
    ".entry-content",
    ".blog-post-content",
]


def _looks_blocked(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _BLOCKED_PAGE_MARKERS)


def _extract_from_content_container(html: str) -> str:
    """
    Looks for a common article-content container and returns its
    cleaned text, or "" if none of the selectors match. This never
    invents content -- it only narrows which part of the already-
    fetched HTML is treated as the article body.
    """

    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        for selector in _CONTENT_CONTAINER_SELECTORS:
            container = soup.select_one(selector)
            if container:
                text = container.get_text(separator=" ", strip=True)
                if text and len(text) >= MIN_CONTENT_LENGTH:
                    return text
    except Exception as exc:
        print(f"Content-container extraction failed: {exc}")

    return ""


def extract_article_text(url: str) -> dict:
    """
    Returns a structured result:

    {
        "content": str,              # "" unless extraction genuinely succeeded
        "content_available": bool,
        "extraction_status": str,    # see below
        "source_url": str,
    }

    extraction_status is one of:
        "success", "failed_http_error", "failed_timeout",
        "failed_blocked", "failed_empty", "skipped_invalid_url"
    """

    if not url or "news.google.com" in url:
        # Defense in depth: a Google News wrapper URL should never reach
        # this function (agent/content.py is responsible for gating
        # this), but if it does, refuse rather than scrape an
        # interstitial page.
        return {
            "content": "",
            "content_available": False,
            "extraction_status": "skipped_invalid_url",
            "source_url": url or "",
        }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = httpx.get(
            url,
            timeout=20,
            follow_redirects=True,
            headers=headers,
        )
        response.raise_for_status()
        html = response.text

    except httpx.TimeoutException as exc:
        print(f"Extraction timed out for {url}: {exc}")
        return {
            "content": "",
            "content_available": False,
            "extraction_status": "failed_timeout",
            "source_url": url,
        }

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(f"HTTP {status} extracting {url}: {exc}")
        return {
            "content": "",
            "content_available": False,
            "extraction_status": "failed_http_error",
            "source_url": url,
        }

    except httpx.HTTPError as exc:
        print(f"HTTP extraction failed for {url}: {exc}")
        return {
            "content": "",
            "content_available": False,
            "extraction_status": "failed_http_error",
            "source_url": url,
        }

    except Exception as exc:
        print(f"Unexpected extraction failure for {url}: {exc}")
        return {
            "content": "",
            "content_available": False,
            "extraction_status": "failed_http_error",
            "source_url": url,
        }

    extracted_text = ""
    extraction_method = "trafilatura_precision"

    try:
        extracted_text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        ) or ""
    except Exception as exc:
        print(f"Trafilatura (precision mode) extraction failed for {url}: {exc}")

    # Tier 2: some publisher templates (heavy sidebar/related-post
    # boilerplate around the real article body -- observed on AWS's
    # blog template, among others) cause trafilatura's precision mode
    # to under-extract or return nothing. Retry in recall mode, which
    # is more permissive about what counts as article content, before
    # falling back further.
    if len(extracted_text.strip()) < MIN_CONTENT_LENGTH:
        try:
            recall_text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=False,
                favor_recall=True,
            ) or ""
            if len(recall_text.strip()) > len(extracted_text.strip()):
                extracted_text = recall_text
                extraction_method = "trafilatura_recall"
        except Exception as exc:
            print(f"Trafilatura (recall mode) extraction failed for {url}: {exc}")

    # Tier 3: look for a common article-content container (<article>,
    # <main>, role="main", common content div classes/ids) and use its
    # text specifically, rather than the whole page. This can succeed
    # where both trafilatura modes under-extract on an unusual template.
    if len(extracted_text.strip()) < MIN_CONTENT_LENGTH:
        container_text = _extract_from_content_container(html)
        if len(container_text.strip()) > len(extracted_text.strip()):
            extracted_text = container_text
            extraction_method = "content_container"

    # Tier 4 (last resort): strip the whole page down to visible text.
    # Noisiest option -- only used if nothing more targeted worked.
    if not extracted_text.strip():
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
                tag.decompose()
            extracted_text = soup.get_text(separator=" ", strip=True)
            extraction_method = "full_page_strip"
        except Exception as exc:
            print(f"Fallback HTML extraction failed for {url}: {exc}")
            extracted_text = ""

    cleaned = extracted_text.strip()

    if not cleaned:
        print(f"Extraction produced no usable text for {url} after all fallback tiers.")
        return {
            "content": "",
            "content_available": False,
            "extraction_status": "failed_empty",
            "source_url": url,
        }

    if _looks_blocked(cleaned):
        print(f"Extraction for {url} looks like a bot-challenge/consent page, not article content.")
        return {
            "content": "",
            "content_available": False,
            "extraction_status": "failed_blocked",
            "source_url": url,
        }

    if len(cleaned) < MIN_CONTENT_LENGTH:
        # Too short to be meaningful article content -- do not silently
        # treat this as valid.
        print(f"Extraction for {url} produced only {len(cleaned)} characters (below the {MIN_CONTENT_LENGTH}-char minimum).")
        return {
            "content": "",
            "content_available": False,
            "extraction_status": "failed_empty",
            "source_url": url,
        }

    if extraction_method != "trafilatura_precision":
        print(f"Extraction for {url} succeeded via fallback tier: {extraction_method}.")

    return {
        "content": cleaned,
        "content_available": True,
        "extraction_status": "success",
        "source_url": url,
    }


if __name__ == "__main__":
    test_url = "https://www.example.com"
    result = extract_article_text(test_url)
    print(f"Status: {result['extraction_status']}")
    print(f"Content available: {result['content_available']}")
    print(f"Extracted characters: {len(result['content'])}")
