"""
News Search tool.

Searches for recent news articles across a configurable set of
*providers* and attempts to resolve each article's real publisher URL
through several *safe* strategies -- without ever decoding an opaque
redirect token and without ever fabricating a URL.

Providers (see get_configured_providers() / NEWS_SEARCH_PROVIDERS):

- "direct_rss": fetches curated publisher RSS feeds directly. These
  URLs are correct *by construction* -- there is no wrapper to resolve
  at all, so this is the most reliable provider whenever a feed
  actually covers the query's topic.
- "google_news": Google News RSS search. Google wraps every article
  link in an interstitial redirect URL. We try (a) following real HTTP
  redirects, then (b) parsing the interstitial page's own <link
  rel="canonical"> / <meta http-equiv="refresh"> tags, then (c) a
  DuckDuckGo secondary search by title+source. Google's opaque
  "rss/articles/<token>" value is never decoded or guessed -- if none
  of the above validates, the article is kept as an unresolved
  reference only.
- "bing_news": Bing News RSS search. Bing's own click-tracking URLs
  expose the real destination as a plain, human-readable "url="/"u="
  query parameter (not an opaque token), so that parameter is read
  directly; if absent, the same HTTP-redirect strategy used for Google
  is applied.

RSS metadata (`entry.source.href`) is captured too, but only ever used
as a soft *candidate hint* -- never presented as a verified article URL
by itself, per the "candidate, not exact" rule.

If nothing can be verified for a given article, the original RSS/
wrapper URL is preserved as `rss_url` / `url` and clearly marked
`url_resolution_status = "unresolved"` so downstream code treats it as
a reference link only, never as something to extract full article text
from. If every configured provider is blocked or returns nothing, this
module returns an empty list rather than fabricating results -- the
rest of the pipeline (see agent/content.py, tools/grounding.py) is
built to degrade safely to deterministic, clearly-labeled fallback
content in that case.
"""

import calendar
import os
from typing import Callable, TypedDict
from urllib.parse import parse_qs, quote, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/131.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Domains that must never be accepted as a "resolved" article URL --
# either a news aggregator's own wrapper, or a search engine's results
# page.
_REJECTED_DOMAINS = (
    "news.google.com",
    "google.com",
    "www.google.com",
    "duckduckgo.com",
    "html.duckduckgo.com",
    "bing.com",
    "www.bing.com",
)

# Curated, direct-link publisher RSS feeds. Every entry's <link> in
# these feeds is already a direct publisher URL -- there is no wrapper
# to resolve, so articles found here are "verified_direct" by
# construction. This list intentionally favors well-known technology /
# business publishers similar to the assignment's example domain (AI
# agent news for developers); it is easy to extend.
_DIRECT_RSS_FEEDS = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "VentureBeat": "https://venturebeat.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Wired": "https://www.wired.com/feed/rss",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "Engadget": "https://www.engadget.com/rss.xml",
    "ZDNET": "https://www.zdnet.com/news/rss.xml",
    "MIT Technology Review": "https://www.technologyreview.com/feed/",
    # Official company engineering/research blogs. Best-effort feed
    # URLs -- companies restructure their blogs and RSS availability
    # changes over time, so a stale/broken URL here simply fails this
    # one feed fetch (logged, then skipped, same as any other feed
    # failure) rather than breaking the provider or fabricating a
    # result. Run `python -m tools.diagnostics` to see which of these
    # actually resolve on a given network right now.
    "Microsoft AI Blog": "https://blogs.microsoft.com/ai/feed/",
    "Google AI Blog": "https://blog.google/technology/ai/rss/",
    "OpenAI News": "https://openai.com/news/rss.xml",
    "Anthropic News": "https://www.anthropic.com/rss.xml",
    "AWS Machine Learning Blog": "https://aws.amazon.com/blogs/machine-learning/feed/",
}

# Common stopwords stripped before matching query keywords against feed
# entries, so a query like "the latest AI agent news" matches on
# {"latest", "agent", "news"} rather than requiring an exact phrase.
_QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "about", "latest", "news", "today", "this", "week", "new",
}


class NewsArticle(TypedDict):
    title: str
    source: str
    source_url: str
    published: str
    published_ts: float
    summary: str
    provider: str
    rss_url: str
    url: str
    resolved_url: str
    url_resolution_status: str
    url_resolution_method: str


# --------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------

def clean_html(value: str) -> str:
    """Remove HTML tags from RSS summaries."""
    if not value:
        return ""
    return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)


def is_google_news_wrapper(url: str) -> bool:
    if not url:
        return False
    return "news.google.com" in urlparse(url).netloc


def validate_candidate_url(candidate_url: str) -> bool:
    """
    A candidate URL is only acceptable if it:
    - is well-formed http/https
    - is not still a Google/Bing wrapper
    - is not a search engine's own results/redirect page
    """

    if not candidate_url:
        return False

    parsed = urlparse(candidate_url)

    if parsed.scheme not in {"http", "https"}:
        return False

    if not parsed.netloc:
        return False

    netloc = parsed.netloc.lower()

    if any(rejected in netloc for rejected in _REJECTED_DOMAINS):
        return False

    return True


def _resolve_via_http_redirect(wrapper_url: str) -> str:
    """
    Generic strategy: follow real HTTP redirects for any wrapper URL
    (Google News or Bing News), and if the destination still isn't a
    direct publisher URL, look for a canonical link or meta-refresh URL
    embedded in that page's own HTML (a normal, safe HTML parse -- not
    decoding any opaque token).
    """

    try:
        response = httpx.get(
            wrapper_url,
            timeout=15,
            follow_redirects=True,
            headers=REQUEST_HEADERS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Redirect resolution request failed: {exc}")
        return ""
    except Exception as exc:
        print(f"Redirect resolution unexpected failure: {exc}")
        return ""

    final_url = str(response.url)

    if validate_candidate_url(final_url):
        return final_url

    # The wrapper served its own interstitial page -- look for a
    # canonical / refresh URL it embeds.
    try:
        soup = BeautifulSoup(response.text, "html.parser")

        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            candidate = canonical["href"].strip()
            if validate_candidate_url(candidate):
                return candidate

        refresh = soup.find("meta", attrs={"http-equiv": lambda v: v and v.lower() == "refresh"})
        if refresh and refresh.get("content"):
            content = refresh["content"]
            if "url=" in content.lower():
                candidate = content.split("url=", 1)[-1].strip().strip("'\"")
                if validate_candidate_url(candidate):
                    return candidate

    except Exception as exc:
        print(f"Redirect resolution HTML parse failed: {exc}")

    return ""


def get_source_domain(source_name: str) -> str:
    """Convert a known source name into a rough domain hint (soft candidate)."""

    source = source_name.lower().strip()

    mappings = {
        "reuters": "reuters.com",
        "siliconangle": "siliconangle.com",
        "the washington post": "washingtonpost.com",
        "washington post": "washingtonpost.com",
        "wsj": "wsj.com",
        "wall street journal": "wsj.com",
        "politico": "politico.com",
        "openai": "openai.com",
        "microsoft": "microsoft.com",
        "microsoft azure": "azure.microsoft.com",
        "techcrunch": "techcrunch.com",
        "venturebeat": "venturebeat.com",
        "anthropic": "anthropic.com",
        "nvidia": "nvidia.com",
        "salesforce": "salesforce.com",
    }

    return mappings.get(source, "")


def search_original_article_url(title: str, source: str) -> str:
    """
    Secondary fallback: search DuckDuckGo's HTML results page for the
    original publisher article, using the title + source as the query.
    Best-effort -- if DuckDuckGo blocks/rate-limits the request (a
    known, observed failure mode), this simply returns "" and the
    caller keeps the original reference URL.
    """

    source_domain = get_source_domain(source)
    search_query = f'"{title}" {source}'
    if source_domain:
        search_query += f" site:{source_domain}"

    search_url = f"https://html.duckduckgo.com/html/?q={quote(search_query)}"

    try:
        response = httpx.get(
            search_url,
            timeout=20,
            follow_redirects=True,
            headers=REQUEST_HEADERS,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.select("a.result__a"):
            href = (link.get("href") or "").strip()
            if validate_candidate_url(href):
                return href

    except Exception as exc:
        print(f"Secondary search failed for '{title}': {exc}")

    return ""


def _query_keywords(query: str) -> set[str]:
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in query).split()
    return {w for w in words if w not in _QUERY_STOPWORDS and len(w) > 2}


def _entry_matches_query(title: str, summary: str, keywords: set[str]) -> bool:
    if not keywords:
        return True
    haystack = f"{title} {summary}".lower()
    return any(keyword in haystack for keyword in keywords)


def _published_ts(entry) -> float:
    published_struct = entry.get("published_parsed")
    try:
        return float(calendar.timegm(published_struct)) if published_struct else 0.0
    except (TypeError, OverflowError, ValueError):
        return 0.0


# --------------------------------------------------------------------
# Provider: direct_rss
# --------------------------------------------------------------------

def _search_direct_rss(query: str, max_results: int = 10) -> list[NewsArticle]:
    """
    Fetches curated publisher RSS feeds directly and keyword-filters
    their entries against the query. Every URL returned here is a
    direct publisher link straight from that publisher's own feed --
    "verified_direct" by construction, not by resolution.
    """

    keywords = _query_keywords(query)
    matches: list[tuple[float, NewsArticle]] = []

    for source_name, feed_url in _DIRECT_RSS_FEEDS.items():
        try:
            response = httpx.get(
                feed_url,
                timeout=15,
                follow_redirects=True,
                headers=REQUEST_HEADERS,
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception as exc:
            print(f"direct_rss: feed fetch failed for {source_name} ({feed_url}): {exc}")
            continue

        for entry in feed.entries:
            link = (entry.get("link") or "").strip()
            title = (entry.get("title") or "").strip()

            if not link or not title:
                continue

            raw_summary = entry.get("summary") or entry.get("description") or ""
            summary = clean_html(raw_summary)

            if not _entry_matches_query(title, summary, keywords):
                continue

            if not validate_candidate_url(link):
                continue

            if not summary or summary == title:
                summary = f"Article titled '{title}' published by {source_name}."

            article: NewsArticle = {
                "title": title,
                "source": source_name,
                "source_url": link,
                "published": entry.get("published", ""),
                "published_ts": _published_ts(entry),
                "summary": summary,
                "provider": "direct_rss",
                "rss_url": link,
                "url": link,
                "resolved_url": link,
                "url_resolution_status": "verified_direct",
                "url_resolution_method": "direct_rss_feed",
            }

            matches.append((article["published_ts"], article))

    matches.sort(key=lambda pair: pair[0], reverse=True)
    return [article for _, article in matches[:max_results]]


# --------------------------------------------------------------------
# Provider: google_news
# --------------------------------------------------------------------

def resolve_article_url(
    google_news_url: str,
    title: str,
    source: str,
) -> dict:
    """
    Runs each Google News resolution strategy in order and validates
    every candidate before accepting it. Never fabricates a URL: if
    nothing validates, the caller keeps the original Google News URL
    as a reference only.
    """

    if not is_google_news_wrapper(google_news_url):
        # Some RSS providers already give a direct link.
        if validate_candidate_url(google_news_url):
            return {
                "resolved_url": google_news_url,
                "status": "verified_direct",
                "method": "direct_rss_link",
            }

    candidate = _resolve_via_http_redirect(google_news_url)
    if candidate:
        print(f"Resolved via redirect: {title!r} -> {candidate}")
        return {
            "resolved_url": candidate,
            "status": "verified_direct",
            "method": "redirect",
        }

    candidate = search_original_article_url(title=title, source=source)
    if candidate:
        print(f"Resolved via secondary search: {title!r} -> {candidate}")
        return {
            "resolved_url": candidate,
            "status": "verified_direct",
            "method": "secondary_search",
        }

    print(f"URL resolution failed for {title!r}; keeping Google News reference URL.")
    return {
        "resolved_url": "",
        "status": "unresolved",
        "method": "none",
    }


def _search_google_news(query: str, max_results: int = 10) -> list[NewsArticle]:
    """Search recent news articles using Google News RSS."""

    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={quote(query)}"
        "&hl=en-US"
        "&gl=US"
        "&ceid=US:en"
    )

    try:
        response = httpx.get(
            rss_url,
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 NewsletterAgent/1.0"},
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception as exc:
        print(f"google_news: search failed for query '{query}': {exc}")
        return []

    articles: list[NewsArticle] = []
    seen_urls: set[str] = set()

    for entry in feed.entries:
        google_news_url = (entry.get("link") or "").strip()
        title = (entry.get("title") or "").strip()

        if not google_news_url or not title:
            continue

        source_data = entry.get("source") or {}
        source_name = (source_data.get("title") or "").strip() if source_data else ""
        source_href = (source_data.get("href") or "").strip() if source_data else ""

        raw_summary = entry.get("summary") or entry.get("description") or ""
        summary = clean_html(raw_summary)

        if not summary or summary == title or summary == f"{title} {source_name}":
            summary = (
                f"Article titled '{title}' published by "
                f"{source_name or 'an external news source'}."
            )

        resolution = resolve_article_url(
            google_news_url=google_news_url,
            title=title,
            source=source_name,
        )

        resolved_url = resolution["resolved_url"]
        best_url = resolved_url if resolution["status"] == "verified_direct" else google_news_url

        if best_url in seen_urls:
            continue
        seen_urls.add(best_url)

        article: NewsArticle = {
            "title": title,
            "source": source_name,
            "source_url": source_href,  # candidate hint only, not a verified article URL
            "published": entry.get("published", ""),
            "published_ts": _published_ts(entry),
            "summary": summary,
            "provider": "google_news",
            "rss_url": google_news_url,
            "url": best_url,
            "resolved_url": resolved_url,
            "url_resolution_status": resolution["status"],
            "url_resolution_method": resolution["method"],
        }

        articles.append(article)

        if len(articles) >= max_results:
            break

    return articles


# --------------------------------------------------------------------
# Provider: bing_news
# --------------------------------------------------------------------

def _resolve_bing_url(link: str) -> dict:
    """
    Bing News RSS links are sometimes already direct, and sometimes a
    "bing.com/news/apiclick.aspx?...&url=<plain-text-destination>"
    click-tracking wrapper. That destination is a literal, readable
    query parameter Bing itself puts there -- reading it is not
    decoding an opaque token, just parsing a URL query string. If it's
    absent, we fall back to the same HTTP-redirect strategy used for
    Google.
    """

    if validate_candidate_url(link) and "bing.com" not in urlparse(link).netloc.lower():
        return {"resolved_url": link, "status": "verified_direct", "method": "direct_rss_link"}

    parsed = urlparse(link)
    if "bing.com" in parsed.netloc.lower():
        query_params = parse_qs(parsed.query)
        for key in ("url", "u"):
            values = query_params.get(key)
            if values and validate_candidate_url(values[0]):
                return {
                    "resolved_url": values[0],
                    "status": "verified_direct",
                    "method": "bing_query_param",
                }

        candidate = _resolve_via_http_redirect(link)
        if candidate:
            return {"resolved_url": candidate, "status": "verified_direct", "method": "redirect"}

    return {"resolved_url": "", "status": "unresolved", "method": "none"}


def _search_bing_news(query: str, max_results: int = 10) -> list[NewsArticle]:
    """Search recent news articles using Bing News RSS (no API key required)."""

    rss_url = f"https://www.bing.com/news/search?q={quote(query)}&format=RSS"

    try:
        response = httpx.get(
            rss_url,
            timeout=20,
            follow_redirects=True,
            headers=REQUEST_HEADERS,
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception as exc:
        print(f"bing_news: search failed for query '{query}': {exc}")
        return []

    articles: list[NewsArticle] = []
    seen_urls: set[str] = set()

    for entry in feed.entries:
        raw_link = (entry.get("link") or "").strip()
        title = (entry.get("title") or "").strip()

        if not raw_link or not title:
            continue

        source_data = entry.get("source") or {}
        source_name = (
            (source_data.get("title") or "").strip()
            if isinstance(source_data, dict)
            else str(entry.get("publisher") or "")
        )

        raw_summary = entry.get("summary") or entry.get("description") or ""
        summary = clean_html(raw_summary)
        if not summary or summary == title:
            summary = f"Article titled '{title}' published by {source_name or 'an external news source'}."

        resolution = _resolve_bing_url(raw_link)

        resolved_url = resolution["resolved_url"]
        best_url = resolved_url if resolution["status"] == "verified_direct" else raw_link

        if best_url in seen_urls:
            continue
        seen_urls.add(best_url)

        article: NewsArticle = {
            "title": title,
            "source": source_name,
            "source_url": raw_link,
            "published": entry.get("published", ""),
            "published_ts": _published_ts(entry),
            "summary": summary,
            "provider": "bing_news",
            "rss_url": raw_link,
            "url": best_url,
            "resolved_url": resolved_url,
            "url_resolution_status": resolution["status"],
            "url_resolution_method": resolution["method"],
        }

        articles.append(article)

        if len(articles) >= max_results:
            break

    return articles


# --------------------------------------------------------------------
# Multi-provider orchestration
# --------------------------------------------------------------------

_DEFAULT_PROVIDERS = ["direct_rss", "google_news", "bing_news"]

_PROVIDER_FUNCS: dict[str, Callable[[str, int], list[NewsArticle]]] = {
    "direct_rss": _search_direct_rss,
    "google_news": _search_google_news,
    "bing_news": _search_bing_news,
}


def get_configured_providers() -> list[str]:
    """
    Reads NEWS_SEARCH_PROVIDERS (comma-separated, e.g.
    "direct_rss,bing_news") to let a deployer choose provider order or
    disable a blocked provider entirely, without touching code. Falls
    back to the default order if unset or if nothing valid is listed.
    """

    raw = os.environ.get("NEWS_SEARCH_PROVIDERS", "")
    if not raw.strip():
        return list(_DEFAULT_PROVIDERS)

    configured = [p.strip().lower() for p in raw.split(",") if p.strip()]
    valid = [p for p in configured if p in _PROVIDER_FUNCS]

    if not valid:
        print(
            f"NEWS_SEARCH_PROVIDERS={raw!r} contained no recognized provider "
            f"names; falling back to default order {_DEFAULT_PROVIDERS}."
        )
        return list(_DEFAULT_PROVIDERS)

    return valid


def news_search_tool(
    query: str,
    max_results: int = 10,
    providers: list[str] | None = None,
) -> list[NewsArticle]:
    """
    Runs each configured provider in order, merging and de-duplicating
    results by URL, until enough verified-direct articles are
    collected (or every provider has been tried). Every returned
    article carries a "provider" field so callers/diagnostics can see
    exactly where it came from and how its URL was resolved.

    Returns an empty list -- never fabricated data -- if every
    provider fails or is blocked.
    """

    active_providers = providers if providers is not None else get_configured_providers()

    collected: list[NewsArticle] = []
    seen_urls: set[str] = set()
    summary_lines: list[str] = []

    for provider_name in active_providers:
        provider_fn = _PROVIDER_FUNCS.get(provider_name)
        if provider_fn is None:
            print(f"Unknown provider '{provider_name}', skipping.")
            continue

        try:
            provider_articles = provider_fn(query, max_results)
        except Exception as exc:
            print(f"Provider '{provider_name}' raised an unexpected error: {exc}")
            provider_articles = []

        added = 0
        for article in provider_articles:
            url = article.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            article["provider"] = provider_name
            collected.append(article)
            added += 1

        verified_total = sum(1 for a in collected if a["url_resolution_status"] == "verified_direct")
        summary_lines.append(
            f"  - {provider_name}: {len(provider_articles)} returned, {added} new "
            f"(unique), {verified_total} verified-direct collected so far"
        )

        if verified_total >= max_results:
            break

    print(f"Provider search for query {query!r} (providers tried in order: {active_providers}):")
    for line in summary_lines:
        print(line)

    # Prefer verified-direct articles first; research.py re-sorts by
    # recency afterward, so this ordering only affects which articles
    # survive the max_results cut when there are more candidates than
    # requested.
    collected.sort(key=lambda a: 0 if a.get("url_resolution_status") == "verified_direct" else 1)

    return collected[:max_results]


if __name__ == "__main__":
    results = news_search_tool(query="latest AI agent news", max_results=5)

    resolved_count = sum(1 for a in results if a["url_resolution_status"] == "verified_direct")
    print(f"\nFound {len(results)} articles ({resolved_count} with a verified direct URL):\n")

    for index, article in enumerate(results, start=1):
        print(f"{index}. {article['title']}")
        print(f"   Provider: {article['provider']}")
        print(f"   Source: {article['source']}")
        print(f"   Resolution: {article['url_resolution_status']} ({article['url_resolution_method']})")
        print(f"   URL: {article['url']}")
        print()
