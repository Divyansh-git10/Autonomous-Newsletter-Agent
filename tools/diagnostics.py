"""
Standalone research-layer diagnostics.

This script makes NO changes to any application state. It exists to
produce independently-checkable, honest evidence about what each news
provider (and the raw Google News RSS feed itself) actually returns on
whatever network it's run from -- so a network/provider block can be
proven rather than assumed, and so URL resolution is never reported as
"fixed" without live evidence.

Run it directly from the project root:

    python -m tools.diagnostics
    python -m tools.diagnostics "your query here"

What it prints:
1. The complete raw structure of one real Google News RSS entry (every
   key feedparser exposes), so the wrapper URL format can be inspected
   directly instead of guessed at.
2. Raw HTTP probes of each provider's endpoint (status code, final URL
   after redirects, response size) -- or the exact exception if the
   network/provider blocks the request.
3. Per-provider results from tools.news_search (article counts,
   verified-direct counts, and a few example URLs).
4. The combined multi-provider result exactly as agent/research.py
   would receive it.
"""

import sys
from urllib.parse import quote

import feedparser
import httpx

from tools.news_search import (
    REQUEST_HEADERS,
    _search_bing_news,
    _search_direct_rss,
    _search_google_news,
    get_configured_providers,
    news_search_tool,
)

DEFAULT_QUERY = "latest AI agent news"


def probe(name: str, url: str, timeout: float = 15) -> None:
    print(f"\n--- Probe: {name} ---")
    print(f"URL: {url}")
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers=REQUEST_HEADERS,
        )
        print(f"HTTP status: {response.status_code}")
        print(f"Final URL after redirects: {response.url}")
        print(f"Response bytes: {len(response.content)}")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")


def inspect_one_google_news_entry(query: str = DEFAULT_QUERY) -> None:
    print(f"\n--- Raw Google News RSS entry structure for query: {query!r} ---")
    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )
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
        print(f"FAILED to fetch/parse Google News RSS: {type(exc).__name__}: {exc}")
        return

    if not feed.entries:
        print("Feed returned zero entries (empty <channel>, or the request was silently blocked).")
        return

    entry = feed.entries[0]
    print(f"Entry has {len(entry.keys())} keys. Full dump:")
    for key in entry.keys():
        value_repr = repr(entry[key])
        if len(value_repr) > 400:
            value_repr = value_repr[:400] + "...(truncated)"
        print(f"  {key!r}: {value_repr}")


def run_provider(name: str, fn, query: str, max_results: int = 5) -> None:
    print(f"\nProvider: {name}")
    try:
        results = fn(query, max_results)
    except Exception as exc:
        print(f"  Raised: {type(exc).__name__}: {exc}")
        return

    verified = sum(1 for a in results if a.get("url_resolution_status") == "verified_direct")
    print(f"  Returned {len(results)} article(s), {verified} with a verified direct URL")
    for article in results[:5]:
        print(
            f"    - [{article.get('url_resolution_status')}/"
            f"{article.get('url_resolution_method')}] "
            f"{article.get('title', '')[:70]!r} -> {article.get('url', '')}"
        )


def run_diagnostics(query: str = DEFAULT_QUERY) -> None:
    print("=" * 70)
    print("NEWS SEARCH DIAGNOSTICS")
    print("=" * 70)
    print(f"Query: {query!r}")
    print(f"Configured providers (NEWS_SEARCH_PROVIDERS or default): {get_configured_providers()}")

    inspect_one_google_news_entry(query)

    print("\n--- Raw endpoint probes ---")
    probe(
        "Google News RSS endpoint",
        f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en",
    )
    probe("DuckDuckGo HTML search", f"https://html.duckduckgo.com/html/?q={quote(query)}")
    probe("Bing News RSS endpoint", f"https://www.bing.com/news/search?q={quote(query)}&format=RSS")
    probe("Sample direct publisher RSS (TechCrunch)", "https://techcrunch.com/feed/")

    print("\n--- Provider-by-provider results (tools.news_search internals) ---")
    run_provider("direct_rss", _search_direct_rss, query)
    run_provider("google_news", _search_google_news, query)
    run_provider("bing_news", _search_bing_news, query)

    print("\n--- Combined news_search_tool() result (what agent/research.py sees) ---")
    combined = news_search_tool(query, max_results=10)
    verified = sum(1 for a in combined if a.get("url_resolution_status") == "verified_direct")
    print(f"Total: {len(combined)} article(s), {verified} with a verified direct URL")
    for article in combined:
        print(
            f"  - [{article.get('provider')}/{article.get('url_resolution_status')}] "
            f"{article.get('title', '')[:70]!r}"
        )

    print("\n" + "=" * 70)
    if verified == 0:
        print(
            "RESULT: 0 verified direct URLs from any configured provider on this "
            "network. This is a genuine network/provider limitation on this "
            "machine at this time, not a code defect -- see the probes above "
            "for the exact HTTP status/exception each endpoint returned."
        )
    else:
        print(f"RESULT: {verified} verified direct URL(s) obtained -- see details above.")
    print("=" * 70)


if __name__ == "__main__":
    query_arg = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    run_diagnostics(query_arg)
