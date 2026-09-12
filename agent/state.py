from typing import Literal, TypedDict


class Article(TypedDict, total=False):
    title: str
    source: str
    source_url: str  # RSS-provided publisher homepage hint (NOT a verified article URL)
    published: str
    published_ts: float

    # --- URL resolution (see tools/news_search.py) ---
    provider: str  # which search provider returned this article:
    # "direct_rss" | "google_news" | "bing_news"
    matched_query: str  # the specific research query that found this
    # article (see agent/research.py) -- used for the query-aware
    # relevance bonus in agent/relevance.py
    relevance_score: float  # deterministic score from agent/relevance.py
    relevance_tier: str  # "strict" | "moderate" | "rejected"
    rss_url: str  # original source RSS/wrapper URL, always preserved
    url: str  # best available link: resolved_url if verified, else rss_url
    resolved_url: str  # verified direct publisher URL, "" if none was found
    url_resolution_status: str  # "verified_direct" | "unresolved"
    url_resolution_method: str
    # one of: "direct_rss_feed" (URL correct by construction, no resolution
    # needed), "direct_rss_link" (RSS provider already gave a direct link),
    # "redirect" (HTTP redirect chain resolved it), "bing_query_param"
    # (Bing's own click-tracking URL exposes the real link in plain text),
    # "secondary_search", or "none" (unresolved)

    # --- extraction (see tools/article_extractor.py, agent/content.py) ---
    summary: str  # raw RSS summary/description
    content: str  # extracted full article text, "" if unavailable
    content_available: bool
    extraction_status: str
    # one of: "success", "failed_http_error", "failed_timeout",
    # "failed_empty", "failed_blocked", "skipped_wrapper_url",
    # "skipped_no_url", "not_attempted"
    is_title_only: bool

    # --- summarization (see tools/summarizer.py, agent/summarize.py) ---
    what_happened: str
    key_takeaways: list[str]
    why_it_matters: str
    source_limitations: str
    grounding_status: str
    # one of: "grounded" (LLM summary of real content),
    # "fallback" (deterministic, content unavailable),
    # "error_fallback" (LLM call failed despite content being available),
    # "downgraded_after_revision" (forced back to fallback during review)


class NewsletterState(TypedDict, total=False):
    goal: str
    mode: Literal["autonomous", "human"]

    search_queries: list[str]
    search_attempt: int

    raw_articles: list[Article]
    ranked_articles: list[Article]
    summary_stats: dict
    research_stats: dict  # provider-level diagnostics: results/verified/
    # unresolved/relevant/selected counts per provider, plus totals --
    # see agent/research.py and agent/ranking.py

    draft_newsletter: str
    critique: dict

    revision_count: int
    approved: bool

    final_output: str
    output_path: str
