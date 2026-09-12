"""
Relevance ranking + publisher-diversity selection node.

Runs between research and content_enrichment. agent/research.py's job
is purely to *collect* a candidate pool from every configured provider
(see tools/news_search.py); this node's job is to *select* the final
5-7 articles that are genuinely about AI agents (or clearly
agent-related, per the moderate top-up tier) and reasonably diverse
across publishers, before any network extraction or Groq call is spent
on them.

This is where most of the token-usage reduction in this pipeline
actually comes from: extraction (agent/content.py) and summarization
(agent/summarize.py) only ever see the articles this node selects, not
the full candidate pool research_node collected.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from agent.relevance import (
    MIN_STRICT_BEFORE_MODERATE_TOPUP,
    RELEVANCE_MODERATE_THRESHOLD,
    RELEVANCE_STRICT_THRESHOLD,
    TARGET_ARTICLE_COUNT,
    rank_and_select_articles,
)


def ranking_node(state: dict[str, Any]) -> dict[str, Any]:
    raw_articles = state.get("raw_articles", [])
    research_stats = dict(state.get("research_stats") or {})

    result = rank_and_select_articles(
        raw_articles,
        target_count=TARGET_ARTICLE_COUNT,
        min_strict_before_topup=MIN_STRICT_BEFORE_MODERATE_TOPUP,
        strict_threshold=RELEVANCE_STRICT_THRESHOLD,
        moderate_threshold=RELEVANCE_MODERATE_THRESHOLD,
    )

    selected = result["selected"]
    strict = result["strict"]
    moderate = result["moderate"]
    rejected = result["rejected"]
    tier_used = result["tier_used"]

    strict_by_provider = Counter(a.get("provider", "unknown") for a in strict)
    moderate_by_provider = Counter(a.get("provider", "unknown") for a in moderate)
    selected_by_provider = Counter(a.get("provider", "unknown") for a in selected)

    provider_stats: dict[str, dict[str, int]] = research_stats.get("providers", {})
    all_provider_names = (
        set(provider_stats)
        | set(strict_by_provider)
        | set(moderate_by_provider)
        | set(selected_by_provider)
    )
    for provider_name in all_provider_names:
        counts = provider_stats.setdefault(
            provider_name, {"results": 0, "verified": 0, "unresolved": 0}
        )
        counts["strict_relevant"] = strict_by_provider.get(provider_name, 0)
        counts["moderate_relevant"] = moderate_by_provider.get(provider_name, 0)
        counts["selected"] = selected_by_provider.get(provider_name, 0)

    selected_publishers = {
        (a.get("source") or a.get("provider") or "unknown").strip().lower()
        for a in selected
    }

    research_stats["providers"] = provider_stats
    research_stats.setdefault("total_collected", len(raw_articles))
    research_stats["total_strict_relevant"] = len(strict)
    research_stats["total_moderate_relevant"] = len(moderate)
    research_stats["total_rejected_irrelevant"] = len(rejected)
    research_stats["total_selected"] = len(selected)
    research_stats["selection_tier"] = tier_used
    research_stats["selected_publisher_count"] = len(selected_publishers)
    # Kept concise (title + source only, no full article content) so
    # print_verification_summary() can show a human reviewer exactly
    # which articles were rejected as off-topic, e.g. to confirm a
    # specific known-irrelevant article was correctly excluded.
    research_stats["rejected_titles"] = [
        {"title": a.get("title", "Untitled"), "source": a.get("source") or a.get("provider", "unknown")}
        for a in rejected
    ]
    research_stats["selected_titles"] = [
        {"title": a.get("title", "Untitled"), "source": a.get("source") or a.get("provider", "unknown")}
        for a in selected
    ]

    print(
        f"Relevance ranking: {len(raw_articles)} candidate(s) -> "
        f"{len(strict)} strict match(es), {len(moderate)} moderate match(es), "
        f"{len(rejected)} rejected as off-topic -> "
        f"{len(selected)} selected (tier used: {tier_used}, "
        f"{len(selected_publishers)} distinct publisher(s))."
    )
    for index, article in enumerate(selected, start=1):
        print(
            f"  {index}. [{article.get('relevance_tier')}, score {article.get('relevance_score')}] "
            f"{article.get('title', '')[:70]!r} "
            f"({article.get('source') or article.get('provider', 'unknown')})"
        )

    return {
        "raw_articles": selected,
        "research_stats": research_stats,
    }


if __name__ == "__main__":
    sample = [
        {"title": "New AI Agent Framework Ships With Multi-Step Planning", "source": "TechCrunch", "provider": "direct_rss", "summary": "", "url_resolution_status": "verified_direct"},
        {"title": "GTA 6 Delayed Again, Fans React", "source": "IGN", "provider": "bing_news", "summary": "", "url_resolution_status": "verified_direct"},
        {"title": "New Satellite Launched To Study Climate", "source": "SpaceNews", "provider": "google_news", "summary": "", "url_resolution_status": "unresolved"},
        {"title": "AWS Launches Autonomous Coding Assistant for Enterprise Developers", "source": "AWS Machine Learning Blog", "provider": "direct_rss", "summary": "The new assistant plans and executes multi-step coding tasks.", "url_resolution_status": "verified_direct"},
    ]
    result = ranking_node({"raw_articles": sample, "research_stats": {}})
    print("\nSelected:")
    for a in result["raw_articles"]:
        print(f"- {a['title']} ({a['relevance_score']}, {a['relevance_tier']})")
