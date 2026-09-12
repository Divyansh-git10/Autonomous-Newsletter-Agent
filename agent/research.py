from tools.news_search import news_search_tool


def research_node(state: dict) -> dict:
    """
    Executes planner-generated search queries and combines the
    resulting articles.

    Two things matter for a good "top 5-7" selection later on:

    1. Recency - within each query's results, the freshest articles
       are moved to the front (Google News RSS does not strictly sort
       by publish date, so relying on RSS order alone can surface
       stale articles ahead of new ones).
    2. Query diversity - articles are interleaved round-robin across
       queries instead of being appended query-by-query. Without this,
       a simple `raw_articles[:7]` selection later can be dominated
       entirely by the first one or two queries, silently dropping
       whole angles (e.g. "safety and security") that the planner
       deliberately asked for.
    """

    queries = state.get("search_queries", [])

    if not queries:
        raise ValueError(
            "No search queries available for research."
        )

    seen_urls: set[str] = set()
    per_query_articles: list[list[dict]] = []

    for query in queries:
        print(f"Researching query: {query}")

        articles = news_search_tool(
            query=query,
            max_results=5,
        )

        # Tag each article with the specific query that found it so
        # agent/relevance.py can give a query-aware relevance bonus
        # (e.g. an "AI agent safety and security" query rewards an
        # article that discusses safety/security specifically) instead
        # of requiring every article to independently justify itself on
        # generic agent terminology alone.
        for article in articles:
            article["matched_query"] = query

        # Freshest first within this query's results.
        articles = sorted(
            articles,
            key=lambda article: article.get("published_ts", 0.0),
            reverse=True,
        )

        deduped: list[dict] = []

        for article in articles:
            url = article.get("url", "")

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            deduped.append(article)

        per_query_articles.append(deduped)

    # Round-robin interleave: take one article from each query in turn
    # so every query angle is represented in the final ordering, rather
    # than exhausting one query's results before moving to the next.
    all_articles: list[dict] = []
    max_len = max(
        (len(group) for group in per_query_articles),
        default=0,
    )

    for index in range(max_len):
        for group in per_query_articles:
            if index < len(group):
                all_articles.append(group[index])

    verified_count = sum(
        1 for a in all_articles if a.get("url_resolution_status") == "verified_direct"
    )
    unresolved_count = len(all_articles) - verified_count

    print(
        f"Research summary: {len(all_articles)} unique articles collected "
        f"across {len(queries)} queries "
        f"({verified_count} with a verified direct URL, "
        f"{unresolved_count} unresolved / reference only)."
    )

    provider_stats: dict[str, dict[str, int]] = {}
    for article in all_articles:
        provider_name = article.get("provider", "unknown")
        counts = provider_stats.setdefault(
            provider_name, {"results": 0, "verified": 0, "unresolved": 0}
        )
        counts["results"] += 1
        if article.get("url_resolution_status") == "verified_direct":
            counts["verified"] += 1
        else:
            counts["unresolved"] += 1

    research_stats = {
        "total_collected": len(all_articles),
        "total_verified": verified_count,
        "total_unresolved": unresolved_count,
        "providers": provider_stats,
    }

    return {
        "raw_articles": all_articles,
        "research_stats": research_stats,
    }


if __name__ == "__main__":
    test_state = {
        "search_queries": [
            "latest AI agent news",
            "enterprise AI agent launches",
            "AI agent safety security",
        ]
    }

    result = research_node(test_state)

    print(
        f"\nTotal unique articles: "
        f"{len(result['raw_articles'])}"
    )

    for index, article in enumerate(
        result["raw_articles"][:10],
        start=1,
    ):
        print(
            f"{index}. "
            f"{article.get('title', '')}"
        )
