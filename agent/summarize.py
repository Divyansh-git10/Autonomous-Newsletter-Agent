from typing import Any

from agent.grounding import build_fallback_summary, needs_fallback
from tools.summarizer import summarize_article_structured


def summarize_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Summarize the selected articles into structured, source-grounded
    fields. Content availability is checked BEFORE the LLM is ever
    invoked: articles without usable extracted content go straight to
    the deterministic fallback and never reach the LLM at all.
    """

    raw_articles = state.get("raw_articles", [])
    # agent/ranking.py already narrows raw_articles down to the final
    # relevant, publisher-diverse selection (<= TARGET_ARTICLE_COUNT)
    # before this node runs -- this slice is now just a defensive cap,
    # not the primary selection mechanism, so summarization never runs
    # on more articles (and therefore more Groq tokens) than intended
    # even if a caller invokes this node directly with a larger list.
    selected_articles = raw_articles[:7]

    print(f"Starting summarization for {len(selected_articles)} articles...")

    summarized_articles = []
    stats = {"llm_grounded": 0, "deterministic_fallback": 0, "error_fallback": 0}

    for index, article in enumerate(selected_articles, start=1):
        title = article.get("title", "Untitled")

        if needs_fallback(article):
            structured = build_fallback_summary(article)
            stats["deterministic_fallback"] += 1
            print(f"[{index}/{len(selected_articles)}] Deterministic fallback (no usable content): {title}")
        else:
            structured = summarize_article_structured(article)
            if structured.get("grounding_status") == "grounded":
                stats["llm_grounded"] += 1
                print(f"[{index}/{len(selected_articles)}] LLM-grounded summary: {title}")
            else:
                stats["error_fallback"] += 1
                print(f"[{index}/{len(selected_articles)}] LLM summarization failed, used fallback: {title}")

        summarized_articles.append({**article, **structured})

    print(
        "Summarization summary: "
        f"{stats['llm_grounded']} LLM-grounded, "
        f"{stats['deterministic_fallback']} deterministic fallback (content unavailable), "
        f"{stats['error_fallback']} error fallback (LLM call failed)."
    )

    return {
        "ranked_articles": summarized_articles,
        "raw_articles": raw_articles,
        "summary_stats": stats,
    }


if __name__ == "__main__":
    print("summarize_node import successful")
