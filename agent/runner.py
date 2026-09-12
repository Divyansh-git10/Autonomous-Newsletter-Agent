from __future__ import annotations

from typing import Any

from agent.exporter import simulate_send
from agent.graph import run_graph


def print_verification_summary(result: dict[str, Any]) -> None:
    """
    Prints the exact counts a human reviewer needs to verify the run:
    per-provider search diagnostics, URL resolution, extraction,
    summarization grounding, and the critic's final verdict.
    """

    raw_articles = result.get("raw_articles", [])
    ranked_articles = result.get("ranked_articles", [])
    critique = result.get("critique") or {}
    research_stats = result.get("research_stats") or {}

    verified_urls = sum(1 for a in raw_articles if a.get("url_resolution_status") == "verified_direct")
    unresolved_urls = len(raw_articles) - verified_urls

    extracted_success = sum(1 for a in raw_articles if a.get("extraction_status") == "success")
    title_only = sum(1 for a in raw_articles if a.get("is_title_only"))

    llm_grounded = sum(1 for a in ranked_articles if a.get("grounding_status") == "grounded")
    deterministic_fallback = sum(1 for a in ranked_articles if a.get("grounding_status") == "fallback")
    error_fallback = sum(1 for a in ranked_articles if a.get("grounding_status") == "error_fallback")
    downgraded = sum(1 for a in ranked_articles if a.get("grounding_status") == "downgraded_after_revision")

    grounding_issues = critique.get("grounding_issues", [])

    provider_stats: dict = research_stats.get("providers", {})

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print("Research providers (results / verified / unresolved / strict / moderate / selected):")
    if provider_stats:
        for provider_name, counts in provider_stats.items():
            print(
                f"  - {provider_name}: {counts.get('results', 0)} / "
                f"{counts.get('verified', 0)} / {counts.get('unresolved', 0)} / "
                f"{counts.get('strict_relevant', 0)} / {counts.get('moderate_relevant', 0)} / "
                f"{counts.get('selected', 0)}"
            )
    else:
        print("  - (no provider diagnostics available -- every provider returned nothing)")
    print(
        f"Total articles found by research: "
        f"{research_stats.get('total_collected', len(raw_articles))}"
    )
    print(
        f"  - Total verified (resolved to a direct publisher URL): "
        f"{research_stats.get('total_verified', 'N/A')}"
    )
    print(
        f"  - Strict relevance match (unambiguous AI-agent stories): "
        f"{research_stats.get('total_strict_relevant', 'N/A')}"
    )
    print(
        f"  - Moderate relevance match (agent-related, used only to top up "
        f"below {5}): {research_stats.get('total_moderate_relevant', 'N/A')}"
    )
    print(
        f"  - Rejected as off-topic:            "
        f"{research_stats.get('total_rejected_irrelevant', 'N/A')}"
    )
    print(
        f"  - Selected for the newsletter (post publisher-diversity): "
        f"{research_stats.get('total_selected', len(raw_articles))} "
        f"(tier used: {research_stats.get('selection_tier', 'N/A')}, "
        f"{research_stats.get('selected_publisher_count', 'N/A')} distinct publisher(s))"
    )
    rejected_titles = research_stats.get("rejected_titles", [])
    if rejected_titles:
        print(f"  - Rejected article titles ({len(rejected_titles)}):")
        for item in rejected_titles:
            print(f"      - {item.get('title', 'Untitled')} ({item.get('source', 'unknown')})")
    print(f"Selected articles: {len(raw_articles)}")
    print(f"  - Verified direct publisher URL: {verified_urls}")
    print(f"  - Unresolved (reference only):   {unresolved_urls}")
    print(f"Article extraction:")
    print(f"  - Extracted successfully:        {extracted_success}")
    print(f"  - Title-only (no usable content):{title_only}")
    print(f"Summarization ({len(ranked_articles)} articles considered):")
    print(f"  - LLM-grounded summaries:        {llm_grounded}")
    print(f"  - Deterministic fallback:        {deterministic_fallback}")
    print(f"  - Error fallback (LLM failed):   {error_fallback}")
    print(f"  - Downgraded during revision:    {downgraded}")
    print("Final articles included in the newsletter:")
    if ranked_articles:
        for index, article in enumerate(ranked_articles, start=1):
            print(
                f"  {index}. {article.get('title', 'Untitled')} "
                f"({article.get('source') or article.get('provider', 'unknown')}, "
                f"{article.get('grounding_status', 'fallback')})"
            )
    else:
        print("  (none)")
    print(f"Unsupported claims detected by critic: {'YES - ' + str(len(grounding_issues)) + ' issue(s)' if grounding_issues else 'None'}")
    print(f"Revision rounds used:            {result.get('revision_count', 0)}")
    print(f"Final quality score:             {critique.get('quality_score', 'N/A')}")
    if result.get("output_path"):
        print(f"Markdown output:                 {result['output_path']}")
    if result.get("html_path"):
        print(f"HTML output:                     {result['html_path']}")
    print("=" * 60 + "\n")


def run_newsletter_agent(
    goal: str,
    mode: str = "autonomous",
) -> dict[str, Any]:
    """
    Run the complete newsletter agent.

    Args:
        goal: User's plain-English newsletter objective.
        mode: Either 'autonomous' or 'human'.

    Returns:
        Final LangGraph state as a dictionary.
    """

    normalized_mode = mode.strip().lower()

    if normalized_mode not in {"autonomous", "human"}:
        raise ValueError(
            "Invalid mode. Use either 'autonomous' or 'human'."
        )

    if not goal or not goal.strip():
        raise ValueError("Newsletter goal cannot be empty.")

    initial_state: dict[str, Any] = {
        "goal": goal.strip(),
        "mode": normalized_mode,
        "search_queries": [],
        "raw_articles": [],
        "ranked_articles": [],
        "research_stats": {},
        "draft_newsletter": "",
        "critique": {},
        "revision_count": 0,
        "approved": False,
        "final_output": "",
        "output_path": "",
    }

    result = dict(run_graph(initial_state))
    result["mode"] = normalized_mode

    # The graph never explicitly sets "final_output" (only
    # "draft_newsletter" is written by writer/reviser nodes), so make the
    # returned dict self-consistent for any caller that doesn't know to
    # fall back to "draft_newsletter" manually.
    result["final_output"] = result.get("draft_newsletter", "")

    print_verification_summary(result)

    if normalized_mode == "autonomous":
        newsletter = result["final_output"]

        if newsletter.strip():
            # Autonomous mode must "do the full job" from this one
            # function call, including the simulated send/export --
            # it should not depend on the Streamlit UI to save files.
            export_info = simulate_send(newsletter)
            result["output_path"] = export_info["markdown_path"]
            result["html_path"] = export_info["html_path"]
            result["html_content"] = export_info["html_content"]

    return result


if __name__ == "__main__":
    result = run_newsletter_agent(
        goal=(
            "Create a weekly newsletter covering the latest AI-agent "
            "news for developers."
        ),
        mode="autonomous",
    )

    print("\nAgent execution completed.")
    print(f"Mode: {result.get('mode')}")
    print(f"Articles researched: {len(result.get('raw_articles', []))}")
    print(
        f"Articles summarized: "
        f"{len(result.get('ranked_articles', []))}"
    )

    critique = result.get("critique") or {}
    print(f"Quality score: {critique.get('quality_score', 'N/A')}")

    print("\nNewsletter:\n")
    print(
        result.get("final_output")
        or result.get("draft_newsletter")
        or "No newsletter generated."
    )