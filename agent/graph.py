from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.content import content_enrichment_node
from agent.critic import critic_node
from agent.planner import planner_node
from agent.ranking import ranking_node
from agent.research import research_node
from agent.reviser import reviser_node
from agent.state import NewsletterState
from agent.summarize import summarize_node
from agent.writer import writer_node


def should_revise(state: NewsletterState) -> str:
    """
    Decide whether the newsletter should be revised.

    Revision is allowed only when:
    - Critic requested a revision
    - Maximum revision count has not been reached
    """

    critique = state.get("critique") or {}
    needs_revision = bool(critique.get("needs_revision", False))
    revision_count = int(state.get("revision_count", 0))

    if needs_revision and revision_count < 2:
        return "revise"

    return "end"


def build_newsletter_graph():
    """Build and compile the newsletter agent graph."""

    workflow = StateGraph(NewsletterState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("research", research_node)
    workflow.add_node("ranking", ranking_node)
    workflow.add_node("content_enrichment", content_enrichment_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reviser", reviser_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "research")
    workflow.add_edge("research", "ranking")
    workflow.add_edge("ranking", "content_enrichment")
    workflow.add_edge("content_enrichment", "summarize")
    workflow.add_edge("summarize", "writer")
    workflow.add_edge("writer", "critic")

    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "reviser",
            "end": END,
        },
    )

    workflow.add_edge("reviser", "critic")

    return workflow.compile()


newsletter_graph = build_newsletter_graph()


def run_graph(initial_state: dict[str, Any]) -> NewsletterState:
    """Execute the compiled newsletter graph."""

    return newsletter_graph.invoke(initial_state)


if __name__ == "__main__":
    print("Running newsletter graph...\n")

    initial_state: NewsletterState = {
        "goal": (
            "Create a weekly newsletter covering the latest AI-agent "
            "developments for developers and technical teams."
        ),
        "mode": "autonomous",
        "search_queries": [],
        "raw_articles": [],
        "ranked_articles": [],
        "draft_newsletter": "",
        "critique": {},
        "revision_count": 0,
        "approved": False,
        "final_output": "",
        "output_path": "",
    }

    result = run_graph(initial_state)

    print("\nGraph execution completed.")
    print(f"Search queries: {result.get('search_queries', [])}")
    print(f"Raw articles: {len(result.get('raw_articles', []))}")
    print(f"Summarized articles: {len(result.get('ranked_articles', []))}")
    print(f"Revision count: {result.get('revision_count', 0)}")

    critique = result.get("critique") or {}
    print(f"Final quality score: {critique.get('quality_score', 'N/A')}")

    print("\nGenerated newsletter preview:\n")
    print(result.get("final_output") or result.get("draft_newsletter", ""))