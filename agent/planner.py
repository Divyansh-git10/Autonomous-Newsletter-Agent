import json
from datetime import datetime, timezone

from tools.llm_utils import get_llm, is_rate_limit_error
from tools.llm_utils import response_to_text

_FALLBACK_QUERIES = [
    "latest AI agent news",
    "enterprise AI agent launches",
    "AI agent safety security",
]


def planner_node(state: dict) -> dict:
    """
    Converts the user's newsletter goal into focused research queries.
    """

    goal = state.get("goal", "").strip()

    if not goal:
        raise ValueError("Newsletter goal cannot be empty.")

    # The LLM's own training data has a knowledge cutoff well in the
    # past relative to whenever this actually runs, so without an
    # explicit anchor it tends to default to a stale year (e.g. "2024
    # AI agent developments") baked into its training distribution.
    # Passing today's real date and explicitly forbidding a hardcoded
    # past year keeps generated queries genuinely "latest news" queries
    # regardless of how long ago the model was trained.
    current_date = datetime.now(timezone.utc).strftime("%B %d, %Y")

    prompt = f"""
You are the planning component of an autonomous newsletter agent.

Today's date is {current_date}. Treat this as the true current date --
your own training data may be older than this.

User goal:
{goal}

Create a focused research plan for a newsletter about AI agents.

Requirements:
1. Generate 3 to 5 search queries.
2. Queries should cover recent and relevant developments as of the
   current date above -- use words like "latest" or "recent" rather
   than a specific year, unless the user's goal explicitly asks about a
   past period.
3. Do NOT hardcode or assume any specific past year (e.g. 2023, 2024)
   in a query unless the user's goal explicitly asks for historical
   coverage of that year.
4. Include different angles such as product launches,
   enterprise use cases, research, safety, and security
   when relevant.
5. Avoid overly broad queries.
6. Respond with ONLY the JSON object -- no preamble, no explanation,
   and no reasoning before or after it.
7. Do not use Markdown code fences.

Required JSON format:
{{
  "search_queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}}
"""

    # The LLM call itself (not just JSON parsing) must be guarded: a
    # Groq rate-limit/daily-token-limit failure, a network error, or a
    # missing API key must fall back to the safe default plan rather
    # than crash the whole graph -- planning is the very first node, so
    # an unhandled failure here would take down the entire run.
    try:
        llm = get_llm(max_tokens=700)
        response = llm.invoke(prompt)
        text = response_to_text(response)
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        result = json.loads(text)
        queries = result.get("search_queries", [])

        if not isinstance(queries, list):
            raise ValueError("search_queries must be a list.")

        queries = [
            str(query).strip()
            for query in queries
            if str(query).strip()
        ]

        if not queries:
            raise ValueError("LLM returned an empty search_queries list.")

    except Exception as exc:
        if is_rate_limit_error(exc):
            print(f"Groq rate limit/token limit reached while planning: {exc}")
        else:
            print(f"Planner LLM call or JSON parsing failed: {exc}")

        # Safe, deterministic fallback plan -- never crash the pipeline
        # just because the LLM call failed.
        queries = list(_FALLBACK_QUERIES)

    return {
        "search_queries": queries[:5],
        "search_attempt": state.get("search_attempt", 0),
    }


if __name__ == "__main__":
    test_state = {
        "goal": (
            "Create a weekly newsletter on the latest "
            "AI agent news and send it to our subscribers."
        ),
        "search_attempt": 0,
    }

    result = planner_node(test_state)

    print("\nPlanner output:\n")
    print(result)