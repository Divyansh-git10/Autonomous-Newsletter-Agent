"""
LLM Summarization tool.

Produces a structured, source-grounded summary for one article. This
is only ever called for articles that have already passed
agent.grounding.needs_fallback() (i.e. real extracted content is
available) -- callers are responsible for that gating. This module
still defends itself: if it's ever handed weak content anyway, it
returns the deterministic fallback instead of calling the LLM.
"""

from __future__ import annotations

from typing import Any

from agent.grounding import (
    build_fallback_summary,
    build_llm_failed_summary,
    extract_json_object,
    needs_fallback,
)
from tools.llm_utils import get_llm, is_rate_limit_error, response_to_text

_STRICT_GROUNDING_RULES = """
SOURCE-GROUNDING RULES (must follow exactly):
- Use only the article text supplied below. Do not use general world
  knowledge to fill in missing details.
- Do not infer facts merely from the article title.
- Every factual claim in your summary must be traceable to a specific
  sentence in the supplied article text.
- If the article text does not support a detail, omit it -- do not
  guess or generalize.
- Do not invent dates, numbers, organizations, findings, or
  announcements that are not explicitly present in the text.
"""


def summarize_article_structured(
    article: dict[str, Any],
    revision_note: str = "",
) -> dict[str, Any]:
    """
    Returns a structured, source-grounded summary:

    {
        "what_happened": str,
        "key_takeaways": list[str],
        "why_it_matters": str,
        "source_limitations": str,
        "content_available": bool,
        "grounding_status": "grounded" | "error_fallback",
    }

    If the article doesn't actually have usable content (defense in
    depth -- callers should already be gating on this), the
    deterministic fallback is returned and the LLM is never called.
    """

    if needs_fallback(article):
        return build_fallback_summary(article)

    title = article.get("title", "Untitled article")
    url = article.get("resolved_url") or article.get("url", "")
    content = article.get("content", "")

    revision_instruction = ""
    if revision_note:
        revision_instruction = f"\nIMPORTANT: {revision_note}\n"

    prompt = f"""
You are an AI newsletter research assistant.
{_STRICT_GROUNDING_RULES}
{revision_instruction}
Summarize the following article. Respond with ONLY the JSON object
below -- no preamble, no explanation, no reasoning, and no text before
or after it. Do not wrap it in Markdown code fences.

{{
  "what_happened": "1-2 sentences describing what the article reports, using only the supplied text",
  "key_takeaways": ["2-3 short bullet points, each grounded in the text"],
  "why_it_matters": "1-2 sentences on relevance to AI agents, grounded in the text",
  "source_limitations": "note any gaps in the supplied text, or 'None noted.' if the text is complete"
}}

Article title:
{title}

Article URL:
{url}

Article text:
{content[:6000]}
"""

    try:
        llm = get_llm(max_tokens=2000)
        response = llm.invoke(prompt)
        raw_text = response_to_text(response)
        parsed = extract_json_object(raw_text)

        what_happened = str(parsed.get("what_happened", "")).strip()
        why_it_matters = str(parsed.get("why_it_matters", "")).strip()
        source_limitations = str(parsed.get("source_limitations", "")).strip() or "None noted."

        key_takeaways_raw = parsed.get("key_takeaways", [])
        if isinstance(key_takeaways_raw, list):
            key_takeaways = [str(item).strip() for item in key_takeaways_raw if str(item).strip()]
        elif isinstance(key_takeaways_raw, str) and key_takeaways_raw.strip():
            key_takeaways = [key_takeaways_raw.strip()]
        else:
            key_takeaways = []

        if not what_happened or not key_takeaways:
            raise ValueError("LLM summary is missing required fields.")

        return {
            "what_happened": what_happened,
            "key_takeaways": key_takeaways,
            "why_it_matters": why_it_matters or "Relevance not specified in the source text.",
            "source_limitations": source_limitations,
            "content_available": True,
            "grounding_status": "grounded",
        }

    except Exception as exc:
        # Graceful, non-hallucinating fallback -- never invent a summary
        # just because the LLM call itself failed. A rate-limit/daily-
        # token-limit failure is logged distinctly so it's never
        # confused with a real content/parsing problem, but it still
        # falls back to the same deterministic template -- it must
        # never be counted as a successful LLM-grounded summary.
        if is_rate_limit_error(exc):
            print(f"Groq rate limit/token limit reached while summarizing '{title}': {exc}")
        else:
            print(f"Structured summarization failed for '{title}': {exc}")
        # We only reach this point when needs_fallback(article) was
        # False, i.e. real article content WAS extracted -- only the
        # LLM call itself failed. Use the wording that says so, rather
        # than the generic "no content available" fallback.
        return build_llm_failed_summary(article)


if __name__ == "__main__":
    test_article = {
        "title": "Example AI Agent Article",
        "url": "https://example.com",
        "resolved_url": "https://example.com",
        "content": (
            "AI agents can plan and execute multi-step tasks using tools. "
            "This example article describes a new open-source framework "
            "that lets developers chain tool calls together with a "
            "planner and a critic step." * 3
        ),
        "content_available": True,
        "extraction_status": "success",
        "url_resolution_status": "verified_direct",
        "is_title_only": False,
    }

    print(summarize_article_structured(test_article))
