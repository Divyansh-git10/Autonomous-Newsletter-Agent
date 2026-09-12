from __future__ import annotations

import json
import re
from typing import Any

from agent.grounding import extract_json_object, validate_newsletter_grounding
from tools.llm_utils import get_llm, is_rate_limit_error, response_to_text


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_newsletter(newsletter: str) -> str:
    return (
        newsletter
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace(" ", " ")
        .lower()
        .strip()
    )


def _count_key_takeaway_sections(newsletter: str) -> int:
    pattern = re.compile(
        r"""
        (?:
            \*\*\s*key\s+takeaways\s*:?\s*\*\*
            |
            ^\s*#{1,6}\s+key\s+takeaways\s*:?\s*$
            |
            ^\s*key\s+takeaways\s*:?\s*$
        )
        """,
        flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )
    return len(pattern.findall(newsletter))


def _count_what_happened_sections(newsletter: str) -> int:
    pattern = re.compile(
        r"""
        (?:
            \*\*\s*what\s+happened\s*:?\s*\*\*
            |
            ^\s*#{1,6}\s+what\s+happened\s*:?\s*$
            |
            ^\s*what\s+happened\s*:?\s*$
        )
        """,
        flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )
    return len(pattern.findall(newsletter))


def _count_article_headings(newsletter: str) -> int:
    return len(re.findall(r"^\s*#{1,6}\s+\d+\.\s+.+$", newsletter, flags=re.MULTILINE))


def _check_article_summaries(newsletter: str, article_count: int) -> list[str]:
    issues: list[str] = []
    if article_count <= 0:
        return issues

    article_heading_count = _count_article_headings(newsletter)
    if article_heading_count < article_count:
        issues.append(
            f"Only {article_heading_count} article sections were found, "
            f"but {article_count} were expected."
        )

    what_happened_count = _count_what_happened_sections(newsletter)
    if what_happened_count < article_count:
        issues.append("One or more article sections are missing a 'What happened' summary.")

    return issues


def deterministic_checks(newsletter: str, article_count: int) -> list[str]:
    """Structural / formatting checks (unchanged from before hardening)."""

    issues: list[str] = []

    if not newsletter or not newsletter.strip():
        return ["Newsletter is empty."]

    if len(newsletter.strip()) < 500:
        issues.append("Newsletter is too short and may not contain enough useful detail.")

    normalized = _normalize_newsletter(newsletter)

    if "editor's takeaway" not in normalized:
        issues.append("Missing Editor's Takeaway section.")

    key_takeaway_sections = _count_key_takeaway_sections(newsletter)
    expected_key_takeaways = min(article_count, 5)
    if key_takeaway_sections < expected_key_takeaways:
        issues.append(
            f"Expected at least {expected_key_takeaways} "
            f"'Key takeaways' sections, found {key_takeaway_sections}."
        )

    article_heading_count = _count_article_headings(newsletter)
    expected_article_headings = min(article_count, 5)
    if article_heading_count < expected_article_headings:
        issues.append(
            f"Expected at least {expected_article_headings} numbered article "
            f"headings, found {article_heading_count}."
        )

    if not re.search(r"https?://[^\s)\]]+", newsletter, flags=re.IGNORECASE):
        issues.append("Newsletter does not contain any source URLs.")

    generic_fallback_markers = [
        "this article discusses ai",
        "this article highlights the importance of ai",
        "ai is becoming increasingly important",
        "ai agents improve efficiency",
    ]
    generic_marker_count = sum(marker in normalized for marker in generic_fallback_markers)
    if generic_marker_count >= 2:
        issues.append("Newsletter contains overly generic fallback language.")

    issues.extend(_check_article_summaries(newsletter=newsletter, article_count=article_count))

    return issues


def _parse_critic_response(raw_text: str) -> dict[str, Any]:
    parsed = extract_json_object(raw_text)

    needs_revision = bool(parsed.get("needs_revision", False))
    issues = _string_list(parsed.get("issues"))
    suggestions = _string_list(parsed.get("suggestions"))

    try:
        quality_score = int(parsed.get("quality_score", 0))
    except (TypeError, ValueError):
        quality_score = 0

    if 0 <= quality_score <= 10:
        quality_score *= 10

    quality_score = max(0, min(100, quality_score))

    return {
        "needs_revision": needs_revision,
        "issues": issues,
        "suggestions": suggestions,
        "quality_score": quality_score,
    }


def _fallback_critic(deterministic_issues: list[str]) -> dict[str, Any]:
    if deterministic_issues:
        return {
            "needs_revision": True,
            "issues": deterministic_issues,
            "suggestions": [
                "Fix the deterministic structural/grounding issues before finalizing the newsletter."
            ],
            "quality_score": 45,
        }

    return {
        "needs_revision": False,
        "issues": [],
        "suggestions": [],
        "quality_score": 75,
    }


def critique_newsletter(
    goal: str,
    newsletter: str,
    articles: list[dict[str, Any]],
    revision_count: int = 0,
) -> dict[str, Any]:
    """
    Critiques a newsletter using three layers:
    1. Structural deterministic checks (headings, key takeaways, links).
    2. Grounding deterministic checks (unsupported claims, mislabeled
       sources, Editor's Takeaway generalization) -- see agent/grounding.py.
    3. An LLM editorial review.

    Factual grounding is a HARD requirement: grounding issues are never
    waived just because the newsletter is well formatted, and they
    always force needs_revision=True regardless of what the LLM says.
    """

    article_count = len(articles)

    print("\nRunning deterministic structural checks...")
    structural_issues = deterministic_checks(newsletter=newsletter, article_count=article_count)

    print("Running deterministic grounding checks...")
    grounding_issues = validate_newsletter_grounding(newsletter, articles)

    deterministic_issues = structural_issues + grounding_issues

    if deterministic_issues:
        print("Deterministic checks found issues:")
        for issue in deterministic_issues:
            print(f"- {issue}")
    else:
        print("Deterministic checks passed (structural and grounding).")

    # `articles` is already the small, final selection (agent/ranking.py
    # narrows the candidate pool to <= 7 relevant articles before any of
    # this runs) -- this is never the full research corpus, which keeps
    # this prompt's input size roughly constant regardless of how many
    # articles were originally researched.
    article_context_parts: list[str] = []
    for index, article in enumerate(articles, start=1):
        article_context_parts.append(
            f"""
ARTICLE {index}
Title: {article.get("title", "Untitled article")}
Content available: {article.get("content_available", False)}
Grounding status: {article.get("grounding_status", "fallback")}
URL resolution: {article.get("url_resolution_status", "unresolved")}
What happened: {article.get("what_happened", "")}
Key takeaways: {article.get("key_takeaways", [])}
Why it matters: {article.get("why_it_matters", "")}
Source limitations: {article.get("source_limitations", "")}
""".strip()
        )
    article_context = "\n\n".join(article_context_parts)

    prompt = f"""
You are a strict but practical editor reviewing a newsletter about AI agents.
Factual grounding is a HARD requirement -- a well-formatted newsletter
that contains unsupported claims must NOT receive a high quality score.

Editorial goal:
{goal}

Review the newsletter against the supplied article data and answer these
questions explicitly in your reasoning before scoring:

1. Are all factual claims supported by extracted article content?
2. Does any title-only / content-unavailable article contain unsupported details?
3. Do unavailable articles use the required deterministic fallback wording?
4. Does the Editor's Takeaway generalize a trend from unsupported/title-only sources?
5. Are source URLs accurately labeled (Google News reference vs. verified publisher article)?
6. Are source limitations clearly disclosed?
7. Are the newsletter's claims consistent with the supplied article data?
8. Are there fabricated takeaways, implications, dates, numbers, or announcements?

IMPORTANT:
- Do not demand direct quotes unless quotes are supplied.
- Do not demand formal fact-checking logs unless verification data is supplied.
- A source limitation is acceptable when the source itself lacks details.
- Avoid vague criticism -- mention the specific article or section when possible.
- If the newsletter is structurally complete AND factually grounded, set
  needs_revision to false.
- A small optional improvement should be listed under suggestions, not issues.
- Respond with ONLY the JSON object -- no preamble, no explanation, no reasoning, and no text before or after it. Do not wrap it in Markdown code fences.

Required JSON schema:
{{
  "needs_revision": false,
  "issues": [],
  "suggestions": [],
  "quality_score": 0
}}

Newsletter:
{newsletter}

Supplied article data:
{article_context}
"""

    try:
        llm = get_llm(max_tokens=1200)
        response = llm.invoke(prompt)
        raw_text = response_to_text(response)
        llm_result = _parse_critic_response(raw_text)

        print("\nLLM critic response:")
        print(json.dumps(llm_result, indent=2, ensure_ascii=False))

    except Exception as exc:
        if is_rate_limit_error(exc):
            print(f"\nGroq rate limit/token limit reached during critique: {exc}")
        else:
            print(f"\nLLM critic failed: {exc}")
        # A rate-limit/token-limit failure must never be treated as a
        # successful LLM review -- it falls back to the same
        # deterministic-only critic result as any other LLM failure.
        llm_result = _fallback_critic(deterministic_issues)

    combined_issues: list[str] = list(deterministic_issues)
    for issue in llm_result.get("issues", []):
        if issue not in combined_issues:
            combined_issues.append(issue)

    combined_suggestions = _string_list(llm_result.get("suggestions"))

    # Grounding/structural issues are a hard requirement: they force
    # revision regardless of the LLM's own opinion.
    needs_revision = bool(deterministic_issues or llm_result.get("needs_revision", False))

    quality_score = int(llm_result.get("quality_score", 0))

    if structural_issues:
        quality_score = min(quality_score, 65)

    if grounding_issues:
        # Factual grounding failures are more serious than formatting
        # issues -- cap the score lower so a well-formatted-but-
        # hallucinating newsletter can never score highly.
        quality_score = min(quality_score, 40)

    max_revisions = 2
    if revision_count >= max_revisions:
        needs_revision = False
        combined_suggestions.append(
            "Maximum revision limit reached. Preserve the current newsletter "
            "and report remaining issues for manual review."
        )

    final_result = {
        "needs_revision": needs_revision,
        "issues": combined_issues,
        "suggestions": combined_suggestions,
        "quality_score": quality_score,
        "deterministic_issues": deterministic_issues,
        "grounding_issues": grounding_issues,
        "revision_count": revision_count,
    }

    print("\nFinal critique:")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))

    return final_result


def critic_node(state: dict[str, Any]) -> dict[str, Any]:
    goal = state.get("goal", "")
    newsletter = state.get("draft_newsletter", "")
    articles = state.get("ranked_articles") or state.get("raw_articles") or []
    revision_count = int(state.get("revision_count", 0))

    critique = critique_newsletter(
        goal=goal,
        newsletter=newsletter,
        articles=articles,
        revision_count=revision_count,
    )

    return {
        "critique": critique,
        "approved": not critique["needs_revision"],
    }


if __name__ == "__main__":
    test_goal = "Create a weekly newsletter covering important AI-agent developments."

    test_articles = [
        {
            "title": "Salesforce introduces new AI agents",
            "content_available": True,
            "grounding_status": "grounded",
            "url_resolution_status": "verified_direct",
            "what_happened": "Salesforce introduced AI agents intended to automate sales and customer-support workflows.",
            "key_takeaways": ["The agents target sales automation.", "The agents also target customer-support tasks."],
            "why_it_matters": "Shows enterprise adoption of AI agents.",
            "source_limitations": "None noted.",
        }
    ]

    test_newsletter = """# Weekly AI Agent Insights

## 1. Salesforce introduces new AI agents

**What happened:**
Salesforce introduced AI agents intended to automate sales and customer-support workflows.

**Key takeaways:**
- The agents target sales automation.
- The agents also target customer-support tasks.

**Why it matters:**
Shows enterprise adoption of AI agents.

**Source limitations:**
None noted.

**Source:** [Verified publisher article](https://example.com/salesforce)

## Editor's Takeaway

Enterprise AI agents are increasingly being integrated into business workflows.
"""

    result = critique_newsletter(
        goal=test_goal,
        newsletter=test_newsletter,
        articles=test_articles,
        revision_count=0,
    )

    print("\nCritic result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
