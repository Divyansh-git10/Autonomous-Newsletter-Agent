"""
Reviser.

On critique, this node removes unsupported claims rather than merely
rewording them:

- Articles that are already deterministic/fallback are left untouched
  (there is nothing unsupported to remove -- they're template text).
- Articles marked "grounded" are re-summarized from their original
  extracted content with a stricter instruction, since a hallucination
  in a grounded article means the LLM over-reached beyond the supplied
  text, not that the text itself was bad.
- The newsletter is then rebuilt deterministically (agent.writer,
  same as the initial draft) so formatting/labeling can never drift
  from the structured article data.
- A final validation pass re-checks the rebuilt newsletter. If grounding
  issues somehow remain, the offending grounded articles are forcibly
  downgraded to the deterministic fallback template -- a hard
  guarantee that revision can never merely "reword" an unsupported
  claim into a subtler one.
"""

from __future__ import annotations

from typing import Any

from agent.grounding import build_fallback_summary, validate_newsletter_grounding
from agent.writer import write_newsletter
from tools.summarizer import summarize_article_structured


def _downgrade_to_fallback(article: dict[str, Any]) -> dict[str, Any]:
    fallback = build_fallback_summary(article)
    fallback["grounding_status"] = "downgraded_after_revision"
    return {**article, **fallback}


def reviser_node(state: dict[str, Any]) -> dict[str, Any]:
    goal = state.get("goal", "")
    newsletter = state.get("draft_newsletter", "")
    critique = state.get("critique", {})
    articles = state.get("ranked_articles", []) or []
    revision_count = state.get("revision_count", 0)

    issues = critique.get("issues", [])

    if not issues:
        # The critic flagged needs_revision without giving concrete
        # issues. Nothing concrete to act on, but we still must advance
        # revision_count so the graph's revision-limit guard can
        # eventually stop the loop instead of looping forever.
        return {"revision_count": revision_count + 1}

    issues_text = "\n".join(f"- {issue}" for issue in issues)
    revision_note = (
        "A previous version of this summary may have included details "
        "not explicitly present in the supplied article text, or the "
        "reviewing editor flagged the following issues:\n"
        f"{issues_text}\n"
        "Re-derive the summary strictly and conservatively from the "
        "supplied article text only -- remove any claim you cannot "
        "directly trace to that text rather than merely rephrasing it."
    )

    revised_articles: list[dict[str, Any]] = []

    for article in articles:
        if article.get("grounding_status") == "grounded":
            try:
                structured = summarize_article_structured(article, revision_note=revision_note)
                revised_articles.append({**article, **structured})
            except Exception as exc:
                print(f"Revision re-summarization failed for '{article.get('title')}': {exc}")
                revised_articles.append(_downgrade_to_fallback(article))
        else:
            # Already deterministic fallback text -- nothing to revise.
            revised_articles.append(article)

    revised_newsletter = write_newsletter(goal, revised_articles)

    # Final validation pass: the graph must never terminate with
    # unsupported claims left in place. If anything still looks
    # ungrounded after re-summarization, forcibly downgrade rather than
    # accept the risk.
    remaining_issues = validate_newsletter_grounding(revised_newsletter, revised_articles)

    if remaining_issues:
        print("Final validation pass still found grounding issues; downgrading affected articles:")
        for issue in remaining_issues:
            print(f"- {issue}")

        safe_articles = [
            _downgrade_to_fallback(article)
            if article.get("grounding_status") == "grounded"
            else article
            for article in revised_articles
        ]
        revised_newsletter = write_newsletter(goal, safe_articles)
        revised_articles = safe_articles

    return {
        "draft_newsletter": revised_newsletter,
        "ranked_articles": revised_articles,
        "revision_count": revision_count + 1,
    }
