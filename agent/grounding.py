"""
Deterministic source-grounding utilities.

This module is the single source of truth for:

1. Deciding when an article's content is too weak to trust an LLM with
   (`needs_fallback`), and the exact deterministic fallback wording to
   use in that case (`build_fallback_summary`).
2. Validating a finished newsletter for unsupported claims, mislabeled
   sources, and unfounded Editor's Takeaway generalizations
   (`validate_newsletter_grounding`).

Everything here is deterministic (no LLM calls) so it can be trusted as
a hard safety net regardless of what any LLM call elsewhere does.
"""

from __future__ import annotations

import json
import re
from typing import Any

MIN_CONTENT_LENGTH = 200

# Phrases that assert a concrete factual event. If one of these shows up
# in a section for an article whose content was NOT available, that is a
# strong sign the text is inferring facts from the headline alone rather
# than from real source content. This is a *secondary* signal used on
# top of the primary, structural signal (content_available /
# grounding_status) -- never the only check.
UNSUPPORTED_CLAIM_PHRASES = [
    "released",
    "announced",
    "recommended",
    "found that",
    "discovered",
    "implemented",
    "introduced",
    "the report states",
    "the organization launched",
    "launched",
    "revealed",
    "confirmed",
    "unveiled",
]

# Generic wording that implies a trend across "this week's" sources.
# Only a problem when most/all supporting articles are title-only.
TREND_GENERALIZATION_PHRASES = [
    "this week's articles show",
    "this week's sources show",
    "the overall theme",
    "the available sources emphasize",
    "a clear trend",
    "sources this week highlight",
    "collectively, these articles",
]

DETERMINISTIC_EDITORS_TAKEAWAY = (
    "The available feed contained article titles, but the full article "
    "text could not be retrieved for most sources. Therefore, no "
    "verified trend-level conclusions can be drawn from this week's "
    "sources. Readers should open the source links for complete context."
)


def needs_fallback(article: dict[str, Any]) -> bool:
    """
    True when an article's content is not trustworthy enough to hand to
    an LLM for summarization/writing -- any of these disqualifies it:

    - content missing or empty
    - content too short to be meaningful
    - extraction_status is anything other than "success"
    - the URL is still an unresolved Google News wrapper
    - the article is explicitly marked title-only
    """

    if article.get("is_title_only"):
        return True

    if not article.get("content_available", False):
        return True

    content = (article.get("content") or "").strip()

    if len(content) < MIN_CONTENT_LENGTH:
        return True

    if article.get("extraction_status") != "success":
        return True

    if article.get("url_resolution_status") != "verified_direct":
        # We only trust content pulled from a verified publisher URL.
        return True

    return False


def build_fallback_summary(article: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic fallback for the case where NO usable article content
    was ever available -- extraction failed or was blocked, the content
    was too short to be meaningful, the article is explicitly
    title-only, or the URL was never resolved to a verified publisher
    link. Used whenever needs_fallback() is True, i.e. before the LLM
    is even called. This exact wording must never be replaced with
    invented claims by an LLM.
    """

    return {
        "what_happened": "No details were provided in the source beyond the title.",
        "key_takeaways": [
            "No key takeaways available because the article content could not be retrieved."
        ],
        "why_it_matters": (
            "The title suggests a potentially relevant topic, but its "
            "significance cannot be verified without the article text."
        ),
        "source_limitations": (
            "The article text was unavailable; no factual claims were "
            "inferred from the title."
        ),
        "content_available": False,
        "grounding_status": "fallback",
    }


def build_llm_failed_summary(article: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic fallback for the DIFFERENT case where article content
    WAS successfully extracted (needs_fallback() was False -- extraction
    succeeded and produced real, sufficiently long text from a verified
    URL) but the LLM summarization call itself failed: a rate limit or
    daily token quota error, a network/timeout error, or a malformed or
    unparseable model response.

    This must never claim that only the title was available -- the
    article's content was genuinely retrieved; only the summarization
    step failed. content_available is correctly reported as True so
    downstream consumers (the writer, the critic/reviser prompts, and
    verification logging) can tell this apart from a true extraction
    failure.
    """

    return {
        "what_happened": (
            "The full article text was successfully retrieved, but an "
            "automated summary could not be generated because the "
            "summarization step failed (for example, the LLM service was "
            "rate-limited, unavailable, or returned a response that could "
            "not be parsed)."
        ),
        "key_takeaways": [
            "No AI-generated key takeaways are available because "
            "summarization could not be completed, even though the "
            "article content was available. See the source link for the "
            "full article."
        ],
        "why_it_matters": (
            "Relevance could not be summarized automatically for this "
            "article; readers should consult the original source directly."
        ),
        "source_limitations": (
            "Article content was available, but automated summarization "
            "failed, so no LLM-derived claims are included here."
        ),
        "content_available": True,
        "grounding_status": "error_fallback",
    }


def _strip_code_fences(text: str) -> str:
    """
    Removes Markdown code fences wherever they appear in the text, not
    just when they open/close the entire response. A response like
    "Here is the summary:\n\n```json\n{...}\n```\n\nLet me know if you
    need anything else." previously was not handled -- the old
    anchored (^...$) fence-stripping only matched a fence at the very
    start/end of the whole string.
    """

    cleaned = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()


def _find_balanced_json_objects(text: str) -> list[str]:
    r"""
    Finds every top-level {...} substring using brace-depth tracking
    (respecting string literals, so a brace inside a quoted string
    never affects nesting depth) rather than a naive greedy regex like
    `\{.*\}`, which grabs from the FIRST `{` to the LAST `}` in the
    whole text -- that can accidentally span unrelated braces in
    surrounding prose, or silently include trailing content that isn't
    part of the JSON object at all.
    """

    candidates: list[str] = []
    depth = 0
    start_index: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_index is not None:
                    candidates.append(text[start_index : index + 1])
                    start_index = None

    return candidates


def _attempt_close_truncated_json(text: str) -> str | None:
    """
    Best-effort structural repair for a response that was cut off
    mid-object -- most commonly because a max_tokens output cap ended
    generation before the model finished writing the closing braces.
    This only closes already-open strings/objects/arrays; it never
    invents or guesses at field values, so it can only ever produce a
    syntactically-valid version of exactly the content the model
    already generated (with an already-open string closed, and a
    dangling trailing comma trimmed).

    Returns None if the text isn't actually truncated (braces already
    balanced), since there's nothing to repair.
    """

    depth_stack: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            depth_stack.append(char)
        elif char in "}]":
            if depth_stack:
                depth_stack.pop()

    if not depth_stack and not in_string:
        return None

    repaired = text
    if in_string:
        repaired += '"'

    # A truncated object often ends mid-way through writing the next
    # key/value pair (e.g. `..."issues": ["one", "tw`) or right after a
    # trailing comma -- trim a dangling comma before closing so the
    # repaired JSON doesn't end with a syntax error of its own.
    repaired = re.sub(r",\s*$", "", repaired)

    for opener in reversed(depth_stack):
        repaired += "}" if opener == "{" else "]"

    return repaired


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort extraction of a JSON object from an LLM response,
    handling three real-world failure modes seen in production:

    1. A fenced JSON block with prose before and/or after it.
    2. Multiple brace-delimited fragments in the text, where a naive
       first-{-to-last-} match would span the wrong range.
    3. A response truncated mid-object by a max_tokens output cap --
       repaired structurally (closing already-open syntax only, never
       inventing field values) before giving up.
    """

    cleaned = _strip_code_fences(text.strip())

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for candidate in _find_balanced_json_objects(cleaned):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # No balanced (syntactically complete) object was found -- try
    # treating the text from the first "{" onward as a truncated
    # object and structurally repair it.
    if "{" in cleaned:
        start = cleaned.index("{")
        repaired = _attempt_close_truncated_json(cleaned[start:])
        if repaired:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    print("extract_json_object: repaired a truncated JSON response.")
                    return parsed
            except json.JSONDecodeError:
                pass

    raise ValueError("No JSON object found in LLM response.")


def _find_unsupported_phrases(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in UNSUPPORTED_CLAIM_PHRASES if phrase in lowered]


def validate_newsletter_grounding(
    newsletter_markdown: str,
    articles: list[dict[str, Any]],
) -> list[str]:
    """
    Deterministic grounding checks over the finished newsletter, using
    structured article metadata wherever possible (content_available /
    grounding_status / url_resolution_status) rather than relying on
    keyword matching alone.
    """

    issues: list[str] = []

    if not newsletter_markdown or not newsletter_markdown.strip():
        return ["Newsletter is empty."]

    ungrounded_count = 0
    grounded_count = 0

    for article in articles:
        title = article.get("title", "Untitled article")
        is_ungrounded = (
            not article.get("content_available", False)
            or article.get("grounding_status") in {"fallback", "downgraded_after_revision"}
        )

        if is_ungrounded:
            ungrounded_count += 1
        else:
            grounded_count += 1

        # Find this article's section in the newsletter by title so we
        # only inspect the text that actually belongs to it.
        escaped_title = re.escape(title[:60])
        section_match = re.search(
            rf"{escaped_title}.*?(?=\n##\s|\Z)",
            newsletter_markdown,
            flags=re.DOTALL | re.IGNORECASE,
        )
        section_text = section_match.group(0) if section_match else ""

        if is_ungrounded and section_text:
            hits = _find_unsupported_phrases(section_text)
            if hits:
                issues.append(
                    f"Article '{title}' has unavailable source content but "
                    f"its section contains factual-claim language ({', '.join(hits)}) "
                    "that is not supported by any retrieved article text."
                )

            if "no key takeaways available" not in section_text.lower():
                issues.append(
                    f"Article '{title}' has unavailable content but does not "
                    "use the required deterministic fallback wording for its "
                    "key takeaways."
                )

        # Source-labeling accuracy: an unresolved (wrapper-only) article
        # must never be labeled as a verified/original article.
        resolution_status = article.get("url_resolution_status", "unresolved")
        if section_text:
            lowered_section = section_text.lower()
            if resolution_status != "verified_direct" and (
                "verified publisher article" in lowered_section
                or "original article" in lowered_section
            ):
                issues.append(
                    f"Article '{title}' only has an unresolved Google News "
                    "reference URL but is labeled as a verified/original "
                    "article."
                )

            if resolution_status == "verified_direct" and (
                "google news reference" in lowered_section
            ):
                issues.append(
                    f"Article '{title}' has a verified publisher URL but is "
                    "still labeled as a Google News reference."
                )

    # Editor's Takeaway generalization check.
    takeaway_match = re.search(
        r"##\s*Editor.{0,2}s Takeaway\s*\n(.*)",
        newsletter_markdown,
        flags=re.DOTALL | re.IGNORECASE,
    )
    takeaway_text = takeaway_match.group(1).strip() if takeaway_match else ""

    total_articles = len(articles)
    majority_ungrounded = (
        total_articles > 0 and ungrounded_count >= max(1, total_articles - grounded_count)
        and grounded_count < max(1, total_articles / 2)
    )

    if takeaway_text and majority_ungrounded:
        lowered_takeaway = takeaway_text.lower()
        makes_generalization = any(
            phrase in lowered_takeaway for phrase in TREND_GENERALIZATION_PHRASES
        )
        discloses_limitation = (
            "could not be retrieved" in lowered_takeaway
            or "no verified trend" in lowered_takeaway
            or "cannot be verified" in lowered_takeaway
        )

        if makes_generalization and not discloses_limitation:
            issues.append(
                "Editor's Takeaway generalizes a trend even though most "
                "articles are title-only with unavailable source content."
            )

        if not discloses_limitation and grounded_count == 0:
            issues.append(
                "Editor's Takeaway does not disclose that no article "
                "content could be verified this week."
            )

    return issues
