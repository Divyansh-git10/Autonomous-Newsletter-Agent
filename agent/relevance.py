"""
Deterministic AI-agent relevance scoring, two-tier (strict/moderate)
selection, and publisher-diversity balancing.

This module never calls an LLM -- it exists specifically so that
extraction (network fetches) and summarization (Groq tokens) are only
ever spent on articles that are actually about AI agents, not generic
AI/technology stories that happened to share a keyword with the
research queries (e.g. an "AI-powered" gaming console, a satellite
launch that mentions "autonomous" navigation, a GTA remaster covered by
a tech outlet).

Scoring is intentionally simple and inspectable rather than a black
box, and is tuned to avoid two opposite failure modes seen in
production runs:

- Too loose: accepting generic AI/tech stories that only glancingly
  touch "AI" or "autonomous".
- Too strict: rejecting genuinely relevant articles just because they
  don't contain the literal phrase "AI agent" -- e.g. "AWS launches new
  autonomous coding assistant for enterprise developers" is clearly an
  agent story even without the word "agent" appearing verbatim.

Four signal tiers feed the score: STRONG agent-specific phrases,
RELATED broader-AI-capability phrases that often describe agentic
systems without using agent terminology, WEAK single-word terms that
are only a mild positive on their own, and IRRELEVANT off-topic
markers that subtract from the score. A separate, smaller
query-relevance bonus rewards an article for matching the *specific*
research query that found it (see `matched_query`), which is what lets
a moderately-worded but on-topic article clear the bar without ever
requiring an exact phrase match.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------
# Term tables
# --------------------------------------------------------------------

# Specific, multi-word phrases whose presence is a strong signal the
# article's core subject is actually an AI agent (not merely "AI" in
# general).
_STRONG_AGENT_PHRASES: dict[str, float] = {
    "ai agent": 5,
    "ai agents": 5,
    "agentic ai": 5,
    "autonomous agent": 5,
    "autonomous agents": 5,
    "multi-agent": 5,
    "multi agent": 5,
    "multiagent": 5,
    "multi-agent system": 5.5,
    "multi-agent systems": 5.5,
    "agent orchestration": 5,
    "agent framework": 4.5,
    "agent frameworks": 4.5,
    "agent platform": 4,
    "agent platforms": 4,
    "tool-using llm": 4,
    "tool-use llm": 4,
    "computer-use agent": 5,
    "computer use agent": 5,
    "computer-using agent": 5,
    "browser-use agent": 5,
    "browser use agent": 5,
    "browser agent": 4,
    "browser agents": 4,
    "coding agent": 4.5,
    "coding agents": 4.5,
    "code agent": 3.5,
    "agentic workflow": 4,
    "agentic workflows": 4,
    "agentic coding": 4,
    "agent safety": 4.5,
    "agent security": 4.5,
    "agent evaluation": 4,
    "agent evaluations": 4,
    "agent benchmark": 3.5,
    "agent deployment": 4,
    "agent deployments": 4,
    "enterprise ai agent": 5,
    "enterprise ai agents": 5,
    "enterprise agent": 4,
    "enterprise agents": 4,
    "agent orchestration platform": 5.5,
    "model context protocol": 4,
    "mcp server": 3.5,
    "mcp servers": 3.5,
    "mcp client": 3,
}

# Broader AI-capability phrases that frequently describe agentic
# systems (planning, acting, automating multi-step work) WITHOUT using
# the word "agent" at all. These score meaningfully -- not as high as
# an explicit agent phrase, but high enough on their own (or combined
# with a query-match bonus) to clear the strict bar for a genuinely
# on-topic article, per the "do not require the exact phrase 'AI
# agent'" requirement.
_RELATED_CAPABILITY_PHRASES: dict[str, float] = {
    "ai assistant": 2.5,
    "ai assistants": 2.5,
    "coding assistant": 3,
    "coding assistants": 3,
    "ai coding assistant": 4,
    "digital assistant": 2,
    "autonomous coding": 3.5,
    "autonomous system": 2,
    "autonomous systems": 2,
    "autonomous workflow": 3,
    "autonomous workflows": 3,
    "ai-powered automation": 3,
    "workflow automation": 2.5,
    "task automation": 2,
    "process automation": 2,
    "ai orchestration": 3.5,
    "llm orchestration": 3.5,
    "llm-powered automation": 3,
    "generative ai workflow": 2.5,
    "generative ai workflows": 2.5,
    "multi-step planning": 3,
    "multi-step reasoning": 2.5,
    "tool calling": 2.5,
    "tool-calling": 2.5,
    "function calling": 1.5,
    "ai copilot": 2.5,
    "ai copilots": 2.5,
    "ai-powered assistant": 3,
}

# Generic single-word / short terms that are only a weak positive
# signal on their own -- common in adjacent AI coverage too, so they
# are weighted much lower than the phrase tables above.
_WEAK_AGENT_TERMS: dict[str, float] = {
    "agent": 2,
    "agents": 2,
    "agentic": 3,
    "orchestration": 1,
    "autonomous": 1,
    "copilot": 1,
    "copilots": 1,
    "tool use": 1,
}

# Subjects that, when they dominate an article, mean it is not really
# an AI-agent story even if it technically mentions "AI" somewhere
# (e.g. "This AI-powered gaming console..."). These subtract from the
# score rather than being an outright keyword-ban, so an article that
# is genuinely about, say, an AI agent used for autonomous vehicle
# routing isn't unfairly zeroed out by the word "vehicle" alone.
_IRRELEVANT_TERMS: dict[str, float] = {
    "video game": 4,
    "video games": 4,
    "gaming console": 4,
    "playstation": 4,
    "xbox": 4,
    "nintendo": 4,
    "grand theft auto": 6,
    "gta 6": 6,
    "gta vi": 6,
    "satellite": 4,
    "satellites": 4,
    "rocket launch": 3,
    "spacex launch": 3,
    "smartphone review": 3,
    "iphone review": 3,
    "earbuds": 3,
    "headphones": 3,
    "automobile": 3,
    "car review": 3,
    "sedan": 3,
    "pickup truck": 3,
    "suv": 2,
    "box office": 4,
    "movie review": 4,
    "tv series": 3,
    "celebrity": 4,
    "streaming service": 2,
    "football": 3,
    "basketball": 3,
    "olympics": 3,
    "cryptocurrency price": 2,
    "stock price": 2,
    "smartwatch": 2,
}

# Words a research query commonly contains that carry no discriminating
# signal for the query-match bonus below (either pure filler, or
# already scored via the term tables above -- rewarding them again via
# the query bonus would double-count).
_QUERY_BONUS_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "about", "latest", "news", "today", "this", "week", "new", "update",
    "updates", "recent", "ai", "agent", "agents", "agentic",
}

# --------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------

# The primary quality bar. An article at or above this score is
# unambiguously an AI-agent story.
RELEVANCE_STRICT_THRESHOLD = 3.0

# A second, more permissive bar for articles that are plausibly
# agent-related (broader AI-capability language, or a query-specific
# match) but don't reach the strict bar. This tier is ONLY used to top
# up the selection when the strict tier alone doesn't produce enough
# candidates (see rank_and_select_articles) -- it is never used to
# silently replace the strict bar, and articles below it are still
# rejected outright.
RELEVANCE_MODERATE_THRESHOLD = 1.0

# Backward-compat alias for the previous single-threshold name.
RELEVANCE_ACCEPT_THRESHOLD = RELEVANCE_STRICT_THRESHOLD

# Try to reach this many articles in the final selection...
TARGET_ARTICLE_COUNT = 7
# ...but only dip into the moderate tier if the strict tier alone
# doesn't reach at least this many.
MIN_STRICT_BEFORE_MODERATE_TOPUP = 5


def _text_fields(article: dict[str, Any]) -> tuple[str, str, str, str]:
    title = (article.get("title") or "").lower()
    summary = (article.get("summary") or "").lower()
    source = (article.get("source") or "").lower()
    # Only the first couple thousand characters of extracted content
    # are considered -- the subject of an article is established early,
    # and this keeps scoring cheap even for long articles.
    content = (article.get("content") or "")[:2000].lower()
    return title, summary, source, content


def _query_keywords(query: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return {w for w in words if w not in _QUERY_BONUS_STOPWORDS and len(w) > 2}


def _query_match_bonus(body: str, matched_query: str) -> float:
    """
    Rewards an article for matching the *specific* research query that
    found it (e.g. a "AI agent safety and security" query rewards an
    article whose text discusses "safety"/"security"), rather than
    requiring every article to independently justify itself purely on
    generic agent terminology. Capped so a query with many keywords
    can't dominate the score on its own.

    Matching uses word boundaries (not raw substring search) so a
    short keyword can't accidentally match inside an unrelated longer
    word -- a real production case: the query keyword "product"
    matched inside "production" in an article about system monitoring
    that had nothing to do with product launches.

    IMPORTANT: this function only checks for keyword overlap -- it is
    the caller's responsibility (see score_article_relevance) to only
    invoke this when the article already shows at least some baseline
    agent-domain signal from the phrase tables. Without that gate,
    this bonus alone can and did produce false positives in
    production: a "10 Best Standing Desks" buying guide scored above
    the moderate-relevance threshold purely because its text happened
    to contain the words "best" (from its own title) and "building"
    (from "building Lego sets," nothing to do with software), and a
    story about bypassing Claude's safety guardrails for bioweapons
    research scored moderate purely because the word "research"
    appeared in both the query and the headline -- neither article
    contained a single AI-agent-related word anywhere.
    """

    keywords = _query_keywords(matched_query)
    if not keywords:
        return 0.0

    hits = sum(
        1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", body)
    )
    return min(hits, 3) * 1.0


def score_article_relevance(article: dict[str, Any]) -> float:
    """
    Returns a deterministic relevance score. Higher is more clearly an
    AI-agent (or clearly agent-adjacent) story. Title matches are
    weighted more heavily than summary/content/source matches, since
    the title is the strongest signal of an article's actual subject.
    Uses title, summary/description, source name, extracted content
    (where available), and a bonus for matching the specific research
    query that surfaced the article (`matched_query`, set by
    agent/research.py).
    """

    title, summary, source, content = _text_fields(article)
    body = f"{summary} {source} {content}"

    # Computed separately from the irrelevant-term penalty and the
    # query-match bonus below because it doubles as a gate: the query
    # bonus is only ever allowed to apply on TOP of some genuine
    # agent-domain signal, never as the sole source of relevance. See
    # _query_match_bonus's docstring for the production false
    # positives (a standing-desk buying guide, a bioweapons-safeguard
    # story) this gate exists specifically to prevent -- both scored
    # as "moderate" purely from generic query-keyword overlap despite
    # containing zero agent-related language anywhere in the article.
    phrase_score = 0.0

    for phrase, weight in _STRONG_AGENT_PHRASES.items():
        if phrase in title:
            phrase_score += weight * 2
        elif phrase in body:
            phrase_score += weight

    for phrase, weight in _RELATED_CAPABILITY_PHRASES.items():
        if phrase in title:
            phrase_score += weight * 1.75
        elif phrase in body:
            phrase_score += weight

    for term, weight in _WEAK_AGENT_TERMS.items():
        if term in title:
            phrase_score += weight * 1.5
        elif term in body:
            phrase_score += weight * 0.5

    penalty = 0.0
    for term, weight in _IRRELEVANT_TERMS.items():
        if term in title:
            penalty += weight * 2
        elif term in body:
            penalty += weight

    score = phrase_score - penalty

    matched_query = article.get("matched_query", "")
    if matched_query and phrase_score > 0:
        score += _query_match_bonus(f"{title} {body}", matched_query)

    return round(score, 2)


def relevance_tier(
    score: float,
    strict_threshold: float = RELEVANCE_STRICT_THRESHOLD,
    moderate_threshold: float = RELEVANCE_MODERATE_THRESHOLD,
) -> str:
    if score >= strict_threshold:
        return "strict"
    if score >= moderate_threshold:
        return "moderate"
    return "rejected"


def is_relevant_article(
    article: dict[str, Any],
    threshold: float = RELEVANCE_STRICT_THRESHOLD,
) -> bool:
    return score_article_relevance(article) >= threshold


def _publisher_key(article: dict[str, Any]) -> str:
    source = (article.get("source") or "").strip().lower()
    if source:
        return source
    return (article.get("provider") or "unknown").strip().lower()


def diversify_by_publisher(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Round-robins articles across publishers so a single source cannot
    fill the whole newsletter when other relevant sources are
    available. Expects `articles` to already be sorted by relevance
    (best first) -- each publisher's own relative order is preserved,
    only the interleaving across publishers changes.
    """

    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []

    for article in articles:
        key = _publisher_key(article)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(article)

    interleaved: list[dict[str, Any]] = []

    while True:
        progressed = False
        for key in order:
            if buckets[key]:
                interleaved.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break

    return interleaved


def rank_and_select_articles(
    articles: list[dict[str, Any]],
    target_count: int = TARGET_ARTICLE_COUNT,
    min_strict_before_topup: int = MIN_STRICT_BEFORE_MODERATE_TOPUP,
    strict_threshold: float = RELEVANCE_STRICT_THRESHOLD,
    moderate_threshold: float = RELEVANCE_MODERATE_THRESHOLD,
) -> dict[str, Any]:
    """
    Scores every article and buckets it into "strict" / "moderate" /
    "rejected" (`relevance_tier` on each scored article). Selection
    tries to reach `target_count` (5-7) using ONLY the strict tier
    first; the moderate tier is used to top up the selection ONLY when
    the strict tier alone falls short of `min_strict_before_topup`.
    Rejected articles are never used to fill the count, regardless of
    how few strict/moderate articles are available.

    Publisher-diversity round-robin is applied within each tier before
    combining, so diversity is preferred within the strict tier first,
    then within the moderate top-up.

    Returns a dict:
        {
            "selected": [...],       # final articles, relevance_score +
                                      # relevance_tier set on each
            "scored": [...],         # every input article, same fields
            "strict": [...],         # scored articles in the strict tier
            "moderate": [...],       # scored articles in the moderate tier
            "rejected": [...],       # scored articles below the moderate bar
            "tier_used": "strict" | "strict+moderate" | "none",
        }
    """

    scored: list[dict[str, Any]] = []
    for article in articles:
        score = score_article_relevance(article)
        tier = relevance_tier(score, strict_threshold, moderate_threshold)
        scored.append({**article, "relevance_score": score, "relevance_tier": tier})

    strict = [a for a in scored if a["relevance_tier"] == "strict"]
    moderate = [a for a in scored if a["relevance_tier"] == "moderate"]
    rejected = [a for a in scored if a["relevance_tier"] == "rejected"]

    strict.sort(key=lambda a: a["relevance_score"], reverse=True)
    moderate.sort(key=lambda a: a["relevance_score"], reverse=True)

    strict_diversified = diversify_by_publisher(strict)

    if len(strict_diversified) >= min_strict_before_topup:
        selected = strict_diversified[:target_count]
        tier_used = "strict" if selected else "none"
    else:
        moderate_diversified = diversify_by_publisher(moderate)
        combined = strict_diversified + moderate_diversified
        selected = combined[:target_count]
        used_moderate = any(a["relevance_tier"] == "moderate" for a in selected)
        if not selected:
            tier_used = "none"
        elif used_moderate:
            tier_used = "strict+moderate"
        else:
            tier_used = "strict"

    return {
        "selected": selected,
        "scored": scored,
        "strict": strict,
        "moderate": moderate,
        "rejected": rejected,
        "tier_used": tier_used,
    }
