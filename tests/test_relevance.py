"""
Tests for agent/relevance.py: deterministic AI-agent relevance scoring,
two-tier (strict/moderate) selection, query-aware bonus scoring, and
publisher-diversity selection.
"""

from agent.relevance import (
    RELEVANCE_MODERATE_THRESHOLD,
    RELEVANCE_STRICT_THRESHOLD,
    diversify_by_publisher,
    is_relevant_article,
    rank_and_select_articles,
    relevance_tier,
    score_article_relevance,
)


def _article(title, summary="", source="Example", provider="direct_rss", content="", matched_query=""):
    return {
        "title": title,
        "summary": summary,
        "source": source,
        "provider": provider,
        "content": content,
        "matched_query": matched_query,
    }


def test_strong_agent_phrase_in_title_scores_well_above_threshold():
    article = _article("New AI Agent Framework Ships With Multi-Step Planning")
    score = score_article_relevance(article)
    assert score >= RELEVANCE_STRICT_THRESHOLD
    assert is_relevant_article(article) is True


def test_autonomous_agent_platform_is_relevant():
    article = _article(
        "Startup Launches Autonomous Agent Platform for Enterprise Customers",
        summary="The agent orchestration platform lets teams automate multi-step workflows.",
    )
    assert is_relevant_article(article) is True


def test_gaming_article_is_rejected_even_with_ai_mention():
    article = _article(
        "This AI-Powered Gaming Console Brings Smarter NPCs to Your Living Room",
        summary="The new video game console uses AI for smarter non-player characters.",
    )
    assert is_relevant_article(article) is False


def test_gta_article_is_rejected():
    article = _article("GTA 6 Delayed Again, Fans React Online", summary="Grand Theft Auto fans are upset.")
    assert is_relevant_article(article) is False


def test_satellite_article_is_rejected():
    article = _article(
        "New Satellite Launched to Study Climate Change",
        summary="The satellite will orbit Earth and collect climate data.",
    )
    assert is_relevant_article(article) is False


def test_automobile_article_is_rejected():
    article = _article(
        "2027 Sedan Review: A Smooth Ride With Modern Features",
        summary="This sedan offers a comfortable ride and modern infotainment.",
    )
    assert is_relevant_article(article) is False


def test_generic_ai_article_without_agent_language_scores_low():
    """General AI/technology news (no agent-specific or capability
    language) should not clear even the moderate bar just because it
    mentions "AI"."""

    article = _article(
        "AI Chip Manufacturer Reports Record Quarterly Revenue",
        summary="The chipmaker's AI hardware division saw strong demand this quarter.",
    )
    assert score_article_relevance(article) < RELEVANCE_MODERATE_THRESHOLD


def test_content_is_considered_when_summary_is_sparse():
    article = _article(
        "Company Announces New Product",
        summary="A short announcement.",
        content="The new product is an autonomous agent platform for coding agents and tool-using LLMs.",
    )
    assert is_relevant_article(article) is True


def test_related_capability_article_without_the_word_agent_is_moderate_or_better():
    """Requirement: do not require the exact phrase 'AI agent'. An
    article about an autonomous coding assistant is clearly agent-like
    even without ever using the word "agent"."""

    article = _article(
        "AWS Launches Autonomous Coding Assistant for Enterprise Developers",
        summary="The new assistant plans and executes multi-step coding tasks automatically.",
    )
    score = score_article_relevance(article)
    assert relevance_tier(score) in {"strict", "moderate"}


def test_query_match_bonus_helps_a_borderline_article_on_its_own_query_topic():
    """The query bonus should top up an article that already shows
    some genuine agent-domain signal (here, "autonomous workflow" from
    the related-capability table) -- not manufacture relevance out of
    an article with no agent-related language at all. See
    test_query_bonus_never_applies_without_any_baseline_agent_signal
    for the failure mode this guards against."""

    borderline_no_query = _article(
        "Startup Debuts Autonomous Workflow Tool With Broader Safety Review Process",
        summary="The tool automates repetitive tasks and adds new security safeguards.",
    )
    borderline_with_query = _article(
        "Startup Debuts Autonomous Workflow Tool With Broader Safety Review Process",
        summary="The tool automates repetitive tasks and adds new security safeguards.",
        matched_query="AI agent safety security",
    )

    score_without = score_article_relevance(borderline_no_query)
    score_with = score_article_relevance(borderline_with_query)
    assert score_with > score_without


def test_query_bonus_never_applies_without_any_baseline_agent_signal():
    """Regression test for a real production failure: a '10 Best
    Standing Desks' buying guide scored as moderately relevant to an
    'AI agent' newsletter purely because its text happened to share
    generic words ("best" from its own title, "building" from
    "building Lego sets") with the research query that surfaced it --
    despite the article containing zero AI/agent-related language
    anywhere. The query-match bonus must never be the sole source of
    an article's relevance; it may only top up an article that already
    shows some genuine agent-domain signal from the phrase tables."""

    standing_desks = _article(
        "10 Best Standing Desks Worth Buying in 2026",
        summary="Take your home office to new heights with our favorite motorized standing desks.",
        source="Wired",
        content=(
            "Not every standing desk is worth the investment. Our team has been "
            "testing desks at home for years, including tasks like building Lego "
            "sets and wrapping holiday gifts."
        ),
        matched_query="best practices for developers building AI agents",
    )
    bioweapons_safeguards = _article(
        "Claude users found ways around safeguards for bioweapons research",
        summary="Some dangerous biology looks much like legitimate research, complicating AI safeguards.",
        source="Ars Technica",
        matched_query="latest research on AI agent safety protocols and alignment",
    )

    assert score_article_relevance(standing_desks) <= 0
    assert score_article_relevance(bioweapons_safeguards) <= 0


def test_source_name_contributes_to_scoring():
    article_a = _article("Weekly Roundup", source="Agent Weekly")
    article_b = _article("Weekly Roundup", source="Generic News Co")
    assert score_article_relevance(article_a) > score_article_relevance(article_b)


def test_diversify_by_publisher_interleaves_across_sources():
    articles = [
        _article("Agent story A1", source="PublisherA"),
        _article("Agent story A2", source="PublisherA"),
        _article("Agent story A3", source="PublisherA"),
        _article("Agent story B1", source="PublisherB"),
        _article("Agent story C1", source="PublisherC"),
    ]

    diversified = diversify_by_publisher(articles)

    first_three_sources = {a["source"] for a in diversified[:3]}
    assert len(first_three_sources) >= 2


def test_diversify_by_publisher_preserves_all_articles():
    articles = [
        _article("A1", source="PublisherA"),
        _article("A2", source="PublisherA"),
        _article("B1", source="PublisherB"),
    ]
    diversified = diversify_by_publisher(articles)
    assert len(diversified) == 3
    assert {a["title"] for a in diversified} == {"A1", "A2", "B1"}


def test_rank_and_select_rejects_irrelevant_and_keeps_relevant():
    articles = [
        _article("New AI Agent Framework Ships", source="TechCrunch"),
        _article("GTA 6 Delayed Again", summary="Grand Theft Auto news.", source="IGN"),
        _article("Autonomous Agents Power New Coding Assistant", source="VentureBeat"),
        _article("New Satellite Launched", summary="A satellite orbits Earth.", source="SpaceNews"),
    ]

    result = rank_and_select_articles(articles, target_count=7)
    selected_titles = {a["title"] for a in result["selected"]}

    assert "New AI Agent Framework Ships" in selected_titles
    assert "Autonomous Agents Power New Coding Assistant" in selected_titles
    assert "GTA 6 Delayed Again" not in selected_titles
    assert "New Satellite Launched" not in selected_titles
    assert len(result["scored"]) == len(articles)
    for article in result["scored"]:
        assert "relevance_score" in article
        assert "relevance_tier" in article


def test_rank_and_select_never_pads_with_irrelevant_articles():
    """If fewer than target_count articles are genuinely relevant
    (strict or moderate), the result must be smaller than target_count
    rather than padded with rejected articles."""

    articles = [
        _article("New AI Agent Framework Ships", source="TechCrunch"),
        _article("GTA 6 Delayed Again", summary="Grand Theft Auto news.", source="IGN"),
        _article("New Satellite Launched", summary="A satellite orbits Earth.", source="SpaceNews"),
    ]

    result = rank_and_select_articles(articles, target_count=7)
    assert len(result["selected"]) == 1
    assert all(a["relevance_tier"] != "rejected" for a in result["selected"])


def test_rank_and_select_applies_publisher_diversity_within_budget():
    articles = [
        _article(f"AI agent story {i}", source="SamePublisher") for i in range(5)
    ] + [
        _article("Autonomous agent platform launch", source="OtherPublisher"),
    ]

    result = rank_and_select_articles(articles, target_count=3)
    sources = {a["source"] for a in result["selected"]}
    assert "OtherPublisher" in sources


def test_rank_and_select_tops_up_with_moderate_tier_when_strict_is_scarce():
    """Requirement: if fewer than 5 articles pass the strict threshold,
    use the moderate tier to try to reach 5-7 -- but only then, and it
    must be clearly labeled."""

    # Only 2 strict-tier articles, but several moderate-tier
    # (agent-capability-adjacent, but weakly worded) articles that
    # should top up the selection instead of leaving the newsletter
    # with just 2 articles.
    articles = [
        _article("New AI Agent Framework Ships", source="TechCrunch"),
        _article("Autonomous Agents Power New Coding Assistant", source="VentureBeat"),
        _article(
            "New Tool Promises Autonomous Task Handling for Teams",
            source="AWS Machine Learning Blog",
        ),
        _article("Company Unveils Copilot for Internal Workflows", source="ZDNET"),
        _article("Startup Adds Orchestration Layer to Developer Platform", source="Engadget"),
        _article("GTA 6 Delayed Again", summary="Grand Theft Auto news.", source="IGN"),
        _article("New Satellite Launched", summary="A satellite orbits Earth.", source="SpaceNews"),
    ]

    result = rank_and_select_articles(articles, target_count=7, min_strict_before_topup=5)

    assert len(result["strict"]) == 2
    assert len(result["moderate"]) == 3
    assert len(result["selected"]) == 5
    assert result["tier_used"] == "strict+moderate"
    assert any(a["relevance_tier"] == "moderate" for a in result["selected"])
    selected_titles = {a["title"] for a in result["selected"]}
    assert "GTA 6 Delayed Again" not in selected_titles
    assert "New Satellite Launched" not in selected_titles


def test_rank_and_select_does_not_top_up_when_strict_alone_is_sufficient():
    articles = [
        _article(f"New AI Agent Framework Update {i}", source=f"Publisher{i}")
        for i in range(5)
    ] + [
        _article("Company Unveils Copilot for Internal Workflows", source="ZDNET"),
    ]

    result = rank_and_select_articles(articles, target_count=7, min_strict_before_topup=5)

    assert result["tier_used"] == "strict"
    assert all(a["relevance_tier"] == "strict" for a in result["selected"])
    assert "Company Unveils Copilot for Internal Workflows" not in {
        a["title"] for a in result["selected"]
    }


def test_rank_and_select_result_reports_all_tiers():
    articles = [
        _article("New AI Agent Framework Ships", source="TechCrunch"),
        _article(
            "AWS Launches Autonomous Coding Assistant",
            summary="Plans and executes multi-step coding tasks.",
            source="AWS",
        ),
        _article("GTA 6 Delayed Again", summary="Grand Theft Auto news.", source="IGN"),
    ]

    result = rank_and_select_articles(articles)
    assert set(result.keys()) == {"selected", "scored", "strict", "moderate", "rejected", "tier_used"}
    assert len(result["strict"]) + len(result["moderate"]) + len(result["rejected"]) == len(articles)
