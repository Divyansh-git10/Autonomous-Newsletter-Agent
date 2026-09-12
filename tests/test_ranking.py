"""
Tests for agent/ranking.py: the LangGraph node that applies relevance
filtering + publisher-diversity selection between research and
content_enrichment, and records provider-level diagnostics.
"""

from agent.ranking import TARGET_ARTICLE_COUNT, ranking_node


def _article(title, summary="", source="Example", provider="direct_rss", url_status="verified_direct"):
    return {
        "title": title,
        "summary": summary,
        "source": source,
        "provider": provider,
        "content": "",
        "url_resolution_status": url_status,
    }


def test_ranking_node_filters_out_irrelevant_articles():
    state = {
        "raw_articles": [
            _article("New AI Agent Framework Ships", provider="direct_rss"),
            _article("GTA 6 Delayed Again", summary="Grand Theft Auto fans react.", provider="bing_news"),
            _article("Autonomous Agents Power New Coding Assistant", provider="google_news"),
        ],
        "research_stats": {},
    }

    result = ranking_node(state)

    titles = {a["title"] for a in result["raw_articles"]}
    assert "GTA 6 Delayed Again" not in titles
    assert "New AI Agent Framework Ships" in titles
    assert "Autonomous Agents Power New Coding Assistant" in titles


def test_ranking_node_caps_selection_at_target_count():
    articles = [
        _article(f"New AI Agent Framework Update {i}", source=f"Publisher{i}")
        for i in range(TARGET_ARTICLE_COUNT + 5)
    ]
    result = ranking_node({"raw_articles": articles, "research_stats": {}})
    assert len(result["raw_articles"]) == TARGET_ARTICLE_COUNT


def test_ranking_node_records_research_stats():
    state = {
        "raw_articles": [
            _article("New AI Agent Framework Ships", provider="direct_rss"),
            _article("GTA 6 Delayed Again", summary="Grand Theft Auto fans react.", provider="bing_news"),
        ],
        "research_stats": {
            "total_collected": 2,
            "providers": {
                "direct_rss": {"results": 1, "verified": 1, "unresolved": 0},
                "bing_news": {"results": 1, "verified": 1, "unresolved": 0},
            },
        },
    }

    result = ranking_node(state)
    stats = result["research_stats"]

    assert stats["total_strict_relevant"] == 1
    assert stats["total_moderate_relevant"] == 0
    assert stats["total_rejected_irrelevant"] == 1
    assert stats["total_selected"] == 1
    assert stats["selection_tier"] == "strict"
    assert stats["selected_publisher_count"] == 1
    assert stats["providers"]["direct_rss"]["strict_relevant"] == 1
    assert stats["providers"]["direct_rss"]["selected"] == 1
    assert stats["providers"]["bing_news"]["strict_relevant"] == 0
    assert stats["providers"]["bing_news"]["selected"] == 0


def test_ranking_node_returns_empty_list_when_nothing_is_relevant():
    state = {
        "raw_articles": [
            _article("GTA 6 Delayed Again", summary="Grand Theft Auto fans react.", provider="bing_news"),
            _article("New Satellite Launched", summary="A satellite orbits Earth.", provider="google_news"),
        ],
        "research_stats": {},
    }

    result = ranking_node(state)
    assert result["raw_articles"] == []
    assert result["research_stats"]["total_selected"] == 0


def test_ranking_node_handles_empty_input_safely():
    result = ranking_node({"raw_articles": [], "research_stats": {}})
    assert result["raw_articles"] == []
    assert result["research_stats"]["total_collected"] == 0


def test_ranking_node_tops_up_with_moderate_tier_and_labels_tier_used():
    state = {
        "raw_articles": [
            _article("New AI Agent Framework Ships", source="TechCrunch", provider="direct_rss"),
            _article("Autonomous Agents Power New Coding Assistant", source="VentureBeat", provider="google_news"),
            _article(
                "New Tool Promises Autonomous Task Handling for Teams",
                source="AWS Machine Learning Blog",
                provider="direct_rss",
            ),
            _article("Company Unveils Copilot for Internal Workflows", source="ZDNET", provider="bing_news"),
            _article(
                "Startup Adds Orchestration Layer to Developer Platform",
                source="Engadget",
                provider="direct_rss",
            ),
            _article("GTA 6 Delayed Again", summary="Grand Theft Auto news.", source="IGN", provider="bing_news"),
            _article(
                "New Satellite Launched", summary="A satellite orbits Earth.", source="SpaceNews", provider="google_news"
            ),
        ],
        "research_stats": {},
    }

    result = ranking_node(state)
    stats = result["research_stats"]

    assert stats["total_strict_relevant"] == 2
    assert stats["total_moderate_relevant"] == 3
    assert stats["total_selected"] == 5
    assert stats["selection_tier"] == "strict+moderate"
    titles = {a["title"] for a in result["raw_articles"]}
    assert "GTA 6 Delayed Again" not in titles
    assert "New Satellite Launched" not in titles


def test_ranking_node_prefers_at_least_three_distinct_publishers_when_available():
    state = {
        "raw_articles": [
            _article("New AI Agent Framework Ships", source="TechCrunch", provider="direct_rss"),
            _article("Second AI Agent Framework Update", source="TechCrunch", provider="direct_rss"),
            _article("Autonomous Agents Power New Coding Assistant", source="VentureBeat", provider="google_news"),
            _article("Enterprise Agent Orchestration Platform Debuts", source="ZDNET", provider="bing_news"),
            _article("New Multi-Agent System For Browser Automation", source="Engadget", provider="direct_rss"),
        ],
        "research_stats": {},
    }

    result = ranking_node(state)
    stats = result["research_stats"]

    assert stats["selected_publisher_count"] >= 3
