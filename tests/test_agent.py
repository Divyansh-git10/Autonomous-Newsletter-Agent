"""
End-to-end tests for the LangGraph workflow and run_newsletter_agent(),
fully offline via the fake_llm / mock_pipeline_io fixtures.
"""

import pytest

from agent.runner import run_newsletter_agent


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        run_newsletter_agent(goal="Create a newsletter", mode="not-a-real-mode")


def test_empty_goal_is_rejected():
    with pytest.raises(ValueError):
        run_newsletter_agent(goal="   ", mode="autonomous")


def test_autonomous_mode_runs_and_exports(fake_llm, mock_pipeline_io, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = run_newsletter_agent(
        goal="Create a weekly newsletter on the latest AI agent news.",
        mode="autonomous",
    )

    assert result["mode"] == "autonomous"
    assert result["final_output"].startswith("# Weekly AI Agent Insights")
    assert len(result["raw_articles"]) > 0
    assert len(result["ranked_articles"]) > 0
    assert result["output_path"]
    assert result["html_path"]

    saved_files = list((tmp_path / "outputs").glob("newsletter_*.md"))
    assert len(saved_files) == 1  # exactly one export, not a duplicate


def test_human_mode_does_not_export_before_approval(fake_llm, mock_pipeline_io, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = run_newsletter_agent(
        goal="Create a weekly newsletter on the latest AI agent news.",
        mode="human",
    )

    assert result["mode"] == "human"
    assert result["final_output"].startswith("# Weekly AI Agent Insights")
    assert not result.get("output_path")
    assert not result.get("html_path")

    outputs_dir = tmp_path / "outputs"
    assert not outputs_dir.exists() or not list(outputs_dir.glob("newsletter_*.md"))


def test_final_output_is_always_populated(fake_llm, mock_pipeline_io, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = run_newsletter_agent(
        goal="Create a weekly newsletter on the latest AI agent news.",
        mode="human",
    )
    assert result["final_output"]
    assert result["final_output"] == result["draft_newsletter"]


def test_research_node_interleaves_queries_and_dedupes(mock_pipeline_io):
    from agent.research import research_node

    result = research_node({"search_queries": ["query one", "query two"]})

    urls = [a["url"] for a in result["raw_articles"]]
    assert len(urls) == len(set(urls))
    assert len(result["raw_articles"]) > 0


def test_research_node_requires_queries():
    from agent.research import research_node

    with pytest.raises(ValueError):
        research_node({"search_queries": []})


def test_reviser_advances_revision_count_even_with_no_issues():
    from agent.reviser import reviser_node

    state = {
        "goal": "test goal",
        "draft_newsletter": "# Weekly AI Agent Insights\n",
        "critique": {"issues": [], "suggestions": []},
        "ranked_articles": [],
        "revision_count": 0,
    }

    result = reviser_node(state)
    assert result["revision_count"] == 1


def test_graph_cannot_loop_forever_when_critic_always_demands_revision(
    fake_llm, mock_pipeline_io, tmp_path, monkeypatch
):
    """
    Regression test: even if the critic (deterministic or LLM) keeps
    saying needs_revision=True forever, the graph must still terminate
    within the revision-count cap instead of looping indefinitely.
    """
    monkeypatch.chdir(tmp_path)
    fake_llm.critic_needs_revision = True
    fake_llm.critic_issues = ["The editor always wants one more pass."]

    result = run_newsletter_agent(
        goal="Create a weekly newsletter on the latest AI agent news.",
        mode="autonomous",
    )

    # Must terminate (this call returning at all proves no infinite loop),
    # and the revision cap (2) must have been respected.
    assert result["revision_count"] <= 2
    assert result["final_output"]


def test_pipeline_filters_out_irrelevant_articles_end_to_end(fake_llm, tmp_path, monkeypatch):
    """
    Regression test for the relevance-filtering requirement: even if
    the research layer returns a mix of genuinely AI-agent articles and
    unrelated general-tech/entertainment articles, only the AI-agent
    articles should survive into the final newsletter.
    """
    monkeypatch.chdir(tmp_path)

    mixed_articles = [
        {
            "title": "New AI Agent Framework Ships With Multi-Step Planning",
            "source": "TechCrunch",
            "provider": "direct_rss",
            "source_url": "https://techcrunch.com",
            "published": "Mon, 01 Sep 2026 00:00:00 GMT",
            "published_ts": 1798761601.0,
            "summary": "A new AI agent framework was released for developers.",
            "rss_url": "https://techcrunch.com/agent-framework",
            "url": "https://techcrunch.com/agent-framework",
            "resolved_url": "https://techcrunch.com/agent-framework",
            "url_resolution_status": "verified_direct",
            "url_resolution_method": "direct_rss_feed",
        },
        {
            "title": "GTA 6 Delayed Again, Fans React Online",
            "source": "IGN",
            "provider": "bing_news",
            "source_url": "https://ign.com",
            "published": "Mon, 01 Sep 2026 00:00:00 GMT",
            "published_ts": 1798761602.0,
            "summary": "Grand Theft Auto 6 has been delayed once more, upsetting fans.",
            "rss_url": "https://ign.com/gta6-delay",
            "url": "https://ign.com/gta6-delay",
            "resolved_url": "https://ign.com/gta6-delay",
            "url_resolution_status": "verified_direct",
            "url_resolution_method": "direct_rss_link",
        },
        {
            "title": "New Satellite Launched to Study Climate Change",
            "source": "SpaceNews",
            "provider": "google_news",
            "source_url": "https://spacenews.com",
            "published": "Mon, 01 Sep 2026 00:00:00 GMT",
            "published_ts": 1798761603.0,
            "summary": "A new climate-monitoring satellite reached orbit today.",
            "rss_url": "https://spacenews.com/satellite",
            "url": "https://spacenews.com/satellite",
            "resolved_url": "https://spacenews.com/satellite",
            "url_resolution_status": "verified_direct",
            "url_resolution_method": "direct_rss_feed",
        },
        {
            "title": "Autonomous Agents Power New Coding Assistant Platform",
            "source": "VentureBeat",
            "provider": "direct_rss",
            "source_url": "https://venturebeat.com",
            "published": "Mon, 01 Sep 2026 00:00:00 GMT",
            "published_ts": 1798761604.0,
            "summary": "The new coding assistant uses autonomous agents to plan and execute tasks.",
            "rss_url": "https://venturebeat.com/coding-agent",
            "url": "https://venturebeat.com/coding-agent",
            "resolved_url": "https://venturebeat.com/coding-agent",
            "url_resolution_status": "verified_direct",
            "url_resolution_method": "direct_rss_feed",
        },
    ]

    def fake_news_search_tool(query, max_results=10):
        return [dict(a) for a in mixed_articles]

    def fake_extract_article_text(url):
        return {
            "content": (
                "This is a long, realistic article body with well over two "
                "hundred characters describing the story in detail. " * 5
            ),
            "content_available": True,
            "extraction_status": "success",
            "source_url": url,
        }

    monkeypatch.setattr("agent.research.news_search_tool", fake_news_search_tool)
    monkeypatch.setattr("agent.content.extract_article_text", fake_extract_article_text)

    result = run_newsletter_agent(
        goal="Create a weekly newsletter on the latest AI agent news.",
        mode="autonomous",
    )

    selected_titles = [a["title"] for a in result["raw_articles"]]

    assert "New AI Agent Framework Ships With Multi-Step Planning" in selected_titles
    assert "Autonomous Agents Power New Coding Assistant Platform" in selected_titles
    assert "GTA 6 Delayed Again, Fans React Online" not in selected_titles
    assert "New Satellite Launched to Study Climate Change" not in selected_titles

    assert "GTA 6" not in result["final_output"]
    assert "Satellite" not in result["final_output"]

    research_stats = result.get("research_stats") or {}
    assert research_stats.get("total_collected") == 4
    assert research_stats.get("total_rejected_irrelevant") == 2
    assert research_stats.get("total_strict_relevant") == 2
