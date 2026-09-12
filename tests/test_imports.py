"""
Smoke test: every project module should import cleanly.

This catches broken imports, syntax errors, and circular-import
mistakes before anything else runs.
"""

import importlib

import pytest

MODULES = [
    "agent.state",
    "agent.grounding",
    "agent.planner",
    "agent.research",
    "agent.content",
    "agent.summarize",
    "agent.writer",
    "agent.critic",
    "agent.reviser",
    "agent.exporter",
    "agent.graph",
    "agent.runner",
    "tools.llm_utils",
    "tools.news_search",
    "tools.article_extractor",
    "tools.summarizer",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
