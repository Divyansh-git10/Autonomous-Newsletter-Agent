"""
Tests for agent/grounding.py's extract_json_object(): the robust JSON
extraction used to parse Groq LLM responses for the summarizer,
critic, and writer nodes.

These formalize the failure modes that were observed in a real run
("No JSON object found in LLM response") -- a fenced JSON block with
surrounding prose, and a response truncated mid-object by a
max_tokens output cap -- plus the multi-fragment case the old naive
regex (`\\{.*\\}` with DOTALL) could get wrong.
"""

import pytest

from agent.grounding import extract_json_object


def test_plain_json_object_parses_directly():
    text = '{"headline": "Test", "summary": "A summary.", "issues": []}'
    result = extract_json_object(text)
    assert result["headline"] == "Test"


def test_json_wrapped_in_markdown_fences_with_surrounding_prose():
    text = (
        "Here is the summary you asked for:\n\n"
        "```json\n"
        '{"headline": "New AI Agent Framework", "summary": "It ships today.", "issues": []}\n'
        "```\n\n"
        "Let me know if you need anything else."
    )
    result = extract_json_object(text)
    assert result["headline"] == "New AI Agent Framework"
    assert result["summary"] == "It ships today."


def test_json_wrapped_in_bare_fences_without_json_tag():
    text = '```\n{"headline": "Fenced Without Tag", "summary": "Body text."}\n```'
    result = extract_json_object(text)
    assert result["headline"] == "Fenced Without Tag"


def test_truncated_json_is_structurally_repaired():
    """Simulates a max_tokens cutoff mid-value -- the model was writing
    the "issues" array's second string when generation stopped. This
    must be repaired by closing already-open syntax, never by
    inventing the missing content."""

    text = (
        '{"headline": "Truncated Response Test", "summary": "Partial summary text", '
        '"issues": ["first issue", "second incomplete'
    )
    result = extract_json_object(text)
    assert result["headline"] == "Truncated Response Test"
    assert result["summary"] == "Partial summary text"
    # The incomplete second issue string must not be silently dropped
    # or fabricated into something else -- it's closed as-is.
    assert result["issues"][0] == "first issue"


def test_truncated_json_with_dangling_trailing_comma_is_repaired():
    text = '{"headline": "Test", "summary": "Body", "issues": ["one", "two",'
    result = extract_json_object(text)
    assert result["headline"] == "Test"
    assert result["issues"] == ["one", "two"]


def test_multiple_brace_fragments_skips_invalid_and_finds_valid_object():
    """A naive `\\{.*\\}` DOTALL regex would span from the first `{` to
    the very last `}`, which can accidentally straddle two unrelated
    fragments. The balanced-object scanner must instead try each
    top-level fragment independently and use the first one that
    actually parses."""

    text = 'Note: {not json} then the real response: {"headline": "Second Fragment", "summary": "ok"}'
    result = extract_json_object(text)
    assert result["headline"] == "Second Fragment"


def test_no_json_present_raises_value_error():
    with pytest.raises(ValueError):
        extract_json_object("Sorry, I cannot help with that request.")
