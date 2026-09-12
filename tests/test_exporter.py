"""
Tests for agent/exporter.py: Markdown -> HTML rendering and file export.
"""

from agent.exporter import export_newsletter, extract_subject_line, markdown_to_html


def test_markdown_to_html_renders_real_html_not_a_pre_block():
    html = markdown_to_html("# Title\n\nSome **bold** text.")

    assert "<pre>" not in html
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html


def test_extract_subject_line_uses_first_heading():
    assert extract_subject_line("# Weekly AI Agent Insights\n\nBody") == (
        "Weekly AI Agent Insights"
    )


def test_extract_subject_line_falls_back_when_no_heading():
    assert extract_subject_line("no heading here") == "Weekly AI Agent Insights"


def test_export_newsletter_writes_md_and_html(tmp_path):
    newsletter = "# Weekly AI Agent Insights\n\nSome content here."

    info = export_newsletter(newsletter, output_dir=tmp_path)

    md_path = tmp_path / info["markdown_path"].split("/")[-1].split("\\")[-1]
    html_path = tmp_path / info["html_path"].split("/")[-1].split("\\")[-1]

    assert md_path.exists()
    assert html_path.exists()
    assert md_path.read_text(encoding="utf-8") == newsletter
    assert "<pre>" not in html_path.read_text(encoding="utf-8")
