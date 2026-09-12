"""
Newsletter export utilities.

Centralizes Markdown -> HTML rendering and file writing so that both
the autonomous CLI path (agent/runner.py) and the Streamlit
Human-in-the-Loop approval path (app.py) export a newsletter in
exactly the same way, instead of duplicating the logic in two places.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown as markdown_lib

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.6;
            color: #222;
        }}

        h1, h2, h3 {{
            color: #1f2937;
        }}

        h1 {{
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 12px;
        }}

        h2 {{
            margin-top: 2em;
        }}

        a {{
            color: #2563eb;
        }}

        hr {{
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 2em 0;
        }}

        strong {{
            color: #111827;
        }}

        ul, ol {{
            padding-left: 1.4em;
        }}

        code {{
            background: #f3f4f6;
            padding: 2px 5px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
{body}
</body>
</html>
"""


def markdown_to_html(
    newsletter_markdown: str,
    title: str = "Weekly AI Agent Insights",
) -> str:
    """
    Convert the newsletter Markdown into a properly rendered HTML page
    (instead of dumping escaped Markdown inside a <pre> block).
    """

    body_html = markdown_lib.markdown(
        newsletter_markdown,
        extensions=["extra", "sane_lists", "nl2br"],
    )

    return _HTML_TEMPLATE.format(title=title, body=body_html)


def extract_subject_line(newsletter_markdown: str) -> str:
    """Use the first Markdown H1 heading as the simulated email subject."""

    match = re.search(
        r"^#\s+(.+)$",
        newsletter_markdown,
        flags=re.MULTILINE,
    )

    if match:
        return match.group(1).strip()

    return "Weekly AI Agent Insights"


def export_newsletter(
    newsletter_markdown: str,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """
    Save the newsletter as timestamped Markdown and HTML files.

    Returns the file paths plus the rendered HTML/subject so callers
    do not need to re-read the files back from disk.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    markdown_path = output_path / f"newsletter_{timestamp}.md"
    html_path = output_path / f"newsletter_{timestamp}.html"

    markdown_path.write_text(newsletter_markdown, encoding="utf-8")

    html_content = markdown_to_html(newsletter_markdown)
    html_path.write_text(html_content, encoding="utf-8")

    return {
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
        "html_content": html_content,
        "subject": extract_subject_line(newsletter_markdown),
    }


def simulate_send(
    newsletter_markdown: str,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """
    Simulate sending the newsletter: save it to disk AND print the
    subject + a content preview to the console. This satisfies both
    options the assignment offers ("save as file OR print the email
    content + subject") at once.
    """

    export_info = export_newsletter(newsletter_markdown, output_dir=output_dir)

    print("\n" + "=" * 60)
    print(f"SIMULATED SEND -- Subject: {export_info['subject']}")
    print("=" * 60)
    preview = newsletter_markdown[:1000]
    print(preview)
    if len(newsletter_markdown) > 1000:
        print("... (truncated - see the saved file for the full newsletter)")
    print(f"\nSaved Markdown to: {export_info['markdown_path']}")
    print(f"Saved HTML to:     {export_info['html_path']}")

    return export_info


if __name__ == "__main__":
    sample = (
        "# Weekly AI Agent Insights\n\n"
        "## 1. Example Article\n\n"
        "**What happened:** Something agentic happened.\n\n"
        "**Source:** [Read the original article](https://example.com)\n"
    )

    simulate_send(sample, output_dir="outputs")
