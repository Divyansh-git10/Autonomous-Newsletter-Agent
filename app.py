from pathlib import Path

import streamlit as st

from agent.exporter import export_newsletter
from agent.reviser import reviser_node
from agent.runner import run_newsletter_agent


st.set_page_config(
    page_title="Autonomous Newsletter Agent",
    page_icon="\U0001F4F0",
    layout="wide",
)


st.title("\U0001F4F0 Autonomous Newsletter Agent")
st.caption(
    "Research → Summarize → Write → Critique → Review → Output"
)
st.caption(
    "**Tools used:** News Search (Direct RSS / Google News / Bing News) • "
    "Article Extraction (httpx + trafilatura) • "
    "LLM Summarization / Writing / Critique (Groq) • "
    "Newsletter Export (Markdown + HTML)"
)

st.divider()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def display_execution_summary(result, mode):
    st.subheader("Execution Summary")

    research_stats = result.get("research_stats") or {}
    total_found = research_stats.get("total_collected", len(result.get("raw_articles", [])))

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Execution Mode",
            "Human-in-the-Loop" if mode == "human" else "Autonomous",
        )

    with col2:
        st.metric(
            "Articles Found",
            total_found,
            help="Total candidates collected across all research providers before relevance filtering.",
        )

    with col3:
        st.metric(
            "Relevant & Selected",
            len(result.get("raw_articles", [])),
            help="Articles that passed the AI-agent relevance filter and publisher-diversity selection.",
        )

    with col4:
        st.metric(
            "Articles Summarized",
            len(result.get("ranked_articles", [])),
        )

    with col5:
        critique = result.get("critique") or {}
        st.metric(
            "Quality Score",
            critique.get("quality_score", 0),
        )

    if research_stats:
        tier_used = research_stats.get("selection_tier", "strict")
        tier_note = (
            " (topped up with moderately-related articles because fewer than 5 "
            "strictly on-topic candidates were found)"
            if tier_used == "strict+moderate"
            else ""
        )
        st.caption(
            f"Relevance filter: {research_stats.get('total_strict_relevant', 0)} strict match(es), "
            f"{research_stats.get('total_moderate_relevant', 0)} moderate match(es), "
            f"{research_stats.get('total_rejected_irrelevant', 0)} rejected as off-topic. "
            f"Selected from {research_stats.get('selected_publisher_count', 0)} distinct publisher(s)."
            f"{tier_note}"
        )


def display_review_information(result):
    st.subheader("Review Information")

    critique = result.get("critique") or {}

    if critique.get("needs_revision"):
        st.warning("The critic suggested revisions.")
    else:
        st.success("The newsletter passed the review checks.")

    issues = critique.get("issues", [])
    suggestions = critique.get("suggestions", [])

    if issues:
        st.markdown("### Issues")
        for issue in issues:
            st.write(f"- {issue}")

    if suggestions:
        st.markdown("### Suggestions")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")

    st.write(
        f"**Automatic revision count:** "
        f"{result.get('revision_count', 0)}"
    )


def display_export_section(newsletter: str):
    st.divider()
    st.subheader("Export Newsletter")

    # Export exactly once per approved draft. Streamlit reruns this whole
    # script on every widget interaction (including clicking a download
    # button below), so without this cache, each click would silently
    # write another duplicate timestamped file pair to outputs/.
    if "export_info" not in st.session_state:
        st.session_state.export_info = export_newsletter(newsletter)

    export_info = st.session_state.export_info

    st.success("Newsletter files created successfully.")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="\U0001F4E5 Download Markdown",
            data=newsletter,
            file_name=Path(export_info["markdown_path"]).name,
            mime="text/markdown",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="\U0001F4E5 Download HTML",
            data=export_info["html_content"],
            file_name=Path(export_info["html_path"]).name,
            mime="text/html",
            use_container_width=True,
        )

    st.caption(f"Markdown saved at: `{export_info['markdown_path']}`")
    st.caption(f"HTML saved at: `{export_info['html_path']}`")

    with st.expander("Preview rendered HTML"):
        st.components.v1.html(
            export_info["html_content"],
            height=500,
            scrolling=True,
        )


# ---------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------

with st.sidebar:
    st.header("Agent Configuration")

    mode_label = st.radio(
        "Execution Mode",
        options=[
            "Fully Autonomous",
            "Human-in-the-Loop",
        ],
    )

    mode = (
       "autonomous"
       if mode_label == "Fully Autonomous"
       else "human"
    )

    if mode == "autonomous":
        st.info(
            "Fully Autonomous mode runs the complete workflow "
            "automatically and exports the final newsletter."
        )
    else:
        st.info(
            "Human-in-the-Loop mode generates a draft first. "
            "The newsletter is exported only after human approval."
        )

    st.divider()
    st.caption("**Tools used by this agent:**")
    st.caption("1. News Search Tool (Direct RSS / Google News / Bing News)")
    st.caption("2. Article Extraction Tool (httpx + trafilatura)")
    st.caption("3. LLM Summarization / Writing / Critique Tool (Groq)")
    st.caption("4. Newsletter Export Tool (Markdown + HTML)")


# ---------------------------------------------------------
# Newsletter goal
# ---------------------------------------------------------

st.subheader("Define Your Newsletter Goal")

default_goal = (
    "Create a weekly newsletter about the latest developments in AI agents, "
    "including new frameworks, enterprise use cases, developer tools, "
    "research, and security updates. Summarize the top relevant articles "
    "and generate a clean newsletter."
)

goal = st.text_area(
    "Plain-English Goal",
    value=default_goal,
    height=150,
    placeholder="Example: Create a weekly newsletter about AI agent news...",
)


run_button = st.button(
    "\U0001F680 Run Newsletter Agent",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# Run agent
# ---------------------------------------------------------

if run_button:
    if not goal.strip():
        st.error("Please enter a newsletter goal.")
        st.stop()

    # Clear any previous pending draft / export cache.
    st.session_state.pop("pending_result", None)
    st.session_state.pop("pending_mode", None)
    st.session_state.pop("approved_result", None)
    st.session_state.pop("export_info", None)

    st.divider()
    st.subheader("Agent Execution")

    progress_placeholder = st.empty()

    with st.spinner("Agent is researching, writing, and reviewing..."):
        try:
            progress_placeholder.info(
                "\U0001F4CB Planning research queries and collecting articles..."
            )

            result = run_newsletter_agent(
                goal=goal.strip(),
                mode=mode,
            )

            progress_placeholder.success(
                "✅ Agent execution completed successfully."
            )

            if mode == "human":
                st.session_state.pending_result = result
                st.session_state.pending_mode = mode
            else:
                st.session_state.approved_result = result

                # Autonomous mode already exported inside
                # run_newsletter_agent() -- reuse that export instead of
                # writing a second, duplicate copy of the files here.
                if result.get("output_path") and result.get("html_path"):
                    st.session_state.export_info = {
                        "markdown_path": result["output_path"],
                        "html_path": result["html_path"],
                        "html_content": result.get("html_content", ""),
                    }

        except Exception as exc:
            progress_placeholder.error(
                f"❌ Agent execution failed: {exc}"
            )
            st.exception(exc)
            st.stop()


# ---------------------------------------------------------
# Human-in-the-Loop approval section
# ---------------------------------------------------------

if "pending_result" in st.session_state:
    result = st.session_state.pending_result

    display_execution_summary(result, "human")

    st.divider()
    st.subheader("Human Review Required")

    st.warning(
        "The draft newsletter is ready for review. Approve it to export "
        "the final newsletter, or request changes with feedback below."
    )

    st.subheader("Draft Newsletter")

    draft_newsletter = result.get("draft_newsletter") or result.get(
        "final_output",
        "No draft newsletter generated.",
    )

    st.markdown(draft_newsletter)

    st.divider()
    display_review_information(result)

    st.divider()

    feedback = st.text_area(
        "Feedback for revision (optional)",
        key="human_feedback",
        placeholder=(
            "e.g. Add more detail on the OpenAI AgentKit article, or "
            "shorten the enterprise section."
        ),
    )

    approval_col, revision_col = st.columns(2)

    with approval_col:
        approve_button = st.button(
            "✅ Approve and Export Newsletter",
            type="primary",
            use_container_width=True,
        )

    with revision_col:
        revise_button = st.button(
            "\U0001F501 Request Changes",
            use_container_width=True,
        )

    if approve_button:
        st.session_state.approved_result = result
        st.session_state.pop("pending_result", None)
        st.success(
            "Human approval received. The newsletter is approved "
            "and ready for export."
        )
        st.rerun()

    if revise_button:
        # Reuse the same reviser_node the automatic critique loop uses,
        # but drive it from the human's feedback instead of the critic's
        # own issues. This keeps the completed research/summaries intact
        # and only regenerates the newsletter text -- no need to discard
        # the draft and re-run the whole pipeline from scratch.
        feedback_text = feedback.strip()

        revision_state = dict(result)
        revision_state["critique"] = {
            "issues": (
                [feedback_text]
                if feedback_text
                else ["Human reviewer requested a revision."]
            ),
            "suggestions": [],
        }

        with st.spinner("Revising the newsletter based on your feedback..."):
            try:
                partial = reviser_node(revision_state)
                revision_state.update(partial)
            except Exception as exc:
                st.error(f"Revision failed: {exc}")
                st.exception(exc)
                st.stop()

        st.session_state.pending_result = revision_state
        st.success("Newsletter revised. Please review the updated draft below.")
        st.rerun()


# ---------------------------------------------------------
# Final output section
# ---------------------------------------------------------

if "approved_result" in st.session_state:
    result = st.session_state.approved_result

    display_execution_summary(result, result.get("mode", mode))

    st.divider()
    display_review_information(result)

    st.divider()
    st.subheader("Generated Newsletter")

    newsletter = result.get("final_output") or result.get(
        "draft_newsletter",
        "No newsletter output generated.",
    )

    st.markdown(newsletter)

    display_export_section(newsletter)
