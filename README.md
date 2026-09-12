# Autonomous Newsletter Agent

**Live Demo:** https://autonomous-newsletter-agent.onrender.com
**GitHub Repository:** https://github.com/Divyansh-git10/Autonomous-Newsletter-Agent

A mini autonomous AI agent, built with **LangGraph**, that turns a plain-English
goal into a weekly newsletter about AI agent news. It researches, summarizes,
writes, critiques and revises its own output, then either exports it
automatically (Fully Autonomous mode) or waits for a human to approve or
request changes first (Human-in-the-Loop mode).

The pipeline is hardened against hallucination end-to-end: an LLM is never
allowed to invent facts for an article whose real content could not be
retrieved. See "Source Grounding & Hallucination Prevention" below.

## Live Demo

- **App:** https://autonomous-newsletter-agent.onrender.com
- **Repository:** https://github.com/Divyansh-git10/Autonomous-Newsletter-Agent
- **Deployment platform:** [Render](https://render.com/) (free tier web service)

The app is deployed as-is from this repository -- enter a goal, choose
**Fully Autonomous** or **Human-in-the-Loop** in the sidebar, and click
**Run Newsletter Agent** (see "Running the Frontend" below for what each
mode does). Note: on Render's free tier the service spins down after
inactivity, so the first request after a while can take up to ~30-60
seconds to wake up.

## Assignment Objective

This project implements the "AI Developer Assignment" brief: a mini
autonomous Newsletter Agent that

- researches the latest AI-agent news,
- summarizes the top 5-7 relevant articles,
- generates a clean Markdown/HTML newsletter,
- simulates sending it (saves it to disk and prints a subject + preview),
- runs as a multi-step workflow (planning → research → writing → review → output),
- uses at least 2-3 distinct tools,
- performs self-reflection/critique with revision,
- exposes one autonomous entry point (`run_newsletter_agent(goal)`),
- supports both Fully Autonomous and Human-in-the-Loop modes,
- ships with a simple frontend,

and is packaged for a GitHub repository with a path to a hosted demo (see
"Deployment" below).

## Features

- Plain-English goal input (e.g. *"Create a weekly newsletter on the latest
  AI agent news and send it to our subscribers."*)
- Autonomous research across multiple search queries via Google News RSS
- Multi-strategy URL resolution that tries to recover the real publisher
  URL behind Google's redirect wrapper, without ever fabricating one
- Deterministic relevance filtering + publisher-diversity selection of the
  top ~5-7 articles (see "Relevance Filtering & Source Diversity")
- Strict source-grounding: an article with no verifiable content gets a
  clearly labeled, deterministic "content unavailable" section instead of
  an LLM guess based on its headline -- and an article whose content WAS
  retrieved but whose LLM call failed gets a third, distinct wording that
  says so, rather than being described as title-only (see "Source
  Grounding & Hallucination Prevention")
- Clean Markdown **and** rendered HTML newsletter export
- Simulated sending: saves the newsletter to disk and prints the subject +
  content preview to the console
- Multi-step reasoning: planning → research → ranking → extraction →
  summarization → writing → review → output
- Self-critique with automatic revision -- deterministic structural checks,
  deterministic grounding/hallucination checks, and an LLM editorial
  review, capped at 2 automatic revision rounds
- Fully Autonomous / Human-in-the-Loop toggle, with the export step
  genuinely gated behind human approval in HITL mode
- In HITL mode, a rejected draft can be revised in place using your written
  feedback instead of being discarded and starting over
- Single autonomous entry point: `run_newsletter_agent(goal, mode)`
- A concise console verification summary after every run: total collected
  vs. verified vs. strict/moderate/rejected articles (with rejected and
  selected titles listed), extraction success/failure, LLM-grounded vs.
  fallback summary counts, critic result, and final output paths
- Simple Streamlit frontend
- The LLM provider/model is configurable (see "Model / API Configuration")

## Architecture / Workflow

```
START
  │
  ▼
planner ───────────────► turns the goal into 3-5 search queries
  │
  ▼
research ──────────────► runs each query against a multi-provider news
  │                       search (see "URL Resolution" below), interleaves
  │                       results round-robin across queries, sorts by
  │                       recency, records per-provider diagnostics
  ▼
ranking ───────────────► deterministically scores every candidate for
  │                       genuine AI-agent relevance, rejects off-topic
  │                       articles (gaming, satellites, cars, entertainment,
  │                       generic tech/AI), then selects the final 5-7 with
  │                       publisher-diversity balancing -- extraction and
  │                       LLM calls only ever run on this narrowed set
  │                       (see "Relevance Filtering & Source Diversity")
  ▼
content_enrichment ────► extracts full article text ONLY for the selected
  │                       articles that have a verified publisher URL; a
  │                       wrapper/reference-only URL is never handed to the
  │                       extractor
  ▼
summarize ─────────────► checks content availability BEFORE calling the LLM;
  │                       usable content -> LLM-grounded structured summary;
  │                       unavailable/unresolved -> deterministic fallback,
  │                       LLM is never invoked for that article; if content
  │                       WAS available but the LLM call itself failed, a
  │                       distinct "content available, summarization
  │                       failed" fallback is used instead
  ▼
writer ────────────────► renders each article's Markdown section
  │                       deterministically from the structured summary
  │                       (an ungrounded article's text can never be
  │                       LLM-generated); the Editor's Takeaway is
  │                       LLM-synthesized ONLY from grounded articles, or
  │                       a fixed disclosure paragraph when too few
  │                       articles are grounded
  ▼
critic ◄───────────────┐ structural checks + deterministic grounding/
  │                     │ hallucination checks + an LLM editorial review;
  │                     │ grounding issues always force a revision and cap
  │                     │ the quality score, regardless of the LLM's own opinion
  │ needs_revision      │
  │ (max 2 rounds) ─────► reviser (re-derives grounded summaries more
  │                        conservatively, then re-validates; anything
  │                        still ungrounded is forcibly downgraded to the
  │                        deterministic fallback template)
  │
  ▼ otherwise
END
```

Outside the graph, `app.py` (Streamlit) adds the Human-in-the-Loop
checkpoint: in HITL mode the graph's output is held as a **draft** and is
only exported to disk after you click **Approve**. Clicking **Request
Changes** re-runs the same `reviser_node` the automatic loop uses, driven
by your typed feedback, without re-running research from scratch.

## Fully Autonomous Mode

`run_newsletter_agent(goal, mode="autonomous")` runs the entire pipeline
above end-to-end, including the export/"simulated send" step, with no
pause for human approval. This is the mode used when running the CLI
(`python -m agent.runner`) or selecting **Fully Autonomous** in the
Streamlit sidebar.

## Human-in-the-Loop Mode

`run_newsletter_agent(goal, mode="human")` runs the same pipeline but stops
at a **draft** -- it does not export anything. In the Streamlit app you can
then:

- **Approve and Export Newsletter** -- exports the draft exactly once.
- **Request Changes** -- type feedback and the draft is revised using that
  feedback (via `agent/reviser.py`), without re-running research from
  scratch. Repeat as many times as needed.

Export never happens in HITL mode until you click **Approve**.

## URL Resolution

Article search runs through a **configurable multi-provider strategy**
(`tools/news_search.py`), because Google News RSS's own wrapper links
turned out -- on real, unrestricted networks, not just in a sandboxed
test environment -- to frequently resist resolution by any safe,
non-guessing HTTP/HTML technique. Rather than reverse-engineer Google's
undocumented redirect token (see "Why we did not decode Google's
redirect token" below), the fix was to add providers whose URLs don't
need risky resolution in the first place, and to make the whole search
layer transparent about what actually happened.

**Providers, tried in this order by default** (override with the
`NEWS_SEARCH_PROVIDERS` environment variable, e.g.
`NEWS_SEARCH_PROVIDERS=direct_rss,bing_news` to skip Google entirely):

1. **`direct_rss`** -- fetches a curated list of publisher RSS feeds
   (TechCrunch, VentureBeat, The Verge, Wired, Ars Technica, Engadget,
   ZDNET, MIT Technology Review, plus official engineering/research
   blogs -- Microsoft, Google AI, OpenAI, Anthropic, AWS ML) directly
   and keyword-filters their entries against the query. Every URL here
   is the publisher's own `<link>` -- **verified-direct by
   construction**, with no wrapper to resolve at all. This is the most
   reliable provider whenever a feed covers the query's topic, and
   needs no API key. Official-blog feed URLs are best-effort: companies
   restructure their blogs and RSS availability changes over time, so a
   stale URL just fails that one feed fetch (logged, then skipped) --
   run `python -m tools.diagnostics` to see which ones actually resolve
   right now.
2. **`google_news`** -- Google News RSS search. For each entry, tries,
   in order: (a) a direct link if the feed already gives one, (b)
   following real HTTP redirects with a realistic User-Agent, then
   parsing the interstitial page's own `<link rel="canonical">` /
   `<meta http-equiv="refresh">` tag if Google's own page is still what
   comes back, (c) a DuckDuckGo secondary search by title + source. This
   project does **not** decode Google's opaque redirect token -- see
   below.
3. **`bing_news`** -- Bing News RSS search (`bing.com/news/search?...
   &format=RSS`, no API key required). Bing's own click-tracking wrapper
   exposes the real destination as a **plain, human-readable `url=`
   query parameter** -- reading that is parsing a literal value Bing
   itself put there, not decoding an opaque token, so it's read directly
   when present; otherwise the same HTTP-redirect strategy used for
   Google is applied.

Every candidate from every provider is validated before acceptance
(`validate_candidate_url`): it must be a well-formed `http(s)` URL, it
must not still be a Google/Bing/search-engine domain, and results are
deduplicated by URL across providers. RSS-provided `source.href`
metadata is captured too, but only ever used as a soft candidate hint --
never presented as a verified article URL by itself. Each article
carries a `provider` field recording which provider found it.

If nothing validates for a given article, its original RSS/wrapper URL
is preserved as a **reference link only** (`url_resolution_status =
"unresolved"`), and the newsletter labels it accordingly ("Google News
reference" / "Bing News reference") rather than "Verified publisher
article". **If every configured provider fails or is network-blocked,
`news_search_tool` returns an empty list** -- it never fabricates
results -- and the rest of the pipeline (grounding / deterministic
fallback, see below) is what keeps the newsletter honest and complete in
that case.

### Why we did not decode Google's redirect token

Google's `news.google.com/rss/articles/<token>` links encode the
destination in an opaque, undocumented token that (as far as could be
verified without Google's private "batchexecute" internal API) requires
either executing the page's JavaScript or reverse-engineering an
unstable, unofficial endpoint to decode -- with no way to *verify from
outside Google* that a decoded value is actually correct rather than a
plausible-looking guess. Implementing that was deliberately rejected:
an incorrectly "resolved" URL that looks verified but isn't would be a
subtler violation of this project's "never fabricate a URL" rule than
an honest, clearly-labeled resolution failure. The `direct_rss` and
`bing_news` providers exist specifically to route around this problem
instead of guessing at it.

### Diagnosing URL resolution on your own network

Because a fully offline/sandboxed development environment has its own
outbound network fully blocked (every endpoint below returns
`403`/`ProxyError` from it), URL resolution should be verified on a
real, unrestricted network. Run:

```bash
python -m tools.diagnostics
python -m tools.diagnostics "your custom query here"
```

This prints, with **no mocking and no code changes**: the complete raw
structure of one live Google News RSS entry, raw HTTP probes of each
provider's endpoint (status code / final URL / response size, or the
exact exception if blocked), each provider's individual result counts,
and the combined `news_search_tool()` result exactly as
`agent/research.py` would receive it -- so a network or provider block
can be *proven*, not assumed, and so this README never has to claim
"fixed" without live evidence.

## Relevance Filtering & Source Diversity

Before any extraction or LLM call happens, `agent/ranking.py` (the
`ranking` node) narrows the full candidate pool down to the articles
that are actually going into the newsletter. This is a **two-tier**
system (`agent/relevance.py`, no LLM call involved in scoring):

- **Three phrase tiers, not one**: `_STRONG_AGENT_PHRASES` (explicit
  multi-word AI-agent phrases -- "ai agent", "agentic ai",
  "multi-agent system", "agent orchestration", "coding agent",
  "computer-use agent", "agent safety", "mcp server", etc.) score
  highest; `_RELATED_CAPABILITY_PHRASES` (broader agent-adjacent
  capability language that never needs the literal word "agent" --
  "autonomous workflow", "ai orchestration", "multi-step planning",
  "tool calling", "ai copilot", "workflow automation", etc.) score a
  moderate amount; bare `_WEAK_AGENT_TERMS` ("agent", "autonomous",
  "copilot" on their own) score the least. A fixed list of off-topic
  subject markers (gaming, consumer hardware, satellites, automobiles,
  entertainment, sports) still subtracts from the score regardless of
  tier, so a gaming article that happens to mention "AI" is still
  rejected. Scoring looks at **title, RSS summary, source name, and
  extracted content** where available -- title matches are weighted
  most heavily.
- **Query-aware bonus, gated on genuine baseline signal**: each article
  is tagged with the specific research query that surfaced it
  (`matched_query`, set in `agent/research.py`); `score_article_relevance()`
  adds a bonus (capped at +3, word-boundary keyword matching) when the
  article's own text overlaps with that query's keywords -- but **only
  if the article already has a positive score from the strong/related-
  capability/weak phrase tables on its own**. This lets an article score
  adequately for being clearly *on-topic for the specific angle the
  planner asked about*, without ever letting the bonus manufacture
  relevance out of a purely incidental word match (see the regression
  note below).
- **Two selection thresholds, not one lowered threshold.** A first,
  strict threshold (`RELEVANCE_STRICT_THRESHOLD = 3.0`) is checked
  first. Only if fewer than 5 articles clear it
  (`MIN_STRICT_BEFORE_MODERATE_TOPUP`) does the pipeline fall back to a
  second, clearly-separate moderate threshold (`RELEVANCE_MODERATE_THRESHOLD
  = 1.0`) to top up the remaining slots with agent-*related* (not
  agent-*unambiguous*) articles. The strict bar itself is never
  lowered; a second, explicitly-labeled tier was added instead. Which
  tier(s) contributed is recorded verbatim in `research_stats["selection_tier"]`
  ("strict" or "strict+moderate") and surfaced in the CLI/Streamlit
  output, so a reader can always tell whether every selected article
  met the strict bar or some were moderate top-ups.
- **A hard relevance floor either way**: articles that clear neither
  threshold are rejected outright, never used as filler. If a research
  run only turns up 3 relevant articles total (strict + moderate
  combined), the newsletter has 3 articles -- it is never padded with
  off-topic ones to hit a target count.
- **Publisher diversity, computed per tier**: strict-tier and
  moderate-tier articles are each round-robin interleaved across
  publishers (`diversify_by_publisher`) before being combined, so one
  prolific source (e.g. a single company blog) can't fill the entire
  newsletter -- and can't crowd out a moderate-tier top-up from a
  different publisher either -- when other relevant sources are
  available. `research_stats["selected_publisher_count"]` reports how
  many distinct publishers made the final cut.
- **Transparent diagnostics**: `research_stats` (surfaced in the CLI's
  verification summary and the Streamlit execution summary) reports,
  per provider, how many results it returned, how many resolved to a
  verified direct URL, how many were **strict**-relevant, how many were
  **moderate**-relevant, and how many were actually selected; it also
  now records the rejected articles' titles so a reviewer can confirm a
  specific known-off-topic article was excluded.
- **Regression note -- the query bonus cannot manufacture relevance from
  nothing.** A real run surfaced this the hard way: a "10 Best Standing
  Desks" buying guide and a story about bypassing Claude's safety
  guardrails for bioweapons research both scored as "moderate" relevant
  to an AI-agent newsletter, purely because their text happened to share
  a generic word with the research query that found them ("best" from
  the desk guide's own title, "building" from "building Lego sets";
  "research" from the bioweapons headline) -- neither article contained
  a single AI/agent-related word anywhere. This is now fixed as
  described above; see
  `test_query_bonus_never_applies_without_any_baseline_agent_signal` in
  `tests/test_relevance.py` for the exact regression case, reproduced
  from the real article text.

## Reducing LLM (Groq) Token Usage

Several changes specifically target Groq's daily token limit:

- **Summarization only ever runs on the final selected set.** Because
  relevance filtering + diversity selection (above) happens *before*
  extraction and summarization, the LLM is never invoked for an article
  that isn't going into the newsletter.
- **Extracted content sent to the summarizer is truncated** to 6,000
  characters (`tools/summarizer.py`) -- enough for the article's actual
  substance, without paying for tens of thousands of characters of
  boilerplate/footer text some pages include.
- **Output token caps are set per call site** instead of one large
  global default: the planner's query-list JSON needs very little
  (700), the per-article summary JSON needs a moderate amount (2,000),
  the Editor's Takeaway is one short paragraph (400), and the critic's
  JSON verdict needs a few hundred (1,200) -- see `tools/llm_utils.get_llm(max_tokens=...)`.
  Every prompt also explicitly instructs the model to return *only* the
  JSON object with no preamble or reasoning text.
- **The critic never sees the full research corpus** -- only the small,
  already-selected `ranked_articles` list (<= 7 articles), which stays
  roughly constant regardless of how many articles were originally
  researched.
- **Every LLM call site is wrapped so a Groq failure can never crash
  the pipeline.** A rate-limit / daily-token-limit error
  (`tools.llm_utils.is_rate_limit_error` detects the common
  429/quota/rate-limit markers) is logged with a clear, specific message
  and falls back to a safe deterministic path. A rate-limited/failed
  call is always counted as `error_fallback`, never as a successful
  LLM-grounded summary -- see "Model / API Configuration" below for what
  this means in practice.

## Planner Date Awareness

`agent/planner.py` passes the real current date (`datetime.now(timezone.utc)`)
into its prompt and explicitly instructs the LLM not to hardcode or
assume a specific past year -- without this, an LLM's own training-data
knowledge cutoff tends to bias it toward referencing a stale year (e.g.
"2024 AI agent developments") regardless of when the agent is actually
run. Generated queries use "latest" / "recent" framing anchored to that
real date instead.

## Robust JSON Parsing From LLM Responses

Every node that expects structured output from Groq (summarizer, writer,
critic) parses it with `agent/grounding.extract_json_object()`, which
tries, in order: fence-stripping anywhere in the text, a direct
`json.loads`, a balanced-brace scanner that tracks nesting depth and
string-literal state (`_find_balanced_json_objects`), and a purely
mechanical truncation repair (`_attempt_close_truncated_json`) that closes
an unclosed string, trims a dangling trailing comma, and appends the
exact closing brackets/braces needed -- it can only ever complete JSON the
model already started writing, never invent a field or value. Only if all
of that fails does it raise `ValueError`, at which point the calling node
falls back to its existing deterministic path. See `tests/test_grounding.py`
for the fenced/truncated/multi-fragment cases this covers.

## Article Extraction Fallback Tiers

`tools/article_extractor.py` fetches a verified direct article URL and
extracts its readable text with `trafilatura`, trying four graduated
tiers and using whichever first produces at least `MIN_CONTENT_LENGTH`
(200) characters of real text, never fabricating content if all of them
come up short:

1. **Trafilatura, precision mode** (`favor_precision=True`).
2. **Trafilatura, recall mode** (`favor_precision=False, favor_recall=True`)
   -- retried only if tier 1 produced too little text; recovers
   AWS-blog-style templates with heavy sidebar/related-post boilerplate.
3. **Common content-container selectors** (`<article>`, `<main>`,
   `[role="main"]`, `#content`, `.post-content`, `.entry-content`, etc.,
   via BeautifulSoup).
4. **Full-page strip** (last resort).

The bot-challenge/consent-page detector (`_looks_blocked`) also catches
common WAF/CDN challenge-page markers ("request unsuccessful",
"reference id:", "checking your browser", "verify you are a human",
"unusual traffic", "attention required") in addition to the existing
paywall/cookie-consent markers.

## Source Grounding & Hallucination Prevention

This is the most important guarantee in the codebase, enforced
**structurally**, not just by prompting:

- `agent/grounding.py` defines `needs_fallback(article)`: an article is
  only trusted with an LLM summarization call if it has a **verified
  direct publisher URL**, extraction actually **succeeded**, and the
  extracted text is **at least 200 characters**. Anything else -- missing
  content, empty content, a failed/skipped extraction, or a still-
  unresolved wrapper URL -- goes straight to a fixed, deterministic
  fallback and the LLM is **never called** for that article.
- **Three distinct, honestly-worded outcomes**, not two:
  1. **Content unavailable** (extraction failed, page blocked, title-only,
     or URL never resolved) -- `build_fallback_summary()`:

     ```
     What happened: No details were provided in the source beyond the title.
     Key takeaways: No key takeaways available because the article content could not be retrieved.
     Why it matters: The title suggests a potentially relevant topic, but its significance cannot be verified without the article text.
     Source limitations: The article text was unavailable; no factual claims were inferred from the title.
     ```

  2. **Content available, but LLM summarization failed** (e.g. a Groq
     rate limit/quota error, a malformed/unparseable model response) --
     `build_llm_failed_summary()`. This is deliberately worded
     differently: it says the article's text **was** retrieved and only
     the automated summarization step failed, with `content_available:
     True`. It never claims "only the title was available" when that
     isn't true.
  3. **LLM-grounded** -- content was available and the LLM produced a
     structured, source-grounded summary.
- `agent/writer.py` renders every article's Markdown section
  **deterministically in code** from these structured fields, choosing
  among all three wordings based on the article's actual, factual
  extraction state (`content_available` + `grounding_status`) -- never
  from LLM-produced wording. An ungrounded article's section text is
  never passed through an LLM at all, so it cannot be embellished.
- The **Editor's Takeaway** is LLM-synthesized only from the subset of
  articles that are actually grounded, and only when at least half the
  selected articles are grounded. Otherwise it deterministically falls
  back to a paragraph that discloses the limitation instead of
  generalizing a trend from title-only headlines.
- `agent/critic.py` runs deterministic grounding checks
  (`validate_newsletter_grounding`) alongside its LLM review: it flags
  factual-claim language ("announced", "released", "discovered", etc.)
  appearing in a section for an article marked content-unavailable,
  mislabeled source lines, and unsupported Editor's Takeaway
  generalizations. These deterministic findings always force a revision
  and cap the quality score, **regardless of what the LLM critic itself
  concludes**.
- `agent/reviser.py` re-derives (not just rewords) any flagged grounded
  article's summary more conservatively from its original source text,
  rebuilds the newsletter, and re-validates; anything still flagged after
  that is forcibly downgraded to the deterministic fallback template
  before the graph can terminate.

## Tools Used

1. **News Search Tool** (`tools/news_search.py`) -- searches a configurable,
   ordered set of providers (`direct_rss`, `google_news`, `bing_news`;
   see `NEWS_SEARCH_PROVIDERS`) and resolves each article's real publisher
   URL through multiple validated strategies, never fabricating one.
2. **Article Extraction Tool** (`tools/article_extractor.py`) -- fetches and
   extracts readable article text from verified URLs only, with structured
   success/failure status.
3. **LLM Summarization / Writing / Critique Tool** (`tools/summarizer.py`,
   `tools/llm_utils.py`) -- Groq-backed LLM calls, invoked only when there
   is real content to ground them in.
4. **Newsletter Export Tool** (`agent/exporter.py`) -- renders the Markdown
   newsletter to HTML and saves both formats to disk (the simulated send).

## Technologies Used

Python 3.10+, LangGraph, LangChain, `langchain-groq` (Groq LLM; default
model `openai/gpt-oss-20b`, configurable via `GROQ_MODEL` -- see "Model /
API Configuration"), Streamlit, `feedparser`, `httpx`, `beautifulsoup4`,
`trafilatura`, `markdown`, `python-dotenv`, `pytest`.

## Project Structure

```
Assignment_Folder/
├── app.py                     # Streamlit frontend (goal input, mode toggle, HITL approval)
├── agent/
│   ├── state.py                # NewsletterState / Article TypedDicts
│   ├── grounding.py              # deterministic fallback text (both variants) + hallucination validators
│   ├── planner.py                 # goal -> search queries
│   ├── research.py                 # runs queries, dedupes, interleaves, sorts by recency
│   ├── relevance.py                  # deterministic AI-agent relevance scoring + publisher diversity
│   ├── ranking.py                     # LangGraph node: applies relevance.py, selects final 5-7 articles
│   ├── content.py                   # gates extraction to verified URLs only
│   ├── summarize.py                  # gates LLM summarization on content availability
│   ├── writer.py                      # deterministic article rendering + grounded Editor's Takeaway
│   ├── critic.py                       # structural + grounding + LLM self-review
│   ├── reviser.py                       # re-derives / downgrades ungrounded content
│   ├── exporter.py                       # Markdown -> HTML rendering + file export ("simulated send")
│   ├── graph.py                           # builds/compiles the LangGraph StateGraph
│   └── runner.py                           # run_newsletter_agent(goal, mode) entry point + verification summary
├── tools/
│   ├── news_search.py            # News Search tool: multi-provider search + URL resolution
│   ├── diagnostics.py             # standalone, no-mocking network/provider diagnostics (python -m tools.diagnostics)
│   ├── article_extractor.py      # Article Extraction tool: structured status, never fakes content
│   ├── summarizer.py              # LLM summarization tool: structured, source-grounded output
│   └── llm_utils.py                # shared ChatGroq client + response parsing (model configurable)
├── outputs/                    # exported newsletter_<timestamp>.md / .html files
│   ├── newsletter_final_demo.md      # curated final submission demo (committed)
│   └── newsletter_final_demo.html    # curated final submission demo (committed)
├── tests/                      # pytest suite (mocked, no live network/LLM calls)
├── requirements.txt
├── .env.example                 # placeholder env vars -- copy to .env and fill in
├── .env                          # your real GROQ_API_KEY (gitignored, not committed)
├── Procfile                       # optional: start command for Heroku/Render/Railway-style hosts
└── README.md
```

## Environment Variables

| Variable        | Required | Default              | Purpose                                   |
|-----------------|----------|-----------------------|--------------------------------------------|
| `GROQ_API_KEY`  | Yes      | --                    | Groq API key used for all LLM calls        |
| `GROQ_MODEL`    | No       | `openai/gpt-oss-20b`  | Overrides the Groq model (`tools/llm_utils.py`) |
| `NEWS_SEARCH_PROVIDERS` | No | `direct_rss,google_news,bing_news` | Ordered provider list (see "URL Resolution") |

Copy `.env.example` to `.env` and fill in real values -- `.env` is
gitignored and must never be committed:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY
```

Get a free Groq API key at https://console.groq.com/.

## Installation

```bash
git clone https://github.com/Divyansh-git10/Autonomous-Newsletter-Agent.git
cd Autonomous-Newsletter-Agent

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then edit .env with your real GROQ_API_KEY
```

## Running Locally (CLI)

```bash
python -m agent.runner
```

Runs `run_newsletter_agent(goal, mode="autonomous")`, prints a full
console log (research → extraction → summarization → critique →
verification summary), and exports the newsletter to `outputs/`.

### Example usage of the main agent function

```python
from agent.runner import run_newsletter_agent

result = run_newsletter_agent(
    goal="Create a weekly newsletter on the latest AI agent news and send it to our subscribers.",
    mode="autonomous",   # or "human"
)

print(result["final_output"])       # the Markdown newsletter
print(result["output_path"])        # where it was saved (autonomous mode)
print(result["critique"]["quality_score"])
```

## Running the Frontend (Streamlit)

```bash
streamlit run app.py
```

1. Enter a plain-English goal (a sensible default is pre-filled).
2. Choose **Fully Autonomous** or **Human-in-the-Loop** in the sidebar
   (see "Fully Autonomous Mode" / "Human-in-the-Loop Mode" above).
3. Click **Run Newsletter Agent**.

## Testing

```bash
pytest tests/ -v
```

135 tests, fully offline (LLM and network calls mocked), covering:

- module imports
- URL resolution (`tests/test_url_resolution.py`): wrapper detection,
  candidate validation, redirect resolution, secondary-search fallback,
  all-strategies-fail safety, invalid-candidate rejection, multi-provider
  orchestration, provider configuration via `NEWS_SEARCH_PROVIDERS`
- relevance scoring (`tests/test_relevance.py`): strong/related/weak
  phrase tiers, off-topic rejection, query-match bonus (including the
  standing-desk/bioweapons regression case), source-name scoring,
  publisher-diversity interleaving, strict-vs-moderate two-tier selection
- the ranking node (`tests/test_ranking.py`): end-to-end filtering,
  target-count capping, `research_stats` diagnostics, moderate-tier
  top-up labeling, publisher-diversity outcomes
- planner date-awareness (`tests/test_planner.py`)
- robust JSON parsing (`tests/test_grounding.py`)
- extraction (`tests/test_extraction.py`): direct success, wrapper URLs
  never reaching the extractor, HTTP errors, timeouts, empty extraction,
  recall-mode fallback, content-container fallback, WAF/bot-challenge
  detection
- summarization (`tests/test_summarize.py`): content-available invokes
  the LLM, content-unavailable bypasses it entirely, rate-limit and
  generic LLM failures fall back with the correct "content available,
  summarization failed" wording (not counted as grounded)
- writing (`tests/test_writer.py`): all three deterministic wordings
  (no content / content-available-but-LLM-failed / grounded), correct
  source labeling, Editor's Takeaway grounding threshold
- critic (`tests/test_critic.py`): unsupported claims and mislabeled
  sources trigger revision, clean grounded newsletters pass
- the end-to-end graph (`tests/test_agent.py`): both modes complete,
  export-before-approval is impossible, revision count always advances
  and the graph cannot loop forever, final output is always populated,
  files are exported exactly once, irrelevant articles are filtered out
  end-to-end
- export rendering (`tests/test_exporter.py`): real HTML, not
  `<pre>`-wrapped Markdown
- LLM configuration (`tests/test_tools.py`): `GROQ_MODEL` env var override
  is honored, default model used when unset

Additional manual checks performed before submission (not part of the
automated suite, since they need real network/API access this project's
own offline dev environment does not have): a Streamlit frontend startup
smoke test, and a full live end-to-end run against real RSS feeds and a
real Groq key (see `outputs/newsletter_final_demo.md` for its curated
result).

## Newsletter Output Location

```
outputs/newsletter_YYYYMMDD_HHMMSS.md      # every autonomous/approved run
outputs/newsletter_YYYYMMDD_HHMMSS.html
outputs/newsletter_final_demo.md            # curated final submission demo (see below)
outputs/newsletter_final_demo.html
```

`newsletter_final_demo.md`/`.html` is committed to the repository as
submission evidence: a 7-article newsletter combining genuinely relevant,
real AI-agent articles collected across this project's own successful
live runs (real titles, publishers, and URLs -- no invented data), built
by feeding those real articles through the actual, unmodified
`agent.writer.write_newsletter()` renderer. Two articles that a real run's
relevance scorer briefly mis-scored during development (a standing-desks
buying guide and a Claude-misuse/bioweapons-safety story -- see the
regression note in "Relevance Filtering & Source Diversity") are
deliberately excluded, since neither is genuine AI-agent news.

## Deployment

This app is a single Streamlit process -- `app.py` calls
`run_newsletter_agent()` in-process, so there is no separate backend/API
and therefore no CORS or API-base-URL configuration to manage.

**Deployed on Render** (the platform actually used for the live demo
above): a Render **Web Service** connected directly to this GitHub
repository's `main` branch, using:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
- **Environment variable:** `GROQ_API_KEY` set in the Render dashboard's
  Environment tab (never committed -- see "Environment Variables" above)

Auto-deploy is enabled, so every push to `main` redeploys the live demo.
To reproduce this deployment yourself:

1. Push this repository to GitHub (see "GitHub Repository" above).
2. On https://dashboard.render.com, create a new **Web Service** pointing
   at your fork/clone, runtime **Python**.
3. Set the Build Command and Start Command exactly as shown above (a
   matching `Procfile` is also included in the repo for hosts that read
   it directly, e.g. Railway/Heroku-style platforms).
4. Add `GROQ_API_KEY` (and optionally `GROQ_MODEL`) under **Environment**.
5. Deploy.

**Alternative -- Streamlit Community Cloud (no code changes, no Procfile
needed):** push to GitHub, create a new app at https://share.streamlit.io
pointing at this repo and `app.py`, and add `GROQ_API_KEY` under the
app's **Secrets**.

**Hosted demo URL:** https://autonomous-newsletter-agent.onrender.com

## Known Limitations

- **Secondary URL search can be blocked.** DuckDuckGo's HTML search page
  can rate-limit or block automated requests (observed: HTTP 403). When
  that happens and redirect resolution also fails, the article is kept as
  an honest "Google News reference" rather than a fabricated direct link.
- **`direct_rss` only covers its curated feed list.** If a query's topic
  isn't covered by any of the curated publishers, that provider
  contributes nothing for that query and the pipeline falls through to
  `google_news` / `bing_news` (or an empty result, handled safely, if
  those are also blocked or unresolved).
- **Google's redirect token is intentionally not decoded** -- see "Why we
  did not decode Google's redirect token" above. This is a deliberate
  scope decision, not an oversight.
- **Official-blog RSS URLs may go stale.** Company blog platforms change
  over time; a feed that 404s is skipped gracefully (never crashes the
  provider), but its articles simply won't appear until the URL is
  updated in `tools/news_search.py`.
- **Relevance scoring is keyword/phrase-based, not semantic.** It will
  miss an AI-agent story that happens to avoid all of the scored
  phrases, and (rarely) could accept a false positive that mentions
  agent-related terms without actually being about AI agents. It is
  deliberately conservative (reject rather than pad with off-topic
  articles) and fully deterministic/inspectable.
- **Publisher diversity only helps when multiple relevant sources exist.**
  If only one publisher's feed actually covers a niche query topic, the
  final selection will still be dominated by that one source.
- **Relevance/diversity selection tops up with a second, moderate tier
  only when the strict tier is scarce (< 5 candidates)** -- a research
  run that turns up, say, 4 strict-tier and 0 moderate-tier articles
  will ship a 4-article newsletter rather than inventing a 5th.
- **Exact-URL deduplication only** -- the same story via two differently
  tokenized Google News links won't be detected as a duplicate.
- **No persistence between runs** -- each run is independent in-memory
  state.
- **Simulated sending only** -- no real SMTP/email integration, per the
  assignment's own "save as file or print" option.
- **Groq's free-tier daily token limit (TPD) can be exhausted during
  real use.** This was observed directly during development/testing: a
  429/quota error mid-run. It is **not a pipeline crash** -- every LLM
  call site catches this and falls back to the deterministic
  "content available, summarization failed" wording (see "Source
  Grounding & Hallucination Prevention"), so the run still completes and
  still produces a newsletter, just with fewer LLM-grounded summaries
  than a run with quota headroom. **Evaluation or production use should
  use a Groq API key/plan with sufficient token quota** for the number
  and length of articles being summarized.
- **Render's free tier spins the service down after inactivity** --
  the first request after a period of no traffic can take up to ~30-60
  seconds while the instance wakes back up; subsequent requests are fast.

## Model / API Configuration

- The LLM **provider** wired into this codebase is Groq
  (`langchain-groq`, `tools/llm_utils.get_llm()`). The **model** is
  configurable via the `GROQ_MODEL` environment variable (default
  `openai/gpt-oss-20b`); this is a genuine, tested override (see
  `tests/test_tools.py`), not just documentation.
- Swapping the LLM **provider** entirely (e.g. to OpenAI, Anthropic, or
  Gemini) is possible but requires a small code change: everything in
  this codebase calls the single `get_llm()` function in
  `tools/llm_utils.py`, so only that one function needs to change to
  return a different LangChain chat model -- it is not a runtime/env-var
  provider switch today.
- **Not all summaries in a given run are necessarily LLM-generated.**
  This README does not claim that. A summary is only ever LLM-generated
  ("grounded") when real article content was extracted **and** the Groq
  call succeeded. When content extraction fails, or the content was
  never resolved, the summary is a fixed deterministic fallback. When
  content extraction succeeds but the Groq call itself fails (commonly:
  rate limit/daily token quota), the summary is a different, honestly
  worded deterministic fallback that says the content was available but
  summarization could not be completed. The console verification summary
  and `research_stats`/`summary_stats` always report exactly how many
  summaries were LLM-grounded vs. each type of fallback for a given run
  -- see "Source Grounding & Hallucination Prevention" above.
- For evaluation or production use, use a Groq API key/plan with
  sufficient token quota for the number and length of articles being
  summarized, to maximize the number of LLM-grounded (rather than
  fallback) summaries.

## Future Improvements

- Additional `direct_rss` publisher feeds, or a topic-to-feed mapping,
  to widen coverage beyond the currently curated list.
- A paid/keyed news API (e.g. NewsAPI, GNews) as an additional provider
  for topics none of the current free providers cover well.
- An embedding-based relevance classifier as a second-opinion signal
  alongside (not instead of) the current deterministic keyword scoring.
- Fuzzy duplicate detection (title/embedding similarity) across articles
  from different queries.
- Move the Human-in-the-Loop checkpoint inside the LangGraph itself using
  `langgraph.types.interrupt()` + a checkpointer, instead of gating export
  at the Streamlit layer.
- Persist run history to a small local database.
- Real email delivery (SMTP) as an alternative to the simulated send.
- Upgrade off Render's free tier (or add a keep-alive ping) to avoid the
  cold-start delay noted in "Known Limitations".
