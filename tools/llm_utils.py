import os
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Default output cap for any call site that doesn't specify its own.
# Individual call sites request a much smaller cap that matches what
# they actually need (see agent/planner.py, tools/summarizer.py,
# agent/writer.py, agent/critic.py) -- this default is only a safety
# ceiling, not a target.
DEFAULT_MAX_TOKENS = 2000

# The model is configurable via the GROQ_MODEL env var (see .env.example);
# this is the default used when it's unset. Swapping the LLM PROVIDER
# entirely (e.g. away from Groq) requires editing this one function --
# it is not a runtime/env-var-driven provider switch, since only Groq
# is wired up elsewhere in the pipeline.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

# Substrings that reliably show up in a Groq/OpenAI-compatible
# rate-limit or token/quota-exhaustion error, across both the specific
# exception types some SDK versions raise and the plain string message
# other versions surface instead. Used only for clearer diagnostic
# logging -- callers still fall back to the same safe, non-hallucinating
# deterministic path either way (see tools/summarizer.py, agent/critic.py,
# agent/writer.py, agent/planner.py).
_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "quota",
    "token limit",
    "tokens per day",
    "tpd",
    "tokens per minute",
    "tpm",
)


def get_llm(max_tokens: int = DEFAULT_MAX_TOKENS) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in the .env file."
        )

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0,
        max_tokens=max_tokens,
    )


def is_rate_limit_error(exc: Exception) -> bool:
    """
    Best-effort detection of a Groq rate-limit / daily-token-limit
    failure, so callers can log a clear, specific message instead of a
    generic "LLM call failed" -- without changing the safe behavior
    (deterministic fallback) that already applies to any LLM failure.
    """

    message = str(exc).lower()
    type_name = type(exc).__name__.lower()
    return any(marker in message or marker in type_name for marker in _RATE_LIMIT_MARKERS)


def response_to_text(response: Any) -> str:
    content = getattr(
        response,
        "content",
        response,
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if text:
                    text_parts.append(
                        str(text)
                    )

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()