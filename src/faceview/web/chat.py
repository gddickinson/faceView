"""Local-only chat for the web interface: one Claude reply per request, in a character's voice.

The API key is read from the environment by the Anthropic SDK on the server (ANTHROPIC_API_KEY, as the desktop app
does) and is never sent to the browser. The model is the desktop app's setting (FACEVIEW_MODEL, see config.py).
Without a key the endpoint answers 503 and the page explains how to enable it. Nothing here runs on the static site,
in tests or in CI.
"""
from __future__ import annotations

import os

from faceview.config import settings
from faceview.llm.character import character_for, load_character_registry
from faceview.llm.conversation import Conversation

SPOKEN_RULES = (
    "\n\nYou are talking face to face through an animated avatar in the faceView web interface; your reply is "
    "lip-synced on screen. Keep every reply to 1-3 short sentences of plain speech: no lists, headings or markdown."
)
MAX_TURNS = 20
MAX_CHARS = 2000


class ChatUnavailable(RuntimeError):
    """No credentials or SDK: the page shows how to enable chat."""


def status() -> dict:
    return {"available": bool(os.environ.get("ANTHROPIC_API_KEY")), "model": settings.anthropic_model,
            "characters": sorted(load_character_registry())}


def build_conversation(persona: str, history: list[dict]) -> Conversation:
    """Repo Conversation with the character sheet as system prompt and the (trimmed, validated) history."""
    conv = Conversation(system=character_for(persona).narrate_identity() + SPOKEN_RULES)
    turns = list(history[-MAX_TURNS:])
    while turns and turns[0].get("role") != "user":  # the API needs a user turn first
        turns.pop(0)
    for turn in turns:
        text = str(turn.get("content", ""))[:MAX_CHARS]
        if turn.get("role") == "user":
            conv.add_user(text)
        elif turn.get("role") == "assistant":
            conv.add_assistant(text)
    return conv


def reply(persona: str, history: list[dict]) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ChatUnavailable("ANTHROPIC_API_KEY is not set for the web server.")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise ChatUnavailable("The anthropic package is not installed (pip install anthropic).") from exc
    conv = build_conversation(persona, history)
    messages = conv.for_anthropic()
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("history must end with a user message")
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=16000,  # room for adaptive thinking; length is set by the prompt
            system=conv.effective_system(),
            messages=messages,
        )
    except anthropic.AuthenticationError as exc:
        raise ChatUnavailable("The API key was rejected.") from exc
    except anthropic.NotFoundError as exc:
        raise ChatUnavailable(f"Model {settings.anthropic_model!r} was not found (set FACEVIEW_MODEL).") from exc
    except anthropic.RateLimitError as exc:
        raise ChatUnavailable("Rate limited by the API; try again shortly.") from exc
    except anthropic.APIStatusError as exc:
        raise ChatUnavailable(f"API error {exc.status_code}.") from exc
    except anthropic.APIConnectionError as exc:
        raise ChatUnavailable("Could not reach the Anthropic API.") from exc
    if resp.stop_reason == "refusal":
        return {"reply": "", "refused": True, "model": resp.model}
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return {"reply": text, "refused": False, "model": resp.model}
