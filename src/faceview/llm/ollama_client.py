"""Ollama backend — local LLM fallback when no Anthropic key is set.

[Ollama](https://ollama.com) runs LLMs locally and exposes them on
``http://127.0.0.1:11434`` by default. This module provides a
streaming engine compatible with our ``EchoEngine`` /
``AnthropicEngine`` interface so the existing ``ClaudeClient``
worker thread can use it without changes.

Auto-fallback chain (set up in `claude_client.ClaudeClient`):

    1. ``ANTHROPIC_API_KEY`` set → Anthropic
    2. Ollama reachable on localhost:11434 → Ollama (default model
       picked from ``FACEVIEW_OLLAMA_MODEL`` env var or first
       installed model)
    3. Otherwise → :class:`EchoEngine` demo

No new runtime dependency — pure stdlib `urllib` + `json`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterator, Optional

from faceview.core.logger import get_logger


log = get_logger("ollama")


DEFAULT_HOST = "http://127.0.0.1:11434"


def is_ollama_available(host: str = DEFAULT_HOST, timeout: float = 0.5) -> bool:
    """Quick reachability check for a running ``ollama serve``."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def list_ollama_models(host: str = DEFAULT_HOST, timeout: float = 1.0) -> list[str]:
    """List installed Ollama models. Returns ``[]`` on any failure."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def pick_default_model(host: str = DEFAULT_HOST) -> Optional[str]:
    """Pick a sensible default model — env-var override, then first installed.

    Prefers text-only chat models (skips ``-vision`` / ``llava`` variants
    whose chat endpoint expects multimodal input we don't supply).
    """
    env = os.environ.get("FACEVIEW_OLLAMA_MODEL")
    if env:
        return env
    models = list_ollama_models(host)
    if not models:
        return None
    # Prefer text-only chat-tuned models. Skip vision / llava variants.
    candidates = [m for m in models
                  if "vision" not in m.lower() and "llava" not in m.lower()]
    if not candidates:
        candidates = models
    for needle in ("llama3", "llama2", "qwen", "mistral", "phi", "gemma"):
        for m in candidates:
            if needle in m.lower():
                return m
    return candidates[0]


class OllamaEngine:
    """Streaming Ollama engine matching the ClaudeClient engine protocol."""

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout

    def stream_reply(self, conv, user_text: str) -> Iterator[str]:
        """Yield text chunks from Ollama's streaming /api/chat endpoint."""
        # Convert our Conversation history into Ollama's chat format.
        messages: list[dict] = []
        sys_fn = getattr(conv, "effective_system", None)
        sys_text = sys_fn() if callable(sys_fn) else getattr(conv, "system", None)
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        # Conversation.messages() is a method that returns a list copy.
        msg_list = conv.messages() if callable(getattr(conv, "messages", None)) \
                   else getattr(conv, "_messages", [])
        for m in msg_list:
            messages.append({
                "role": "user" if m.role == "user" else "assistant",
                "content": m.content,
            })
        # Caller already appended the user message via add_user().
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": 512},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if "message" in chunk and "content" in chunk["message"]:
                    text = chunk["message"]["content"]
                    if text:
                        yield text
                if chunk.get("done"):
                    break
