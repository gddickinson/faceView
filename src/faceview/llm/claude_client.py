"""Anthropic Claude client with a demo-mode fallback.

If ``ANTHROPIC_API_KEY`` is set, real streaming responses are emitted to the
event bus. If not, an ``EchoEngine`` returns a friendly stubbed reply so the
GUI is fully usable without a key.

The send loop runs on a single background thread per client (kept alive
between messages) — the Anthropic SDK's streaming context manager is
synchronous, so a thread is the right shape.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from faceview.config import settings
from faceview.core.event_bus import get_bus
from faceview.core.events import ChatMessage, EventType
from faceview.core.logger import get_logger
from faceview.llm.conversation import Conversation


log = get_logger("claude")


# ── Demo engine ──────────────────────────────────────────────────────────


class EchoEngine:
    """Fallback used when no API key is set; produces a deterministic reply."""

    def stream_reply(self, conv: Conversation, user_text: str):
        # Simulate streaming so the UI animation looks the same as real Claude.
        reply = (
            f"(demo mode — set $ANTHROPIC_API_KEY for real Claude)\n"
            f"You said: {user_text}"
        )
        for chunk in _chunked(reply, 12):
            yield chunk
            time.sleep(0.02)


def _chunked(s: str, n: int):
    for i in range(0, len(s), n):
        yield s[i : i + n]


# ── Real Anthropic engine (lazy import) ──────────────────────────────────


class AnthropicEngine:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:  # pragma: no cover
                from faceview.core.errors import MissingDependency
                raise MissingDependency("anthropic", "dev") from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def stream_reply(self, conv: Conversation, user_text: str):
        client = self._ensure_client()
        messages = conv.for_anthropic()
        # Caller already added the user message to ``conv`` before invoking.
        with client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=conv.system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text


# ── Client facade ────────────────────────────────────────────────────────


class ClaudeClient:
    """Thread-backed client; emits LLM_TOKEN tokens then LLM_REPLY on done."""

    def __init__(
        self,
        conversation: Optional[Conversation] = None,
        engine=None,
    ) -> None:
        self.bus = get_bus()
        self.conversation = conversation or Conversation()
        if engine is not None:
            self.engine = engine
        elif settings.has_claude_key:
            self.engine = AnthropicEngine(
                api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
                model=settings.anthropic_model,
            )
            log.info("llm.engine", engine="anthropic", model=settings.anthropic_model)
        else:
            self.engine = EchoEngine()
            log.info("llm.engine", engine="demo")

        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="claude-worker", daemon=True)
        self._thread.start()

    # ── public ──────────────────────────────────────────────────────

    def send_async(self, msg: ChatMessage | str) -> None:
        text = msg.content if isinstance(msg, ChatMessage) else str(msg)
        self._q.put(text)

    def send_sync(self, text: str) -> str:
        """Block until reply complete; collects tokens. Mostly for tests."""
        self.conversation.add_user(text)
        chunks: list[str] = []
        try:
            for tok in self.engine.stream_reply(self.conversation, text):
                chunks.append(str(tok))
        except Exception as exc:  # noqa: BLE001
            return f"[error: {exc}]"
        reply = "".join(chunks)
        self.conversation.add_assistant(reply)
        return reply

    def stop(self) -> None:
        self._q.put(None)

    # ── worker thread ───────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            user_text = item
            try:
                self.conversation.add_user(user_text)
                chunks: list[str] = []
                for tok in self.engine.stream_reply(self.conversation, user_text):
                    chunks.append(str(tok))
                    self.bus.publish(EventType.LLM_TOKEN, str(tok))
                final = "".join(chunks)
                self.conversation.add_assistant(final)
                self.bus.publish(EventType.LLM_REPLY, ChatMessage("assistant", final))
                self.bus.publish(EventType.TTS_SPEAK, final)
            except Exception as exc:  # noqa: BLE001
                log.error("llm.error", error=str(exc))
                self.bus.publish(EventType.LLM_ERROR, str(exc))
