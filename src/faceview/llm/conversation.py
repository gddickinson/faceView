"""In-memory conversation history.

Lightweight enough that we can copy snapshots into worker threads without
sharing mutable state.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from faceview.core.events import ChatMessage


class Conversation:
    def __init__(self, system: str | None = None) -> None:
        self.system = system or DEFAULT_SYSTEM
        self._messages: list[ChatMessage] = []
        # Callables that return extra system context to prepend to
        # ``system`` at inference time. The cognition layer registers
        # the persistent-context provider; the perception layer
        # registers a live-status provider; both are concatenated in
        # registration order in :meth:`effective_system`.
        self._extras_providers: list = []

    def set_system_extras_provider(self, provider) -> None:
        """Replace ALL extras providers with this single one (legacy API).

        Pass ``None`` to clear. Use :meth:`add_system_extras_provider`
        when you want to *compose* multiple providers (e.g. cognition +
        live perception)."""
        self._extras_providers = [provider] if provider is not None else []

    def add_system_extras_provider(self, provider) -> None:
        """Append an additional ``() -> str`` provider. The texts are
        joined with blank lines in the order they were added."""
        if provider is not None:
            self._extras_providers.append(provider)

    def remove_system_extras_provider(self, provider) -> None:
        try:
            self._extras_providers.remove(provider)
        except ValueError:
            pass

    def effective_system(self) -> str:
        """System prompt seen by the engine on this turn."""
        parts: list[str] = []
        for provider in self._extras_providers:
            try:
                extra = provider() or ""
            except Exception:  # noqa: BLE001 — providers must not crash inference
                extra = ""
            if extra:
                parts.append(extra)
        parts.append(self.system)
        return "\n\n".join(parts)

    # ── context-budget compaction (L10) ────────────────────────

    def estimate_tokens(self) -> int:
        """Whitespace-word-count × 1.3 — close enough to real BPE
        tokenisation that we can budget without pulling tiktoken."""
        total = 0
        sys = self.effective_system()
        if sys:
            total += int(len(sys.split()) * 1.3)
        for m in self._messages:
            total += int(len((m.content or "").split()) * 1.3)
        return total

    def maybe_compact(self, budget_tokens: int = 100_000) -> bool:
        """If the conversation is approaching ``budget_tokens``, fold
        the oldest user+assistant turns into one summary block so
        the engine doesn't blow past the model's context window.

        Returns True when compaction fired. Idempotent: an existing
        summary block stays at the start; only the *next* oldest
        turns get folded into it.
        """
        if budget_tokens <= 0:
            return False
        if self.estimate_tokens() < budget_tokens:
            return False
        # Identify the summary block (if any) — it lives as the first
        # assistant message tagged with the marker.
        marker = "[earlier conversation summarised]"
        # Walk forward collecting messages to fold. Keep at least the
        # last ~30 turns intact so context recency is preserved.
        keep_tail = 30
        if len(self._messages) <= keep_tail:
            return False
        # Pop the existing summary if it's the first message so we can
        # rebuild it with the newly-folded content.
        existing_summary = ""
        if (self._messages and self._messages[0].role == "assistant"
                and marker in self._messages[0].content):
            existing_summary = self._messages[0].content
            self._messages.pop(0)
        # Fold everything up to keep_tail-from-end into a one-line
        # summary per turn pair.
        cut = len(self._messages) - keep_tail
        if cut <= 0:
            return False
        folded = self._messages[:cut]
        self._messages = self._messages[cut:]
        # Build the new summary text.
        lines: list[str] = []
        prev_summary_body = existing_summary.replace(marker, "").strip()
        if prev_summary_body:
            lines.append(prev_summary_body)
        for m in folded:
            who = "you" if m.role == "user" else "I"
            snippet = (m.content or "").strip().replace("\n", " ")
            if len(snippet) > 140:
                snippet = snippet[:137] + "…"
            lines.append(f"- {who}: {snippet}")
        body = "\n".join(lines)
        summary_text = f"{marker}\n{body}"
        # Reinsert at the head.
        self._messages.insert(
            0, ChatMessage(role="assistant", content=summary_text),
        )
        return True

    def append(self, msg: ChatMessage) -> None:
        self._messages.append(msg)

    def add_user(self, text: str) -> ChatMessage:
        m = ChatMessage(role="user", content=text)
        self._messages.append(m)
        return m

    def add_assistant(self, text: str) -> ChatMessage:
        m = ChatMessage(role="assistant", content=text)
        self._messages.append(m)
        return m

    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def for_anthropic(self) -> list[dict]:
        """Convert to the schema the Anthropic SDK expects."""
        return [
            {"role": m.role, "content": m.content}
            for m in self._messages
            if m.role in ("user", "assistant")
        ]

    def to_jsonable(self) -> dict:
        return {
            "system": self.system,
            "messages": [asdict(m) for m in self._messages],
        }

    def __len__(self) -> int:
        return len(self._messages)


DEFAULT_SYSTEM = (
    "You are running inside faceView, a desktop GUI that combines a webcam, "
    "microphone, and chat panel. The user can speak to you, type, or be "
    "watched by camera. Be concise and friendly. If you describe what you see "
    "or hear, base it only on the structured status data the GUI passes you."
)
