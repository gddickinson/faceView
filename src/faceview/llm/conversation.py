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
