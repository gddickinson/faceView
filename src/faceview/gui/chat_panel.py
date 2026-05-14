"""Chat panel: history view + input + send button + markdown rendering.

The panel publishes ``CHAT_USER_MESSAGE`` events on submit and listens
for ``LLM_TOKEN`` (incremental) and ``LLM_REPLY`` (final) events to
update the display. It does not import the LLM client directly —
that decoupling lets us swap real Claude in for the demo-mode echo
and back without touching the view.

Markdown rendering: every finalised chat block (user + assistant)
runs through ``QTextDocument.setMarkdown`` so code fences, tables,
bold/italic/lists/headings render properly instead of showing raw
backticks and asterisks. During streaming, we keep plain-text
append for snappy live feedback; on the final ``LLM_REPLY`` we
re-render the entire transcript so the just-streamed Claude reply
gets the markdown treatment too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent, QTextDocument
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from faceview.core.event_bus import get_bus
from faceview.core.events import ChatLogEntry, ChatMessage, EventType


# Captures the body of the HTML produced by QTextDocument.toHtml().
_BODY_RE = re.compile(r"<body[^>]*>(.*?)</body>", re.DOTALL)


@dataclass
class ChatBlock:
    who: str
    color: str
    text: str
    is_streaming: bool = False


def _render_markdown_to_inner_html(text: str) -> str:
    """Run ``text`` through Qt's commonmark renderer and return the
    inner body HTML (no <html><head> wrapper). Safe on empty input."""
    if not text:
        return ""
    doc = QTextDocument()
    doc.setMarkdown(text)
    raw = doc.toHtml()
    m = _BODY_RE.search(raw)
    return m.group(1) if m else raw


def _render_block_html(block: ChatBlock) -> str:
    """One chat block: coloured "Who:" header + markdown body."""
    if block.is_streaming:
        # Streaming view: plain-text body so we don't reflow markdown
        # on every token. Final pass on LLM_REPLY swaps to rendered.
        body = (
            block.text.replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;")
                      .replace("\n", "<br>")
        )
    else:
        body = _render_markdown_to_inner_html(block.text)
    return (
        f'<div style="margin-top:6px;">'
        f'<span style="color:{block.color};font-weight:bold;">'
        f'{block.who}:</span> {body}'
        f'</div>'
    )


class ChatPanel(QWidget):
    user_submitted = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Authoritative chat history — every operation mutates this
        # list and re-renders the QTextEdit. Streaming uses a "live"
        # last entry that gets text-appended on each token.
        self._blocks: list[ChatBlock] = []
        self._live_block: ChatBlock | None = None
        self._build_ui()
        self._wire_bus()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Chat")
        f = QFont()
        f.setBold(True)
        f.setPointSize(13)
        header.setFont(f)
        root.addWidget(header)

        self.history = QTextEdit(self)
        self.history.setReadOnly(True)
        self.history.setObjectName("chat_history")
        self.history.setPlaceholderText("No messages yet — say hi.")
        root.addWidget(self.history, 1)

        row = QHBoxLayout()
        self.input = _ChatInput(self)
        self.input.setPlaceholderText("Type a message and press Enter…")
        self.input.submit.connect(self._on_submit)
        row.addWidget(self.input, 1)

        send = QPushButton("Send", self)
        send.clicked.connect(self._on_submit)
        row.addWidget(send)

        # Push-to-speak — hold to interrupt the avatar and capture
        # speech without the TTS echo gate dropping it.
        self.push_to_talk = QPushButton("🎤 Hold to talk", self)
        self.push_to_talk.setToolTip(
            "Hold while speaking to interrupt the avatar and route your "
            "voice into chat (bypasses the echo gate)."
        )
        self.push_to_talk.setAutoDefault(False)
        self.push_to_talk.pressed.connect(self._on_push_to_talk_pressed)
        self.push_to_talk.released.connect(self._on_push_to_talk_released)
        row.addWidget(self.push_to_talk)
        root.addLayout(row)

    def _wire_bus(self) -> None:
        bus = get_bus()
        bus.subscribe(EventType.LLM_TOKEN, self._on_token)
        bus.subscribe(EventType.LLM_REPLY, self._on_reply)
        bus.subscribe(EventType.LLM_ERROR, self._on_llm_error)
        bus.subscribe(EventType.CHAT_USER_MESSAGE,
                      self._on_user_msg_published)

    # ── rendering ────────────────────────────────────────────────────

    def _rerender(self) -> None:
        """Rebuild the QTextEdit's HTML from the block list."""
        parts = [_render_block_html(b) for b in self._blocks]
        # Cursor at end so the user stays scrolled to the latest reply.
        self.history.setHtml("".join(parts))
        cursor = self.history.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.history.setTextCursor(cursor)

    def _append_finalised(
        self, who: str, body: str, *, color: str,
    ) -> None:
        """Add a complete (non-streaming) block + re-render."""
        self._blocks.append(ChatBlock(who=who, color=color, text=body))
        self._rerender()
        if body:
            get_bus().publish(
                EventType.CHAT_LOG,
                ChatLogEntry(who=who, text=body, color=color),
            )

    # ── slots ────────────────────────────────────────────────────────

    def _on_submit(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.user_submitted.emit(text)
        get_bus().publish(EventType.CHAT_USER_MESSAGE,
                          ChatMessage("user", text))

    def _on_user_msg_published(self, msg: ChatMessage) -> None:
        self._append_finalised("You", msg.content, color="#1a73e8")

    def _main_window(self):
        """Walk up the parent chain to find the MainWindow."""
        w = self.parent()
        while w is not None:
            if hasattr(w, "push_to_speak_pressed"):
                return w
            w = w.parent()
        return None

    def _on_push_to_talk_pressed(self) -> None:
        mw = self._main_window()
        if mw is not None:
            mw.push_to_speak_pressed()

    def _on_push_to_talk_released(self) -> None:
        mw = self._main_window()
        if mw is not None:
            mw.push_to_speak_released()

    def _on_token(self, token: str) -> None:
        if self._live_block is None:
            self._live_block = ChatBlock(
                who="Claude", color="#9b51e0", text="",
                is_streaming=True,
            )
            self._blocks.append(self._live_block)
        self._live_block.text += str(token)
        # Re-render — streaming blocks render as plain text (cheap)
        # so this stays snappy even with several finalised markdown
        # blocks above.
        self._rerender()

    def _on_reply(self, msg: ChatMessage) -> None:
        if self._live_block is None:
            # The model didn't stream tokens (e.g. demo engine that
            # yields the whole reply at once). Append as a finalised
            # block directly.
            self._append_finalised("Claude", msg.content, color="#9b51e0")
            return
        # Promote the streaming block to a finalised, markdown-
        # rendered one. The reply text from LLM_REPLY is the
        # authoritative full content.
        self._live_block.text = msg.content
        self._live_block.is_streaming = False
        self._live_block = None
        self._rerender()
        if msg.content:
            get_bus().publish(
                EventType.CHAT_LOG,
                ChatLogEntry(who="Claude", text=msg.content,
                             color="#9b51e0"),
            )

    def _on_llm_error(self, err: str) -> None:
        # Drop any in-flight streaming block so the error doesn't
        # render as part of Claude's reply.
        if self._live_block is not None:
            self._blocks.remove(self._live_block)
            self._live_block = None
        self._append_finalised("error", str(err), color="#c00")

    # ── external entry points ────────────────────────────────────────

    def append_external_message(
        self, who: str, text: str, *, color: str = "#666",
    ) -> None:
        """Display a message from a non-bus source (test-mode bots).

        Bypasses the event bus so it can't re-trigger
        ``ClaudeClient.send_async``."""
        self._append_finalised(who, text, color=color)

    # ── seeded demo content (for screenshots without ML) ────────────

    def seed_demo_conversation(self) -> None:
        """Populate the panel with a believable demo so screenshots
        aren't blank. Includes markdown (code fence + emphasis) so
        the rendering capability shows in the README shots."""
        self._blocks = [
            ChatBlock(
                "You", "#1a73e8",
                "Hey Claude — what's the camera seeing right now?",
            ),
            ChatBlock(
                "Claude", "#9b51e0",
                "I see **one face** (you), neutral expression, mouth "
                "closed. Microphone is idle. How can I help?",
            ),
            ChatBlock(
                "You", "#1a73e8",
                "Take a screenshot and save it as `docs/images/main.png`.",
            ),
            ChatBlock(
                "Claude", "#9b51e0",
                "Done — wrote `docs/images/main.png` (1280×800). I used:\n"
                "\n"
                "```python\n"
                "shotter.capture_window(window, 'main.png')\n"
                "```",
            ),
        ]
        self._rerender()


class _ChatInput(QLineEdit):
    submit = Signal()

    def keyPressEvent(self, ev: QKeyEvent) -> None:  # noqa: N802 — Qt API
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            ev.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submit.emit()
            return
        super().keyPressEvent(ev)
