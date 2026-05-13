"""Chat panel: history view + input + send button.

The panel publishes ``CHAT_USER_MESSAGE`` events on submit and listens for
``LLM_TOKEN`` (incremental) and ``LLM_REPLY`` (final) events to update the
display. It does not import the LLM client directly — that decoupling lets
us swap real Claude in for the demo-mode echo and back without touching the
view.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent
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


class ChatPanel(QWidget):
    user_submitted = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._streaming_buffer = ""
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
        # speech without the TTS echo gate dropping it. Released =
        # back to normal mute-during-TTS behaviour.
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
        bus.subscribe(EventType.CHAT_USER_MESSAGE, self._on_user_msg_published)

    # ── slots ────────────────────────────────────────────────────────

    def _on_submit(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.user_submitted.emit(text)
        get_bus().publish(EventType.CHAT_USER_MESSAGE, ChatMessage("user", text))

    def _on_user_msg_published(self, msg: ChatMessage) -> None:
        self._append_block("You", msg.content, color="#1a73e8")

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
        if not self._streaming_buffer:
            self._append_block("Claude", "", color="#9b51e0")
        self._streaming_buffer += str(token)
        cursor = self.history.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(str(token))
        self.history.setTextCursor(cursor)

    def _on_reply(self, msg: ChatMessage) -> None:
        if not self._streaming_buffer:
            self._append_block("Claude", msg.content, color="#9b51e0")
        else:
            # Streaming wrote tokens directly into the open block, so
            # _append_block never logged this line — publish CHAT_LOG
            # here so the monitor still captures Claude's full reply.
            if msg.content:
                get_bus().publish(
                    EventType.CHAT_LOG,
                    ChatLogEntry(who="Claude", text=msg.content, color="#9b51e0"),
                )
        self._streaming_buffer = ""
        self._append_separator()

    def _on_llm_error(self, err: str) -> None:
        self._append_block("error", str(err), color="#c00")

    # ── helpers ──────────────────────────────────────────────────────

    def _append_block(self, who: str, body: str, *, color: str) -> None:
        html = (
            f'<div style="margin-top:6px;">'
            f'<span style="color:{color};font-weight:bold;">{who}:</span> '
            f'{self._escape(body)}'
            f'</div>'
        )
        self.history.append(html)
        # Mirror every rendered line onto the bus so the monitor endpoint
        # can show the same transcript as the GUI without screen-scraping.
        if body:
            get_bus().publish(
                EventType.CHAT_LOG,
                ChatLogEntry(who=who, text=body, color=color),
            )

    def _append_separator(self) -> None:
        self.history.append('<div style="color:#aaa;">—</div>')

    def append_external_message(self, who: str, text: str, *, color: str = "#666") -> None:
        """Display a message from a non-bus source (e.g. test-mode bots).

        Bypasses the event bus so it can't re-trigger ``ClaudeClient.send_async``.
        """
        self._append_block(who, text, color=color)

    @staticmethod
    def _escape(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

    # ── seeded demo content (for screenshots without ML) ────────────

    def seed_demo_conversation(self) -> None:
        """Populate the panel with a believable demo so screenshots aren't blank."""
        self._append_block(
            "You",
            "Hey Claude — what's the camera seeing right now?",
            color="#1a73e8",
        )
        self._append_block(
            "Claude",
            "I see one face (you), neutral expression, mouth closed. "
            "Microphone is idle. How can I help?",
            color="#9b51e0",
        )
        self._append_separator()
        self._append_block(
            "You",
            "Take a screenshot and save it as docs/images/main.png.",
            color="#1a73e8",
        )
        self._append_block(
            "Claude",
            "Done — wrote docs/images/main.png (1280×800).",
            color="#9b51e0",
        )


class _ChatInput(QLineEdit):
    submit = Signal()

    def keyPressEvent(self, ev: QKeyEvent) -> None:  # noqa: N802 — Qt API
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            ev.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submit.emit()
            return
        super().keyPressEvent(ev)
