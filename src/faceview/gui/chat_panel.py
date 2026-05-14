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
from PySide6.QtGui import (
    QFont, QKeyEvent, QKeySequence, QShortcut, QTextCursor, QTextDocument,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QToolButton,
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

        # Find bar — shown via Ctrl+F (see _wire_shortcuts).
        self.find_bar = _ChatFindBar(self)
        self.find_bar.hide()
        self.find_bar.find_next.connect(
            lambda q: self._do_find(q, backward=False),
        )
        self.find_bar.find_prev.connect(
            lambda q: self._do_find(q, backward=True),
        )
        self.find_bar.dismissed.connect(self._dismiss_find)
        root.addWidget(self.find_bar)

        self.history = QTextEdit(self)
        self.history.setReadOnly(True)
        self.history.setObjectName("chat_history")
        self.history.setPlaceholderText("No messages yet — say hi.")
        root.addWidget(self.history, 1)

        self._wire_shortcuts()

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

    def _wire_shortcuts(self) -> None:
        # Ctrl+F (Cmd+F on macOS) opens the find bar focussed on its
        # input. Esc inside the bar dismisses. Enter/Shift-Enter while
        # the bar has focus drives find-next / find-prev.
        find_sc = QShortcut(QKeySequence.StandardKey.Find, self)
        find_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        find_sc.activated.connect(self._show_find)
        # Find-next on F3 / Cmd+G is a nice bonus.
        next_sc = QShortcut(QKeySequence.StandardKey.FindNext, self)
        next_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        next_sc.activated.connect(
            lambda: self._do_find(self.find_bar.query(), backward=False)
        )
        prev_sc = QShortcut(QKeySequence.StandardKey.FindPrevious, self)
        prev_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        prev_sc.activated.connect(
            lambda: self._do_find(self.find_bar.query(), backward=True)
        )

    # ── find / search ────────────────────────────────────────────────

    def _show_find(self) -> None:
        # Pre-fill from the user's current selection so highlighting a
        # word and hitting Cmd+F searches for it.
        cursor = self.history.textCursor()
        if cursor.hasSelection():
            self.find_bar.set_query(cursor.selectedText())
        self.find_bar.show()
        self.find_bar.focus_input()

    def _dismiss_find(self) -> None:
        # Clear any leftover highlight + return focus to the input.
        self.find_bar.hide()
        cursor = self.history.textCursor()
        cursor.clearSelection()
        self.history.setTextCursor(cursor)
        self.input.setFocus()

    def _do_find(self, query: str, *, backward: bool) -> None:
        if not query:
            self.find_bar.set_status("")
            return
        flags = QTextDocument.FindFlag(0)
        if backward:
            flags |= QTextDocument.FindFlag.FindBackward
        # QTextEdit.find advances from the current cursor — exactly
        # the behaviour Ctrl+F users expect.
        ok = self.history.find(query, flags)
        if not ok:
            # Wrap: try from the top (or bottom for backwards).
            cursor = self.history.textCursor()
            if backward:
                cursor.movePosition(QTextCursor.MoveOperation.End)
            else:
                cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.history.setTextCursor(cursor)
            ok = self.history.find(query, flags)
        self.find_bar.set_status(
            "" if ok else f'no match for "{query}"'
        )

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


class _ChatFindBar(QWidget):
    """Find-in-chat overlay shown via Ctrl/Cmd+F.

    Emits :pysignal:`find_next` / :pysignal:`find_prev` with the
    current query; :pysignal:`dismissed` when the user presses Esc
    or clicks the close button."""

    find_next = Signal(str)
    find_prev = Signal(str)
    dismissed = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(4)

        self._input = QLineEdit(self)
        self._input.setPlaceholderText("Find in chat…  (Enter, Shift-Enter)")
        self._input.returnPressed.connect(self._on_return)
        self._input.installEventFilter(self)
        row.addWidget(self._input, 1)

        prev_btn = QToolButton(self)
        prev_btn.setText("↑")
        prev_btn.setToolTip("Previous match (Shift-Enter)")
        prev_btn.clicked.connect(
            lambda: self.find_prev.emit(self.query())
        )
        row.addWidget(prev_btn)

        next_btn = QToolButton(self)
        next_btn.setText("↓")
        next_btn.setToolTip("Next match (Enter)")
        next_btn.clicked.connect(
            lambda: self.find_next.emit(self.query())
        )
        row.addWidget(next_btn)

        close_btn = QToolButton(self)
        close_btn.setText("✕")
        close_btn.setToolTip("Close (Esc)")
        close_btn.clicked.connect(self.dismissed.emit)
        row.addWidget(close_btn)

        self._status = QLabel("", self)
        self._status.setStyleSheet("color:#c0392b;font-size:11px;")
        row.addWidget(self._status)

    # ── public ───────────────────────────────────────────────

    def query(self) -> str:
        return self._input.text().strip()

    def set_query(self, text: str) -> None:
        self._input.setText(text or "")

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def focus_input(self) -> None:
        self._input.setFocus()
        self._input.selectAll()

    # ── events ───────────────────────────────────────────────

    def eventFilter(self, obj, ev):  # noqa: N802 — Qt API
        if obj is self._input and ev.type() == ev.Type.KeyPress:
            if ev.key() == Qt.Key.Key_Escape:
                self.dismissed.emit()
                return True
        return super().eventFilter(obj, ev)

    def _on_return(self) -> None:
        from PySide6.QtWidgets import QApplication
        mods = QApplication.keyboardModifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            self.find_prev.emit(self.query())
        else:
            self.find_next.emit(self.query())
