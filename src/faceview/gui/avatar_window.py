"""AvatarWindow — separate top-level window showing the Claude avatar.

Pairs with :class:`MainWindow` so the user can have *their* face
(webcam) and *Claude's* face (avatar) side by side. The avatar is
driven by ``EventType.AVATAR_FRAME`` and ``EventType.LLM_REPLY``;
the actual rendering happens in a :class:`SimCameraWorker`.

Implementation note: this is intentionally a :class:`QWidget` with a
``Window`` flag rather than a :class:`QMainWindow` — on macOS a second
``QMainWindow`` fights the first for the global menu-bar slot and the
main window's menus disappear whenever the avatar window has focus.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from faceview.core.event_bus import get_bus
from faceview.core.events import ChatMessage, Emotion, EventType
from faceview.gui.avatar_panel import AvatarPanel


class AvatarWindow(QWidget):
    """Standalone window representing Claude visually during a conversation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        # Pass parent=None so this window owns its own lifecycle but still
        # uses the application's (single) menu bar.
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("Claude")
        self.resize(560, 640)

        self.avatar = AvatarPanel(self)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.avatar, 1)

        # Compact "what Claude is doing" strip under the canvas.
        strip = QWidget(self)
        strip.setMinimumHeight(28)
        pal = strip.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#13171f"))
        strip.setAutoFillBackground(True)
        strip.setPalette(pal)
        h = QHBoxLayout(strip)
        h.setContentsMargins(10, 4, 10, 4)
        self.mood_label = QLabel("mood: neutral")
        self.mood_label.setStyleSheet("color:#9aa3b2;")
        self.say_label = QLabel("")
        self.say_label.setStyleSheet("color:#cdd3e0;")
        f = QFont()
        f.setItalic(True)
        self.say_label.setFont(f)
        h.addWidget(self.mood_label, 0)
        h.addSpacing(16)
        h.addWidget(self.say_label, 1)
        v.addWidget(strip, 0)

        self._wire_bus()

    def _wire_bus(self) -> None:
        bus = get_bus()
        bus.subscribe(EventType.EMOTION, self._on_emotion)
        bus.subscribe(EventType.LLM_REPLY, self._on_reply)

    def _on_emotion(self, e: Emotion) -> None:
        self.mood_label.setText(f"mood: {e.label}")

    def _on_reply(self, msg: ChatMessage) -> None:
        text = getattr(msg, "content", "") or ""
        snippet = text if len(text) <= 80 else text[:77] + "…"
        self.say_label.setText(f"“{snippet}”")
