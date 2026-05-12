"""Full STT transcript-history monitor.

Same data the main `TranscriptPanel` shows, but in a free-standing
window with timestamps so the user can keep a long history view
without occupying main-window real estate.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDialog, QLabel, QTextEdit, QVBoxLayout, QWidget

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, Transcript


class TranscriptMonitor(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("STT — full transcript history")
        self.setMinimumSize(540, 360)
        self.setWindowFlag(Qt.WindowType.Window, True)

        root = QVBoxLayout(self)
        head = QLabel("STT transcript — partial = grey/italic, final = bold")
        head.setStyleSheet("color:#cdd3e0;")
        root.addWidget(head)

        self.view = QTextEdit(self)
        self.view.setReadOnly(True)
        f = QFont("Menlo")
        f.setPointSize(11)
        self.view.setFont(f)
        root.addWidget(self.view, 1)

        bus = get_bus()
        bus.subscribe(EventType.TRANSCRIPT_PARTIAL, self._on_partial)
        bus.subscribe(EventType.TRANSCRIPT_FINAL, self._on_final)

    def _on_partial(self, t: Transcript) -> None:
        ts = time.strftime("%H:%M:%S")
        self.view.append(
            f'<span style="color:#888;">[{ts}]</span> '
            f'<span style="color:#888;font-style:italic;">… {self._escape(t.text)}</span>'
        )

    def _on_final(self, t: Transcript) -> None:
        ts = time.strftime("%H:%M:%S")
        self.view.append(
            f'<span style="color:#888;">[{ts}]</span> '
            f'<span style="color:#cdd3e0;font-weight:600;">● {self._escape(t.text)}</span>'
        )

    @staticmethod
    def _escape(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
