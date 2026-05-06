"""Streaming-transcript panel.

Shows partial STT output (greyed) and final segments (solid) as they arrive.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, Transcript


class TranscriptPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._wire_bus()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Transcript")
        f = QFont()
        f.setBold(True)
        f.setPointSize(13)
        header.setFont(f)
        root.addWidget(header)

        self.view = QTextEdit(self)
        self.view.setReadOnly(True)
        self.view.setPlaceholderText("Listening will produce partial then final lines here.")
        root.addWidget(self.view, 1)

    def _wire_bus(self) -> None:
        bus = get_bus()
        bus.subscribe(EventType.TRANSCRIPT_PARTIAL, self._on_partial)
        bus.subscribe(EventType.TRANSCRIPT_FINAL, self._on_final)

    def _on_partial(self, t: Transcript) -> None:
        self.view.append(f'<span style="color:#888;font-style:italic;">… {self._escape(t.text)}</span>')

    def _on_final(self, t: Transcript) -> None:
        self.view.append(f'<span>● {self._escape(t.text)}</span>')

    @staticmethod
    def _escape(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def seed_demo(self) -> None:
        self._on_final(Transcript(text="Hey Claude, what's the weather like today?", is_final=True))
        self._on_partial(Transcript(text="And could you also help me", is_final=False))
        self._on_final(Transcript(text="And could you also help me debug this Python script.", is_final=True))
