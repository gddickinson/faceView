"""Emotion-monitor window — per-class score bars + recent history.

Subscribes to :data:`EventType.EMOTION` and renders:

- a horizontal bar chart of the latest 7-class probability distribution;
- a small rolling timeline of the dominant emotion (last ~30 events).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget

from faceview.core.event_bus import get_bus
from faceview.core.events import Emotion, EventType


CLASSES = ["happy", "neutral", "sad", "surprise", "angry", "fear", "disgust"]
COLORS = {
    "happy":   "#22a36b",
    "neutral": "#666666",
    "sad":     "#5066c0",
    "surprise":"#e8a23a",
    "angry":   "#c0392b",
    "fear":    "#8e44ad",
    "disgust": "#7d6608",
    "unknown": "#222",
}


class _BarChart(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scores: dict[str, float] = {c: 0.0 for c in CLASSES}
        self.setMinimumHeight(220)

    def update_scores(self, e: Emotion) -> None:
        self._scores = {c: float(e.scores.get(c, 0.0)) for c in CLASSES}
        # If DeepFace omitted some classes, ensure the top label is present.
        if e.label not in self._scores and e.label != "unknown":
            self._scores[e.label] = float(e.confidence)
        self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802 — Qt API
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0c0f14"))
        w = self.width()
        h = self.height()
        n = max(1, len(self._scores))
        row_h = (h - 16) / n
        label_w = 90
        bar_max = w - label_w - 16
        for i, (cls, score) in enumerate(self._scores.items()):
            y = 8 + i * row_h
            p.setPen(QColor("#cdd3e0"))
            p.drawText(8, int(y + row_h * 0.65), cls)
            bar_w = max(1, int(bar_max * min(1.0, max(0.0, score))))
            p.fillRect(label_w, int(y + 4), bar_w, int(row_h - 8), QColor(COLORS.get(cls, "#888")))
            p.setPen(QColor("#9aa3b2"))
            p.drawText(label_w + bar_w + 6, int(y + row_h * 0.65), f"{score * 100:.0f}%")


class EmotionMonitor(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Emotion — per-class scores")
        self.setMinimumSize(440, 320)
        self.setWindowFlag(Qt.WindowType.Window, True)

        root = QVBoxLayout(self)
        self.dominant = QLabel("dominant: —")
        self.dominant.setStyleSheet("color:#cdd3e0;font-weight:600;")
        root.addWidget(self.dominant)

        self.bars = _BarChart(self)
        root.addWidget(self.bars, 1)

        self.history = QLabel("history: (waiting for events)")
        self.history.setWordWrap(True)
        self.history.setStyleSheet("color:#9aa3b2;")
        root.addWidget(self.history)

        self._hist: deque[str] = deque(maxlen=30)
        get_bus().subscribe(EventType.EMOTION, self._on_emotion)

    def _on_emotion(self, e: Emotion) -> None:
        self.bars.update_scores(e)
        self.dominant.setText(f"dominant: {e.label} ({e.confidence:.0%})")
        self._hist.append(e.label)
        self.history.setText("history: " + " → ".join(list(self._hist)[-20:]))
