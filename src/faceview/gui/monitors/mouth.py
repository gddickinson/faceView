"""Mouth + viseme monitor — jaw_open trace + recent viseme stream.

Subscribes to :data:`EventType.MOUTH_ACTIVITY` and renders the last
~5 seconds of jaw-open values plus a rolling viseme sequence.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, MouthActivity


_TRACE_LEN = 300  # ~30 s at 10 Hz


class _Trace(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buf: deque[float] = deque(maxlen=_TRACE_LEN)
        self.setMinimumHeight(160)
        t = QTimer(self)
        t.timeout.connect(self.update)
        t.start(80)

    def push(self, v: float) -> None:
        self._buf.append(float(v))

    def paintEvent(self, _ev) -> None:  # noqa: N802 — Qt API
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0c0f14"))
        w = self.width()
        h = self.height()
        if not self._buf:
            p.setPen(QColor("#666"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no mouth data yet")
            return
        # Y baseline + threshold line at 0.10 (the speaking threshold).
        p.setPen(QPen(QColor("#222"), 1))
        p.drawLine(0, h - 1, w, h - 1)
        thresh_y = int(h - 1 - 0.10 * (h - 4) * 3.0)  # exaggerate scale
        p.setPen(QPen(QColor("#444"), 1, Qt.PenStyle.DashLine))
        p.drawLine(0, thresh_y, w, thresh_y)
        # Trace.
        n = len(self._buf)
        step = w / max(1, n - 1)
        pen = QPen(QColor("#1a73e8"))
        pen.setWidth(2)
        p.setPen(pen)
        prev_x, prev_y = None, None
        for i, v in enumerate(self._buf):
            x = int(i * step)
            y = int(h - 1 - min(1.0, v) * (h - 4) * 3.0)
            if prev_x is not None:
                p.drawLine(prev_x, prev_y, x, y)
            prev_x, prev_y = x, y


class MouthMonitor(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mouth — jaw + visemes")
        self.setMinimumSize(520, 280)
        self.setWindowFlag(Qt.WindowType.Window, True)

        root = QVBoxLayout(self)
        self.head = QLabel("speaking: no    viseme: —")
        self.head.setStyleSheet("color:#cdd3e0;font-weight:600;")
        root.addWidget(self.head)

        self.trace = _Trace(self)
        root.addWidget(self.trace, 1)

        self.viseme_strip = QLabel("(no visemes yet)")
        self.viseme_strip.setStyleSheet(
            "background:#0c0f14;color:#22a36b;font-family:monospace;"
            "padding:6px;border:1px solid #222;"
        )
        root.addWidget(self.viseme_strip)

        self._visemes: deque[str] = deque(maxlen=40)
        get_bus().subscribe(EventType.MOUTH_ACTIVITY, self._on_mouth)

    def _on_mouth(self, m: MouthActivity) -> None:
        self.head.setText(
            f"speaking: {'yes' if m.speaking else 'no'}    "
            f"viseme: {m.viseme or '—'}    "
            f"jaw_open: {m.jaw_open:.2f}"
        )
        self.trace.push(m.jaw_open)
        if m.viseme:
            self._visemes.append(m.viseme)
            self.viseme_strip.setText(" ".join(self._visemes))
