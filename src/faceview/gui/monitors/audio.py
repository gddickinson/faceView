"""Audio waveform + VAD-state monitor window.

Subscribes to ``AUDIO_CHUNK`` for the rolling waveform and to
``VAD_SPEECH_START`` / ``VAD_SPEECH_END`` for the speech / silence pill.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType


_BUFFER_SAMPLES = 16_000 * 3   # 3 s @ 16 kHz


class _Waveform(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buf: deque[float] = deque(maxlen=_BUFFER_SAMPLES)
        self.setMinimumHeight(160)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(60)  # ~16 fps redraw

    def append(self, data: np.ndarray) -> None:
        if data is None:
            return
        flat = np.asarray(data).reshape(-1).astype(np.float32, copy=False)
        # AudioCapture publishes int16 chunks — normalise to [-1, 1] for
        # display. (If we ever receive float32 it's already in range.)
        if flat.dtype == np.float32 and (np.abs(flat).max() if flat.size else 0) > 2.0:
            flat = flat / 32768.0
        elif flat.size and np.abs(flat).max() > 2.0:
            flat = flat / 32768.0
        # Downsample chunks > 800 samples for cheaper drawing.
        if flat.size > 800:
            step = max(1, flat.size // 800)
            flat = flat[::step]
        self._buf.extend(float(v) for v in flat.tolist())

    def paintEvent(self, _ev) -> None:  # noqa: N802 — Qt API
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0c0f14"))
        if not self._buf:
            p.setPen(QColor("#666"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no audio yet")
            return
        w = self.width()
        h = self.height()
        mid = h / 2
        n = len(self._buf)
        # Map every column to one buffer sample range.
        step = max(1, n // max(1, w))
        pen = QPen(QColor("#28a745"))
        pen.setWidth(1)
        p.setPen(pen)
        prev_y = mid
        for x in range(w):
            idx = min(n - 1, x * step)
            v = self._buf[idx]
            y = mid - v * (mid * 0.95)
            p.drawLine(x - 1, prev_y, x, y)
            prev_y = y


class AudioMonitor(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audio — waveform + VAD")
        self.setMinimumSize(520, 260)
        self.setWindowFlag(Qt.WindowType.Window, True)

        root = QVBoxLayout(self)
        self.state_pill = QLabel("VAD: silent")
        self.state_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_pill.setStyleSheet(
            "background:#444;color:white;border-radius:12px;"
            "padding:4px 12px;font-weight:600;"
        )
        self.state_pill.setMaximumWidth(160)
        root.addWidget(self.state_pill)

        self.wave = _Waveform(self)
        root.addWidget(self.wave, 1)

        self.hint = QLabel(
            "Enable the microphone in Tools → Configuration… to see audio.\n"
            "Speak — the pill flips to “speech” when VAD detects voice."
        )
        self.hint.setStyleSheet("color:#888;")
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)

        bus = get_bus()
        bus.subscribe(EventType.AUDIO_CHUNK, self._on_chunk)
        bus.subscribe(EventType.VAD_SPEECH_START, lambda _p: self._set_state(True))
        bus.subscribe(EventType.VAD_SPEECH_END, lambda _p: self._set_state(False))

    def _on_chunk(self, data) -> None:
        self.wave.append(data)

    def _set_state(self, speaking: bool) -> None:
        if speaking:
            self.state_pill.setText("VAD: speech")
            self.state_pill.setStyleSheet(
                "background:#1a73e8;color:white;border-radius:12px;"
                "padding:4px 12px;font-weight:600;"
            )
        else:
            self.state_pill.setText("VAD: silent")
            self.state_pill.setStyleSheet(
                "background:#444;color:white;border-radius:12px;"
                "padding:4px 12px;font-weight:600;"
            )
