"""Camera preview panel.

If the camera worker is wired in, frames are received via
:data:`EventType.FRAME` and converted to ``QImage`` for display. Without a
camera (or in headless mode), the panel paints a placeholder pattern so the
GUI still looks complete in screenshots.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType


class CameraPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._latest: QPixmap | None = None
        self._build_ui()
        self._wire_bus()
        self._paint_placeholder()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Camera")
        f = QFont()
        f.setBold(True)
        f.setPointSize(13)
        header.setFont(f)
        root.addWidget(header)

        self.view = QLabel(self)
        self.view.setObjectName("camera_view")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setMinimumSize(QSize(480, 360))
        pal = self.view.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#111"))
        self.view.setAutoFillBackground(True)
        self.view.setPalette(pal)
        root.addWidget(self.view, 1)

    def _wire_bus(self) -> None:
        bus = get_bus()
        bus.subscribe(EventType.FRAME, self._on_frame)

    # ── slots ────────────────────────────────────────────────────────

    def _on_frame(self, payload) -> None:
        # payload is a numpy uint8 BGR array from cv2; we convert to RGB QImage.
        if not isinstance(payload, np.ndarray):
            return
        h, w = payload.shape[:2]
        if payload.ndim == 3 and payload.shape[2] == 3:
            rgb = payload[:, :, ::-1].copy()  # BGR → RGB
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        else:
            qimg = QImage(payload.data, w, h, w, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(
            self.view.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._latest = pix
        self.view.setPixmap(pix)

    # ── placeholder ─────────────────────────────────────────────────

    def _paint_placeholder(self) -> None:
        size = QSize(640, 480)
        pix = QPixmap(size)
        pix.fill(QColor("#0c0f14"))

        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # decorative crosshair
        pen = QPen(QColor("#2a3340"))
        pen.setWidth(2)
        p.setPen(pen)
        cx, cy = size.width() // 2, size.height() // 2
        p.drawLine(cx - 60, cy, cx + 60, cy)
        p.drawLine(cx, cy - 60, cx, cy + 60)
        p.setPen(QPen(QColor("#3b465c"), 2))
        p.drawEllipse(cx - 110, cy - 140, 220, 280)

        # caption
        p.setPen(QColor("#9aa3b2"))
        f = QFont()
        f.setPointSize(13)
        p.setFont(f)
        p.drawText(
            pix.rect().adjusted(0, 200, 0, 0),
            Qt.AlignmentFlag.AlignHCenter,
            "Camera idle\nstart the camera worker to stream frames",
        )
        p.end()

        self._latest = pix
        self.view.setPixmap(
            pix.scaled(
                self.view.size() if self.view.size().width() > 0 else size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, ev) -> None:  # noqa: N802 — Qt API
        super().resizeEvent(ev)
        if self._latest is not None:
            self.view.setPixmap(
                self._latest.scaled(
                    self.view.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
