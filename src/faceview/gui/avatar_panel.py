"""Avatar preview panel — shows the rendered Claude avatar.

This is the visual counterpart to :class:`CameraPanel`. Where the camera
panel shows the *user* (real webcam, analysed by the vision pipeline),
the avatar panel shows *Claude*: a synthetic talking head whose lip-sync
and emotion are driven by the LLM reply stream.

It subscribes to :data:`EventType.AVATAR_FRAME` so it stays cleanly
separated from real camera traffic.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType


class AvatarPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._latest: QPixmap | None = None
        self._build_ui()
        self._wire_bus()
        self._paint_placeholder()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Claude")
        f = QFont()
        f.setBold(True)
        f.setPointSize(13)
        header.setFont(f)
        root.addWidget(header)

        self.view = QLabel(self)
        self.view.setObjectName("avatar_view")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setMinimumSize(QSize(480, 360))
        pal = self.view.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#0a0d12"))
        self.view.setAutoFillBackground(True)
        self.view.setPalette(pal)
        root.addWidget(self.view, 1)

    def _wire_bus(self) -> None:
        get_bus().subscribe(EventType.AVATAR_FRAME, self._on_frame)

    def _on_frame(self, payload) -> None:
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

    def _paint_placeholder(self) -> None:
        size = QSize(640, 480)
        pix = QPixmap(size)
        pix.fill(QColor("#0a0d12"))

        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = size.width() // 2, size.height() // 2
        p.setPen(QPen(QColor("#3b465c"), 2))
        p.drawEllipse(cx - 110, cy - 140, 220, 280)
        p.setPen(QColor("#9aa3b2"))
        f = QFont()
        f.setPointSize(13)
        p.setFont(f)
        p.drawText(
            pix.rect().adjusted(0, 200, 0, 0),
            Qt.AlignmentFlag.AlignHCenter,
            "Avatar idle\nClaude appears here when the avatar worker is running",
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
