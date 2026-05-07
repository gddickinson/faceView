"""Main window — assembles panels and exposes a screenshot helper.

Layout:

    ┌──────────────────────────────────────────────────────────┐
    │ menu / status bar                                       │
    ├────────────────────┬────────────────────┬───────────────┤
    │                    │                    │               │
    │   Camera panel     │   Chat panel       │  Status       │
    │   (overlays)       │   (history+input)  │  panel        │
    │                    │                    │               │
    │                    │                    ├───────────────┤
    │                    │                    │  Transcript   │
    │                    │                    │               │
    └────────────────────┴────────────────────┴───────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from faceview.gui.camera_panel import CameraPanel
from faceview.gui.chat_panel import ChatPanel
from faceview.gui.screenshotter import Screenshotter
from faceview.gui.status_panel import StatusPanel
from faceview.gui.transcript_panel import TranscriptPanel

if TYPE_CHECKING:
    from pathlib import Path


class MainWindow(QMainWindow):
    """Shell that composes panels and owns the screenshot helper."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("faceView")
        self.resize(1280, 800)

        self.shotter = Screenshotter()

        self.camera = CameraPanel(self)
        self.chat = ChatPanel(self)
        self.status_panel = StatusPanel(self)
        self.transcript = TranscriptPanel(self)

        self._build_layout()
        self._build_menu()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("faceView ready")

    def _build_layout(self) -> None:
        # Right column: status on top, transcript below.
        right_col = QWidget(self)
        right_v = QVBoxLayout(right_col)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.addWidget(self.status_panel, 0)
        right_v.addWidget(self.transcript, 1)

        split = QSplitter(Qt.Orientation.Horizontal, self)
        split.addWidget(self.camera)
        split.addWidget(self.chat)
        split.addWidget(right_col)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 3)

        wrapper = QWidget(self)
        h = QHBoxLayout(wrapper)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(split)
        self.setCentralWidget(wrapper)

    def _build_menu(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        a_shot = QAction("Take screenshot", self)
        a_shot.setShortcut(QKeySequence("Ctrl+Shift+S"))
        a_shot.triggered.connect(lambda: self.take_screenshot("manual.png"))
        m_file.addAction(a_shot)

        a_quit = QAction("Quit", self)
        a_quit.setShortcut(QKeySequence.StandardKey.Quit)
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_view = self.menuBar().addMenu("&View")
        a_fx = QAction("Effects panel…", self)
        a_fx.setShortcut(QKeySequence("Ctrl+E"))
        a_fx.triggered.connect(self._open_effects_panel)
        m_view.addAction(a_fx)

    def _open_effects_panel(self) -> None:
        # Lazy import so the GUI module doesn't depend on cv2 etc at boot.
        from faceview.gui.effects_panel import EffectsPanel
        if not hasattr(self, "_fx_panel") or self._fx_panel is None:
            self._fx_panel = EffectsPanel(self)
        self._fx_panel.show()
        self._fx_panel.raise_()
        self._fx_panel.activateWindow()

    # ── public helpers ──────────────────────────────────────────────

    def take_screenshot(self, name: str) -> "Path":
        return self.shotter.capture_window(self, name)

    def seed_demo_state(self) -> None:
        """Populate panels with believable demo content (for screenshots)."""
        self.chat.seed_demo_conversation()
        self.status_panel.seed_demo()
        self.transcript.seed_demo()
        self.statusBar().showMessage(
            "Demo state — owner recognised, mic idle, camera idle."
        )
