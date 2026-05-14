"""Shared base class for MainWindow controllers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from faceview.core.logger import get_logger

if TYPE_CHECKING:
    from faceview.gui.main_window import MainWindow


class BaseController:
    """Thin base — every controller has a back-reference to the window
    and a structlog logger. Cross-controller access goes via the
    window (e.g. ``self.window.audio_ctrl.muted``)."""

    log_name: str = "controller"

    def __init__(self, window: "MainWindow") -> None:
        self.window = window
        self.log = get_logger(self.log_name)

    def status(self, message: str) -> None:
        """Push a transient message to the main window's status bar."""
        try:
            self.window.statusBar().showMessage(message)
        except Exception:  # noqa: BLE001 — status bar must never crash a worker
            pass

    def clear_status(self) -> None:
        try:
            self.window.statusBar().clearMessage()
        except Exception:  # noqa: BLE001
            pass
