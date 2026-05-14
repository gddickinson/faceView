"""Monitor-window manager — opens and reuses the small diagnostic
sub-windows (audio waveform, emotion scores, mouth + visemes,
transcript history)."""

from __future__ import annotations

from typing import Any

from faceview.gui.controllers.base import BaseController


class MonitorController(BaseController):
    log_name = "monitor_ctrl"

    def __init__(self, window) -> None:
        super().__init__(window)
        self._windows: dict[str, Any] = {}

    def open(self, kind: str) -> None:
        from faceview.gui import monitors
        cls = monitors.MONITORS.get(kind)
        if cls is None:
            return
        win = self._windows.get(kind)
        if win is None:
            win = cls(self.window)
            self._windows[kind] = win
        win.show()
        win.raise_()
        win.activateWindow()

    def close_all(self) -> None:
        for win in list(self._windows.values()):
            try:
                win.close()
            except Exception:  # noqa: BLE001
                pass
        self._windows.clear()
