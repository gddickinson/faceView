"""Screenshot helper that works for both visible and offscreen GUIs.

``QWidget.grab()`` returns a ``QPixmap`` of the widget's rendered tree
without requiring the window to be on top, minimized, or even shown — it just
needs to be laid out. That makes it the right primitive for both modes.

Usage::

    shot = Screenshotter()
    path = shot.capture(main_window, "docs/images/main.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from faceview.core.events import EventType
from faceview.core.event_bus import get_bus
from faceview.core.logger import get_logger
from faceview.utils.paths import docs_image_dir


log = get_logger("screenshotter")


class Screenshotter(QObject):
    """Captures any QWidget to PNG using ``widget.grab()``."""

    def capture(
        self,
        widget: QWidget,
        path: str | Path,
        *,
        process_events: bool = True,
        device_pixel_ratio: Optional[float] = None,
    ) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = docs_image_dir() / path.name

        if process_events:
            # Force pending paint events to flush so we don't capture an
            # unfinished frame.
            QApplication.processEvents()

        pix: QPixmap = widget.grab()
        if device_pixel_ratio:
            pix.setDevicePixelRatio(device_pixel_ratio)

        path.parent.mkdir(parents=True, exist_ok=True)
        ok = pix.save(str(path), "PNG")
        if not ok:
            raise RuntimeError(f"Failed to save screenshot to {path}")

        log.info("screenshot.saved", path=str(path), size=(pix.width(), pix.height()))
        get_bus().publish(EventType.SCREENSHOT_TAKEN, str(path))
        return path

    def capture_window(self, widget: QWidget, name: str) -> Path:
        """Capture into ``docs/images/<name>.png``."""
        if not name.endswith(".png"):
            name = f"{name}.png"
        return self.capture(widget, docs_image_dir() / name)
