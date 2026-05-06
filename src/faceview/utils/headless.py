"""Helpers for running the GUI without a visible display.

The Qt offscreen platform plugin renders to a hidden buffer that
``QWidget.grab()`` can still read from — exactly what we want for CI smoke
tests and README screenshot capture.
"""

from __future__ import annotations

import os


def enable_offscreen() -> None:
    """Set ``QT_QPA_PLATFORM=offscreen`` if not already set.

    Must be called BEFORE importing PySide6 or constructing ``QApplication``.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def is_offscreen() -> bool:
    return os.environ.get("QT_QPA_PLATFORM") == "offscreen"
