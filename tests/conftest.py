"""Shared pytest fixtures.

We force the offscreen Qt platform here so tests work in headless CI and on
machines without a display server.
"""

from __future__ import annotations

import os
import sys

# Must be set before any PySide6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FACEVIEW_HEADLESS", "1")

import pytest


@pytest.fixture
def fresh_bus(monkeypatch):
    """Reset the singleton EventBus between tests so subscribers don't leak."""
    from faceview.core import event_bus as eb
    monkeypatch.setattr(eb, "_bus", None)
    yield eb.get_bus()
    monkeypatch.setattr(eb, "_bus", None)


@pytest.fixture
def app(qtbot):
    """A QApplication for tests that need one. pytest-qt provides ``qtbot``."""
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)
