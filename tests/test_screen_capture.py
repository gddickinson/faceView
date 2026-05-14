"""P10 — screen-capture worker + look_at_screen tool."""

from __future__ import annotations

import numpy as np


def test_screen_frame_grabber_singleton(fresh_bus, monkeypatch):
    import faceview.vision.screen as sc
    monkeypatch.setattr(sc.ScreenFrameGrabber, "_instance", None)
    a = sc.ScreenFrameGrabber.shared()
    b = sc.ScreenFrameGrabber.shared()
    assert a is b


def test_screen_grabber_caches_published_frame(fresh_bus, monkeypatch):
    import faceview.vision.screen as sc
    monkeypatch.setattr(sc.ScreenFrameGrabber, "_instance", None)
    g = sc.ScreenFrameGrabber.shared()
    assert not g.have_frame()
    from faceview.core.events import EventType
    fresh_bus.publish(
        EventType.SCREEN_FRAME,
        np.zeros((100, 200, 3), dtype="uint8"),
    )
    assert g.have_frame()


def test_screen_grabber_jpeg_resize(fresh_bus, monkeypatch):
    import faceview.vision.screen as sc
    monkeypatch.setattr(sc.ScreenFrameGrabber, "_instance", None)
    g = sc.ScreenFrameGrabber.shared()
    from faceview.core.events import EventType
    # Large frame — encoder should down-scale.
    fresh_bus.publish(
        EventType.SCREEN_FRAME,
        (np.random.rand(2000, 3000, 3) * 255).astype("uint8"),
    )
    pair = g.latest_jpeg_b64(max_dim=512)
    assert pair is not None
    b64, source = pair
    assert source == "screen"
    assert isinstance(b64, str) and len(b64) > 100


def test_look_at_screen_tool_schemas_present():
    from faceview.llm.vision_tool import (
        LOOK_AT_SCREEN_TOOL_ANTHROPIC,
        LOOK_AT_SCREEN_TOOL_OLLAMA,
        TIER23_TOOLS_ANTHROPIC, TIER23_TOOLS_OLLAMA,
    )
    assert LOOK_AT_SCREEN_TOOL_ANTHROPIC["name"] == "look_at_screen"
    assert (LOOK_AT_SCREEN_TOOL_OLLAMA["function"]["name"]
            == "look_at_screen")
    a_names = {t["name"] for t in TIER23_TOOLS_ANTHROPIC}
    o_names = {t["function"]["name"] for t in TIER23_TOOLS_OLLAMA}
    assert "look_at_screen" in a_names
    assert "look_at_screen" in o_names


def test_run_look_at_screen_anthropic_no_frame(fresh_bus, monkeypatch):
    import faceview.vision.screen as sc
    monkeypatch.setattr(sc.ScreenFrameGrabber, "_instance", None)
    sc.ScreenFrameGrabber.shared()
    from faceview.llm.vision_tool import run_look_at_screen_anthropic
    content = run_look_at_screen_anthropic()
    # When no frame has been seen, returns a text block apologetically.
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "screen-capture" in content[0]["text"] or "View" in content[0]["text"]


def test_run_look_at_screen_anthropic_attaches_image(fresh_bus, monkeypatch):
    import faceview.vision.screen as sc
    monkeypatch.setattr(sc.ScreenFrameGrabber, "_instance", None)
    g = sc.ScreenFrameGrabber.shared()
    from faceview.core.events import EventType
    try:
        import cv2  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("cv2 not installed")
    fresh_bus.publish(
        EventType.SCREEN_FRAME,
        (np.random.rand(120, 200, 3) * 255).astype("uint8"),
    )
    from faceview.llm.vision_tool import run_look_at_screen_anthropic
    content = run_look_at_screen_anthropic(question="what's here?",
                                            region="top")
    # Image content block.
    assert any(b.get("type") == "image" for b in content)
    note = next(b for b in content if b.get("type") == "text")
    assert "screen snapshot" in note["text"]
    assert "top" in note["text"]
    assert "what's here" in note["text"]


def test_screen_capture_worker_lifecycle_no_mss(monkeypatch):
    """If mss isn't installed, ScreenCaptureWorker.start raises a
    MissingDependency rather than crashing the GUI."""
    import sys
    # Hide mss from import.
    monkeypatch.setitem(sys.modules, "mss", None)
    from faceview.core.errors import MissingDependency
    from faceview.vision.screen import ScreenCaptureWorker
    w = ScreenCaptureWorker()
    import pytest
    with pytest.raises(MissingDependency):
        w.start()


def test_main_window_facade_toggles_capture(qtbot, monkeypatch):
    """MainWindow.set_screen_capture_enabled(True) starts the worker;
    (False) stops it. We patch ScreenCaptureWorker to avoid grabbing
    the real screen in CI."""
    import faceview.vision.screen as sc

    class _FakeWorker:
        def __init__(self): self._running = False
        def start(self): self._running = True; return True
        def stop(self): self._running = False
        def is_running(self): return self._running

    monkeypatch.setattr(sc, "ScreenCaptureWorker", _FakeWorker)
    from faceview.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.screen_capture_running() is False
    w.set_screen_capture_enabled(True)
    assert w.screen_capture_running() is True
    w.set_screen_capture_enabled(False)
    assert w.screen_capture_running() is False
