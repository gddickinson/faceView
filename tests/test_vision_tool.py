"""Coverage for the LLM ``look_at_camera`` tool helpers."""

from __future__ import annotations


def test_tool_schemas_have_expected_shape():
    from faceview.llm.vision_tool import (
        LOOK_TOOL_ANTHROPIC, LOOK_TOOL_OLLAMA,
    )
    # Anthropic flat schema
    assert LOOK_TOOL_ANTHROPIC["name"] == "look_at_camera"
    assert "description" in LOOK_TOOL_ANTHROPIC
    assert LOOK_TOOL_ANTHROPIC["input_schema"]["type"] == "object"

    # Ollama function schema
    assert LOOK_TOOL_OLLAMA["type"] == "function"
    fn = LOOK_TOOL_OLLAMA["function"]
    assert fn["name"] == "look_at_camera"
    assert fn["parameters"]["type"] == "object"


def test_vision_tool_enabled_default_on(monkeypatch):
    from faceview.llm.vision_tool import vision_tool_enabled
    monkeypatch.delenv("FACEVIEW_VISION_TOOL", raising=False)
    assert vision_tool_enabled() is True


def test_vision_tool_enabled_disabled(monkeypatch):
    from faceview.llm.vision_tool import vision_tool_enabled
    monkeypatch.setenv("FACEVIEW_VISION_TOOL", "0")
    assert vision_tool_enabled() is False


def test_frame_grabber_caches_frame(fresh_bus, monkeypatch):
    # Force a brand-new grabber so the fresh_bus fixture's reset doesn't
    # leave a stale singleton subscribed to a defunct bus instance.
    import faceview.llm.vision_tool as vt
    monkeypatch.setattr(vt.FrameGrabber, "_instance", None)
    g = vt.FrameGrabber.shared()
    assert not g.have_frame()

    import numpy as np
    fake_frame = np.zeros((48, 64, 3), dtype=np.uint8)
    fresh_bus.publish(vt.EventType.FRAME, fake_frame)
    # Direct dispatch in same-thread Qt connection → cache should be filled.
    assert g.have_frame()


def test_run_look_anthropic_handles_missing_frame(fresh_bus, monkeypatch):
    """When no FRAME has arrived, the tool returns an apologetic text
    block (no image) so the model can still respond cleanly."""
    import faceview.llm.vision_tool as vt
    monkeypatch.setattr(vt.FrameGrabber, "_instance", None)
    g = vt.FrameGrabber.shared()
    content = vt.run_look_anthropic(g)
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "No camera frame" in content[0]["text"]


def test_run_look_anthropic_attaches_image(fresh_bus, monkeypatch):
    """With a frame on the bus, the executor returns an image block + note."""
    import faceview.llm.vision_tool as vt
    monkeypatch.setattr(vt.FrameGrabber, "_instance", None)
    g = vt.FrameGrabber.shared()

    try:
        import cv2  # type: ignore  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("cv2 not installed in this env — JPEG encode skipped")

    import numpy as np
    frame = (np.random.rand(80, 100, 3) * 255).astype(np.uint8)
    fresh_bus.publish(vt.EventType.FRAME, frame)
    content = vt.run_look_anthropic(g)
    assert any(b.get("type") == "image" for b in content)
    img = next(b for b in content if b["type"] == "image")
    assert img["source"]["media_type"] == "image/jpeg"
    assert isinstance(img["source"]["data"], str) and len(img["source"]["data"]) > 100


def test_avatar_frame_falls_back_only_when_no_camera(fresh_bus, monkeypatch):
    """AVATAR_FRAME should fill the cache only when no real FRAME is recent."""
    import time

    import faceview.llm.vision_tool as vt
    monkeypatch.setattr(vt.FrameGrabber, "_instance", None)
    g = vt.FrameGrabber.shared()

    import numpy as np
    cam = np.zeros((10, 10, 3), dtype=np.uint8)
    avatar = np.full((10, 10, 3), 7, dtype=np.uint8)

    fresh_bus.publish(vt.EventType.FRAME, cam)
    # Avatar frame published right after — camera is still "recent",
    # so the cache should keep the camera frame.
    fresh_bus.publish(vt.EventType.AVATAR_FRAME, avatar)
    assert g._latest_source == "camera"

    # Simulate the camera going stale, then publish avatar again.
    g._latest_ts = time.time() - 5.0
    fresh_bus.publish(vt.EventType.AVATAR_FRAME, avatar)
    assert g._latest_source == "avatar"
