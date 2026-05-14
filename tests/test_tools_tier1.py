"""Tier-1 on-demand vision tools: OCR, tracker, CLIP open-vocab check."""

from __future__ import annotations

import numpy as np

from faceview.core.events import DetectedObject, EventType, ObjectsSeen


# ── Object tracker ───────────────────────────────────────────────────────


def _reset_tracker(monkeypatch):
    import faceview.vision.tracker as t
    monkeypatch.setattr(t.ObjectTracker, "_instance", None)
    return t


def test_iou_basics():
    from faceview.vision.tracker import _iou
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    half = _iou((0, 0, 10, 10), (5, 0, 10, 10))
    assert 0.3 < half < 0.4  # 50/150 = 0.333


def test_zone_labels():
    from faceview.vision.tracker import _zone
    assert _zone((10, 10, 20, 20), 100, 100) == "top-left"
    assert _zone((40, 40, 20, 20), 100, 100) == "center"
    assert _zone((75, 75, 20, 20), 100, 100) == "bottom-right"
    assert _zone((40, 75, 20, 20), 100, 100) == "bottom"


def test_start_tracking_fails_without_detection(fresh_bus, monkeypatch):
    t = _reset_tracker(monkeypatch)
    tracker = t.ObjectTracker.shared()
    ok, msg = tracker.start_tracking("cup", duration_s=5)
    assert ok is False
    assert "don't currently see" in msg


def test_start_tracking_then_update_keeps_position(fresh_bus, monkeypatch):
    t = _reset_tracker(monkeypatch)
    tracker = t.ObjectTracker.shared()
    fresh_bus.publish(
        EventType.OBJECTS,
        ObjectsSeen(detections=[
            DetectedObject(label="cup", score=0.9, bbox=(10, 10, 50, 50)),
        ]),
    )
    ok, _ = tracker.start_tracking("cup", duration_s=5)
    assert ok is True
    assert "cup" in tracker.narrate()

    # New OBJECTS event with the cup moved a bit — IoU still high so
    # we should re-anchor without losing the tracker.
    fresh_bus.publish(
        EventType.OBJECTS,
        ObjectsSeen(detections=[
            DetectedObject(label="cup", score=0.9, bbox=(20, 12, 50, 50)),
        ]),
    )
    snap = tracker.status_dict()
    assert len(snap["tracks"]) == 1
    assert snap["tracks"][0]["bbox"][0] == 20


def test_tracker_marks_lost_when_no_more_detection(fresh_bus, monkeypatch):
    import time
    t = _reset_tracker(monkeypatch)
    tracker = t.ObjectTracker.shared()
    fresh_bus.publish(
        EventType.OBJECTS,
        ObjectsSeen(detections=[
            DetectedObject(label="cup", score=0.9, bbox=(10, 10, 50, 50)),
        ]),
    )
    tracker.start_tracking("cup", duration_s=60)
    # Publish a detection that doesn't include cup
    fresh_bus.publish(
        EventType.OBJECTS,
        ObjectsSeen(detections=[
            DetectedObject(label="book", score=0.9, bbox=(0, 0, 30, 30)),
        ]),
    )
    snap = tracker.status_dict()
    assert snap["tracks"] and snap["tracks"][0]["lost"] is True


def test_tracker_expires_after_duration(fresh_bus, monkeypatch):
    import time
    t = _reset_tracker(monkeypatch)
    tracker = t.ObjectTracker.shared()
    fresh_bus.publish(
        EventType.OBJECTS,
        ObjectsSeen(detections=[
            DetectedObject(label="cup", score=0.9, bbox=(10, 10, 50, 50)),
        ]),
    )
    tracker.start_tracking("cup", duration_s=2)
    # Back-date the start so it's already expired.
    with tracker._lock:
        tracker._tracks["cup"].expires_at = time.time() - 1
    # Publish another OBJECTS event — should garbage-collect expired.
    fresh_bus.publish(
        EventType.OBJECTS,
        ObjectsSeen(detections=[
            DetectedObject(label="cup", score=0.9, bbox=(10, 10, 50, 50)),
        ]),
    )
    assert tracker.status_dict()["tracks"] == []


# ── Tool schemas + dispatch wiring ───────────────────────────────────────


def test_new_tool_schemas_present():
    from faceview.llm.vision_tool import (
        READ_TEXT_TOOL_ANTHROPIC, READ_TEXT_TOOL_OLLAMA,
        TRACK_OBJECT_TOOL_ANTHROPIC, TRACK_OBJECT_TOOL_OLLAMA,
        CHECK_VISIBLE_TOOL_ANTHROPIC, CHECK_VISIBLE_TOOL_OLLAMA,
    )
    assert READ_TEXT_TOOL_ANTHROPIC["name"] == "read_text"
    assert TRACK_OBJECT_TOOL_ANTHROPIC["input_schema"]["required"] == ["label"]
    assert CHECK_VISIBLE_TOOL_ANTHROPIC["input_schema"]["required"] == ["query"]
    assert READ_TEXT_TOOL_OLLAMA["type"] == "function"


def test_run_track_object_routes_to_singleton(fresh_bus, monkeypatch):
    """The tool helper drives the same ObjectTracker as the perception
    narrator — so a track started via the tool surfaces in narrate."""
    t = _reset_tracker(monkeypatch)
    import faceview.vision.perception as p
    monkeypatch.setattr(p.PerceptionStore, "_instance", None)
    # Instantiate both singletons BEFORE publishing so their bus
    # subscriptions are in place.
    tracker = t.ObjectTracker.shared()
    store = p.PerceptionStore.shared()
    fresh_bus.publish(
        EventType.OBJECTS,
        ObjectsSeen(detections=[
            DetectedObject(label="cup", score=0.9, bbox=(10, 10, 50, 50)),
        ]),
    )
    from faceview.llm.vision_tool import run_track_object
    msg = run_track_object("cup", duration_s=5)
    assert "tracking 'cup'" in msg.lower()
    # Now the perception narrator should mention it.
    from faceview.core.events import Presence
    fresh_bus.publish(EventType.PRESENCE,
                      Presence(face_count=1, bboxes=[]))
    assert "tracking" in store.narrate_now()


def test_read_text_no_frame_returns_helpful():
    """When no frame is cached, the helper falls back cleanly."""
    import faceview.llm.vision_tool as vt
    # Hand-roll a grabber-like object with no frame
    class _G:
        _lock = __import__('threading').Lock()
        _latest = None
    msg = vt.run_read_text(_G(), region="full")
    assert "No camera frame" in msg


def test_check_visible_validates_query():
    import faceview.llm.vision_tool as vt
    class _G:
        _lock = __import__('threading').Lock()
        _latest = np.zeros((10, 10, 3), dtype=np.uint8)
    msg = vt.run_check_visible(_G(), query="", region="full")
    assert "I need a query" in msg
