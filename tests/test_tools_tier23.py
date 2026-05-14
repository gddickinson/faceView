"""Tier 2 + 3 on-demand tools: color, pose, face_attrs, qr, depth,
gaze_target, segment_object — schema + dispatch coverage.

The model-heavy executors (pose / depth / clip / ocr) are checked
end-to-end where cheap (color, qr), and via no-frame fallback when
they'd otherwise load multi-hundred-MB models we don't want to pull
in CI.
"""

from __future__ import annotations

import threading

import numpy as np


# ── schema bundles ───────────────────────────────────────────────────────


def test_all_tier23_tools_have_schemas():
    from faceview.llm.vision_tool import (
        TIER23_TOOLS_ANTHROPIC, TIER23_TOOLS_OLLAMA,
    )
    names = {t["name"] for t in TIER23_TOOLS_ANTHROPIC}
    expected = {
        "describe_color", "describe_pose", "face_attributes",
        "scan_qr", "estimate_depth", "gaze_target",
        "segment_object",
    }
    assert expected.issubset(names), names

    ollama_names = {t["function"]["name"] for t in TIER23_TOOLS_OLLAMA}
    assert ollama_names == names


def test_segment_object_schema_requires_label():
    from faceview.llm.vision_tool import SEGMENT_OBJECT_TOOL_ANTHROPIC
    assert SEGMENT_OBJECT_TOOL_ANTHROPIC["input_schema"]["required"] == ["label"]


# ── pure-cv2 executors: color + qr ───────────────────────────────────────


class _Grabber:
    """Stand-in FrameGrabber for tool-helper tests."""

    def __init__(self, frame):
        self._lock = threading.Lock()
        self._latest = frame


def test_describe_color_runs_end_to_end():
    from faceview.llm.vision_tool import run_describe_color
    # Build a solid-red image — every cluster should land on red.
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :, 2] = 200  # BGR red channel
    msg = run_describe_color(_Grabber(frame), region="full")
    assert "red" in msg.lower()


def test_describe_color_handles_no_frame():
    from faceview.llm.vision_tool import run_describe_color
    msg = run_describe_color(_Grabber(None))
    assert "No camera frame" in msg


def test_scan_qr_handles_no_frame():
    from faceview.llm.vision_tool import run_scan_qr
    msg = run_scan_qr(_Grabber(None))
    assert "No camera frame" in msg


def test_scan_qr_runs_on_blank_returns_none_found():
    from faceview.llm.vision_tool import run_scan_qr
    msg = run_scan_qr(_Grabber(np.zeros((40, 40, 3), dtype=np.uint8)))
    assert "don't see" in msg.lower() or "no" in msg.lower() \
        or "readable" in msg.lower()


# ── face attributes — needs IdentityRecognizer registered ────────────────


def test_face_attributes_without_recognizer_running(monkeypatch):
    import faceview.vision.face_attr as fa
    monkeypatch.setitem(fa._APP_HANDLE, "app", None)
    from faceview.llm.vision_tool import run_face_attributes
    msg = run_face_attributes(_Grabber(np.zeros((4, 4, 3), dtype=np.uint8)))
    assert "identity recognizer" in msg.lower()


# ── gaze_target — pure heuristic over PerceptionStore ────────────────────


def test_gaze_target_reports_no_face(fresh_bus, monkeypatch):
    import faceview.vision.perception as p
    monkeypatch.setattr(p.PerceptionStore, "_instance", None)
    p.PerceptionStore.shared()  # subscribe
    from faceview.core.events import EventType, Presence
    fresh_bus.publish(EventType.PRESENCE, Presence(face_count=0, bboxes=[]))
    from faceview.llm.vision_tool import run_gaze_target
    msg = run_gaze_target()
    assert "don't see a face" in msg.lower() or "no face" in msg.lower()


def test_gaze_target_combines_iris_and_head(fresh_bus, monkeypatch):
    import faceview.vision.perception as p
    monkeypatch.setattr(p.PerceptionStore, "_instance", None)
    p.PerceptionStore.shared()
    from faceview.core.events import (
        EventType, Gaze, HeadPose, Presence,
    )
    fresh_bus.publish(EventType.PRESENCE, Presence(face_count=1, bboxes=[]))
    fresh_bus.publish(EventType.HEAD_POSE,
                      HeadPose(yaw=0.0, pitch=0.0, roll=0.0))
    fresh_bus.publish(EventType.GAZE,
                      Gaze(direction="camera", yaw=0.0, pitch=0.0,
                           attention=0.9))
    from faceview.llm.vision_tool import run_gaze_target
    msg = run_gaze_target()
    assert "camera" in msg.lower()


# ── segment_object — graceful fallback when no detection on file ────────


def test_segment_object_no_detection(fresh_bus, monkeypatch):
    import faceview.vision.perception as p
    monkeypatch.setattr(p.PerceptionStore, "_instance", None)
    p.PerceptionStore.shared()
    from faceview.llm.vision_tool import run_segment_object
    msg = run_segment_object(
        _Grabber(np.zeros((50, 50, 3), dtype=np.uint8)), label="cup",
    )
    assert "don't currently see" in msg.lower() \
        or "objects list" in msg.lower()


# ── pose / depth — lazy-load: just verify the helpers return strings ────


def test_describe_pose_no_frame():
    from faceview.llm.vision_tool import run_describe_pose
    msg = run_describe_pose(_Grabber(None))
    assert "No camera frame" in msg


def test_estimate_depth_no_frame():
    from faceview.llm.vision_tool import run_estimate_depth
    msg = run_estimate_depth(_Grabber(None))
    assert "No camera frame" in msg
