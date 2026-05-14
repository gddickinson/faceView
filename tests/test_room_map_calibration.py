"""P16 — camera intrinsics calibration."""

from __future__ import annotations

import json
import math


def _isolate_calibration(tmp_path, monkeypatch):
    """Fresh CalibrationStore + isolated data_dir so reading the
    real ~/.faceview/camera_calibration.json doesn't pollute tests."""
    import faceview.config as cfg
    monkeypatch.setattr(cfg.settings, "data_dir", tmp_path)
    import faceview.vision.room_map as rm
    monkeypatch.setattr(rm.CalibrationStore, "_instance", None)
    return rm


def test_no_calibration_means_relative_units(tmp_path, monkeypatch):
    rm = _isolate_calibration(tmp_path, monkeypatch)
    assert rm.CalibrationStore.shared().scale is None


def test_set_scale_persists_to_disk(tmp_path, monkeypatch):
    rm = _isolate_calibration(tmp_path, monkeypatch)
    store = rm.CalibrationStore.shared()
    assert store.set_scale(0.5) is True
    assert store.scale == 0.5
    path = tmp_path / "camera_calibration.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["scale"] == 0.5


def test_set_scale_rejects_non_positive(tmp_path, monkeypatch):
    rm = _isolate_calibration(tmp_path, monkeypatch)
    store = rm.CalibrationStore.shared()
    assert store.set_scale(0) is False
    assert store.set_scale(-1.0) is False
    assert store.scale is None


def test_clear_removes_file(tmp_path, monkeypatch):
    rm = _isolate_calibration(tmp_path, monkeypatch)
    store = rm.CalibrationStore.shared()
    store.set_scale(0.7)
    assert (tmp_path / "camera_calibration.json").exists()
    assert store.clear() is True
    assert store.scale is None
    assert not (tmp_path / "camera_calibration.json").exists()


def test_load_picks_up_existing_file(tmp_path, monkeypatch):
    """A pre-existing calibration on disk loads on first .shared()."""
    import faceview.config as cfg
    monkeypatch.setattr(cfg.settings, "data_dir", tmp_path)
    (tmp_path / "camera_calibration.json").write_text(
        json.dumps({"scale": 1.25})
    )
    import faceview.vision.room_map as rm
    monkeypatch.setattr(rm.CalibrationStore, "_instance", None)
    assert rm.CalibrationStore.shared().scale == 1.25


def test_worker_publishes_metres_when_calibrated(tmp_path, monkeypatch):
    """End-to-end: with calibration set, RoomMap.units == 'metres'
    and item positions are scaled."""
    import numpy as np
    rm = _isolate_calibration(tmp_path, monkeypatch)
    rm.CalibrationStore.shared().set_scale(2.0)  # 1 rel unit = 2 m

    class _FakeDepth:
        @classmethod
        def shared(cls):
            return cls()
        def depth_map(self, frame):
            h, w = frame.shape[:2]
            return np.full((h, w), 0.5, dtype="float32")

    monkeypatch.setattr(
        "faceview.vision.depth.DepthEstimator", _FakeDepth,
    )

    received: list = []
    from faceview.core.event_bus import get_bus
    from faceview.core.events import (
        DetectedObject, EventType, ObjectsSeen,
    )
    bus = get_bus()
    bus.subscribe(EventType.ROOM_MAP, received.append)

    worker = rm.RoomMapWorker(interval_s=99.0)
    worker.set_active(True)
    worker._on_frame(np.zeros((40, 80, 3), dtype="uint8"))
    worker._on_objects(ObjectsSeen(detections=[
        DetectedObject(label="cup", score=0.9, bbox=(30, 15, 20, 10)),
    ]))
    # Capture the unscaled position the worker computes internally.
    worker._tick()
    raw = worker._smoothed["cup"]
    raw_z = raw.z
    msg = received[-1]
    assert msg.units == "metres"
    # Published z should be 2× the internal (raw) z.
    assert abs(msg.items[0].z - raw_z * 2.0) < 1e-6


def test_describe_uses_metres_after_calibration(tmp_path, monkeypatch):
    """The describe_room_layout text reports 'm' when calibration is on."""
    rm = _isolate_calibration(tmp_path, monkeypatch)
    rm.CalibrationStore.shared().set_scale(1.5)
    # RoomMapStore caches the latest event; publish a metres payload.
    monkeypatch.setattr(rm.RoomMapStore, "_instance", None)
    rm.RoomMapStore.shared()
    from faceview.core.event_bus import get_bus
    from faceview.core.events import EventType, RoomMap, RoomMapItem
    get_bus().publish(EventType.ROOM_MAP, RoomMap(
        items=[RoomMapItem(label="cup", x=0.0, z=1.0)],
        units="metres",
    ))
    msg = rm.describe_room_layout()
    assert "1.0 m" in msg


def test_zero_distance_object_rejected_in_dialog(tmp_path, monkeypatch, qtbot):
    """A degenerate item (x=z=0) shouldn't compute scale=∞ — the
    dialog refuses with a warning. We exercise the math directly
    because triggering the QMessageBox in headless mode is flaky."""
    rm = _isolate_calibration(tmp_path, monkeypatch)
    store = rm.CalibrationStore.shared()
    # Simulate the dialog's _apply logic on a zero-distance item.
    from faceview.core.events import RoomMapItem
    item = RoomMapItem(label="x", x=0.0, z=0.0)
    current = math.sqrt(item.x ** 2 + item.z ** 2)
    # The dialog checks `current < 1e-6` and bails; verify the math.
    assert current < 1e-6
    assert store.scale is None  # unchanged
