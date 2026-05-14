"""IoU-based object tracker.

The :class:`ObjectDetector` worker (``vision/objects.py``) emits
``OBJECTS`` events at ~3 Hz with the current frame's detections. This
module sits on top: when the LLM calls ``track_object(label, …)`` we
start a short-lived tracker for that label, then on every subsequent
``OBJECTS`` event we re-anchor to whichever detection of the same
class has the best IoU overlap with the previous bbox.

When the IoU drops to zero for too long the tracker reports "lost".
After ``duration_s`` the tracker auto-expires.

Why not OpenCV's CSRT / KCF trackers? They require the contrib build
and they re-run on every FRAME (heavy). The IoU-vs-detector approach
piggy-backs on the existing detector cadence — zero extra CPU per
frame — and still answers the questions the LLM cares about: *"is X
still there, did it move, where is it now."*

The tracker state is surfaced two ways:

* :meth:`narrate` is read by :class:`PerceptionStore` so the LLM sees
  active trackers in its system prompt every turn.
* :meth:`status_dict` is read by the GUI debug panel.

Singleton: :meth:`ObjectTracker.shared` so the LLM tool, the panel,
and the perception narrator all share the same trackers list.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, ObjectsSeen
from faceview.core.logger import get_logger


log = get_logger("tracker")


# IoU below this is treated as "no overlap, target lost on this frame".
_LOST_IOU = 0.05
# Seconds of "lost" before the tracker reports the object as gone (we
# keep the slot for a bit so a brief occlusion doesn't kill it).
_LOST_GRACE_S = 3.0


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Standard IoU for (x, y, w, h) tuples."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix)
    ih = max(0, iy2 - iy)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def _zone(bbox: tuple[int, int, int, int], frame_w: int = 1, frame_h: int = 1) -> str:
    """Coarse 9-zone label for the bbox centre, frame-relative."""
    x, y, w, h = bbox
    cx = (x + w / 2) / max(1, frame_w)
    cy = (y + h / 2) / max(1, frame_h)
    col = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
    row = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
    if col == "center" and row == "middle":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


class _Track:
    __slots__ = (
        "label", "bbox", "start_bbox", "started_at", "expires_at",
        "last_seen_at", "history", "lost_since",
    )

    def __init__(
        self, label: str, bbox: tuple[int, int, int, int],
        duration_s: float,
    ) -> None:
        now = time.time()
        self.label = label
        self.bbox = bbox
        self.start_bbox = bbox
        self.started_at = now
        self.expires_at = now + duration_s
        self.last_seen_at = now
        self.history: list[tuple[int, int, int, int]] = [bbox]
        self.lost_since: Optional[float] = None

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def is_lost(self, now: float) -> bool:
        return (
            self.lost_since is not None
            and now - self.lost_since > _LOST_GRACE_S
        )


class ObjectTracker:
    """Singleton: maintains per-label trackers updated on OBJECTS events."""

    _instance: "ObjectTracker | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "ObjectTracker":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ObjectTracker()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tracks: dict[str, _Track] = {}
        self._last_detections: list = []
        self._frame_size: tuple[int, int] = (1, 1)
        get_bus().subscribe(EventType.OBJECTS, self._on_objects)
        get_bus().subscribe(EventType.FRAME, self._on_frame)

    # ── bus handlers ────────────────────────────────────────────────

    def _on_objects(self, payload: ObjectsSeen) -> None:
        if payload is None:
            return
        with self._lock:
            self._last_detections = list(payload.detections)
            now = time.time()
            for key, track in list(self._tracks.items()):
                # Expiry
                if track.is_expired(now):
                    del self._tracks[key]
                    continue
                best_det = None
                best_iou = 0.0
                for det in payload.detections:
                    if det.label.lower() != track.label.lower():
                        continue
                    iou = _iou(track.bbox, det.bbox)
                    if iou > best_iou:
                        best_iou, best_det = iou, det
                if best_det is not None and best_iou >= _LOST_IOU:
                    track.bbox = best_det.bbox
                    track.last_seen_at = now
                    track.lost_since = None
                    track.history.append(best_det.bbox)
                    if len(track.history) > 60:
                        track.history.pop(0)
                else:
                    if track.lost_since is None:
                        track.lost_since = now
                    if track.is_lost(now):
                        del self._tracks[key]
                        log.info("tracker.lost", label=track.label)

    def _on_frame(self, frame) -> None:
        if frame is None:
            return
        h, w = frame.shape[:2]
        self._frame_size = (w, h)

    # ── start / stop ────────────────────────────────────────────────

    def start_tracking(
        self, label: str, duration_s: float = 10.0,
    ) -> tuple[bool, str]:
        """Seed a tracker from the most recent OBJECTS detection."""
        key = label.lower().strip()
        if not key:
            return False, "I need an object label to track."
        duration_s = max(2.0, min(60.0, float(duration_s)))
        with self._lock:
            # If already tracking this label, just refresh duration.
            if key in self._tracks:
                t = self._tracks[key]
                t.expires_at = time.time() + duration_s
                return True, (f"Already tracking '{label}' — refreshed "
                              f"for another {duration_s:.0f} s.")
            # Find best detection matching the label.
            candidates = [
                d for d in self._last_detections
                if d.label.lower() == key
            ]
            if not candidates:
                return False, (f"I don't currently see a '{label}' in "
                               "the frame to start tracking.")
            # Pick the detection with the largest score.
            candidates.sort(key=lambda d: -d.score)
            seed = candidates[0]
            self._tracks[key] = _Track(seed.label, seed.bbox, duration_s)
        log.info("tracker.started", label=label,
                 duration_s=duration_s, bbox=seed.bbox)
        return True, (f"Started tracking '{label}' at the "
                      f"{_zone(seed.bbox, *self._frame_size)} of the "
                      f"frame for the next {duration_s:.0f} seconds.")

    def stop_tracking(self, label: str) -> bool:
        with self._lock:
            return self._tracks.pop(label.lower().strip(), None) is not None

    # ── reads ───────────────────────────────────────────────────────

    def status_dict(self) -> dict:
        now = time.time()
        with self._lock:
            tracks = []
            fw, fh = self._frame_size
            for t in self._tracks.values():
                tracks.append({
                    "label": t.label,
                    "bbox": list(t.bbox),
                    "zone": _zone(t.bbox, fw, fh),
                    "age_s": round(now - t.started_at, 1),
                    "remaining_s": round(max(0.0, t.expires_at - now), 1),
                    "lost": t.lost_since is not None,
                })
        return {"tracks": tracks}

    def narrate(self) -> str:
        """One short line for the system-prompt perception block."""
        snap = self.status_dict()
        tracks = snap["tracks"]
        if not tracks:
            return ""
        parts = []
        for t in tracks:
            tag = "lost" if t["lost"] else t["zone"]
            parts.append(
                f"{t['label']} ({tag}, {t['remaining_s']:.0f} s left)"
            )
        return "tracking: " + "; ".join(parts)
