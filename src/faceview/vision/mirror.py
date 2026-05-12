"""MirrorState — translate live user-vision events into avatar FaceParams.

When mirror mode is on, Claude's avatar mimics the user's expression
and mouth movement live. The translation lives here so the GUI can
flip the mode on/off without touching the renderer.

Inputs (from the bus):

- :data:`EventType.MOUTH_ACTIVITY` — jaw_open + viseme (drives mouth)
- :data:`EventType.EMOTION` — happy/sad/surprise/etc. (drives baseline AUs)
- :data:`EventType.PRESENCE` — when the user steps away, fall back to idle

Output: a :class:`FaceParams` returned by :meth:`face_params` that the
avatar's :class:`SimCameraWorker` can use as the override for its tick.
"""

from __future__ import annotations

import time
from typing import Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import Emotion, EventType, HeadPose, MouthActivity, Presence
from faceview.vision.sim_face import FaceParams


_EMOTION_PARAMS = {
    "happy":    dict(smile=0.85, brow_raise=0.15, eye_open=0.95),
    "neutral":  dict(smile=0.05, brow_raise=0.0,  eye_open=1.0),
    "sad":      dict(smile=-0.4, brow_raise=-0.2, eye_open=0.7),
    "surprise": dict(smile=0.1,  brow_raise=0.9,  eye_open=1.0, jaw_open=0.25),
    "angry":    dict(smile=-0.3, brow_raise=-0.7, eye_open=1.0),
    "fear":     dict(smile=-0.2, brow_raise=0.8,  eye_open=1.0),
    "disgust":  dict(smile=-0.3, brow_raise=-0.4, eye_open=0.9),
}


class MirrorState:
    """Aggregates the latest vision events into a single FaceParams."""

    def __init__(self) -> None:
        self.active = False
        self._latest_emotion: Optional[Emotion] = None
        self._latest_mouth: Optional[MouthActivity] = None
        self._latest_presence: Optional[Presence] = None
        self._latest_head: Optional[HeadPose] = None
        self._last_mouth_ts: float = 0.0
        self._last_emotion_ts: float = 0.0
        self._last_head_ts: float = 0.0

    def attach_bus(self) -> None:
        bus = get_bus()
        bus.subscribe(EventType.EMOTION, self._on_emotion)
        bus.subscribe(EventType.MOUTH_ACTIVITY, self._on_mouth)
        bus.subscribe(EventType.PRESENCE, self._on_presence)
        bus.subscribe(EventType.HEAD_POSE, self._on_head_pose)

    def _on_head_pose(self, h: HeadPose) -> None:
        self._latest_head = h
        self._last_head_ts = time.time()

    def _on_emotion(self, e: Emotion) -> None:
        self._latest_emotion = e
        self._last_emotion_ts = time.time()

    def _on_mouth(self, m: MouthActivity) -> None:
        self._latest_mouth = m
        self._last_mouth_ts = time.time()

    def _on_presence(self, p: Presence) -> None:
        self._latest_presence = p

    # ── translation ─────────────────────────────────────────────────

    def face_params(self, _t: float) -> Optional[FaceParams]:
        """Compose a FaceParams from the most recent vision events.

        Returns ``None`` if the data is too stale to be useful (avatar
        falls back to its idle behaviour in that case).
        """
        if not self.active:
            return None
        now = time.time()
        # If both signals are very stale, hand control back to the
        # avatar's idle so it doesn't freeze on the last expression.
        emotion_fresh = (now - self._last_emotion_ts) < 5.0
        mouth_fresh = (now - self._last_mouth_ts) < 1.0
        if not (emotion_fresh or mouth_fresh):
            return None

        params = FaceParams.neutral()
        if emotion_fresh and self._latest_emotion is not None:
            preset = _EMOTION_PARAMS.get(self._latest_emotion.label, _EMOTION_PARAMS["neutral"])
            for k, v in preset.items():
                setattr(params, k, float(v))
        if mouth_fresh and self._latest_mouth is not None:
            m = self._latest_mouth
            # Map normalised face-mesh jaw_open (typically 0..0.4)
            # to FaceParams jaw_open (0..0.6).
            params.jaw_open = max(params.jaw_open, min(0.6, m.jaw_open * 1.5))
            if m.viseme in ("EE",):
                params.smile = max(params.smile, 0.35)
            elif m.viseme in ("OO",):
                params.smile = min(params.smile, -0.1)
                params.jaw_open = max(params.jaw_open, 0.18)

        # Head pose mirrors the user's yaw / pitch / roll. Stale data
        # (no face for >1 s) lets the head drift back to neutral
        # rather than freezing in a tilted pose.
        head_fresh = (now - self._last_head_ts) < 1.0
        if head_fresh and self._latest_head is not None:
            h = self._latest_head
            # FaceParams.yaw is in [-1, 1], same scale as HeadPose. We
            # tone down slightly to avoid extreme rotations producing
            # broken renders on some render modes.
            params.yaw = float(h.yaw) * 0.8
            params.pitch = float(h.pitch) * 0.7
            # FaceParams doesn't always carry roll; only set if attr exists.
            if hasattr(params, "head_roll"):
                params.head_roll = float(h.roll) * 0.7
        return params
