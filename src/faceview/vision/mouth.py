"""Mouth-activity + viseme detection from MediaPipe Face Landmarker blendshapes.

True open-vocabulary visual speech recognition is impractical from Python on
Apple Silicon in 2026 (see README's *Lip-reading scope* section). What works
*well* in real time is mapping the 52 ARKit-style blendshape coefficients
into:

- a **speaking / silent** binary based on jaw-open + mouth-funnel + lower-lip
  motion-magnitude, with a small temporal window;
- a **viseme class** (AA / EE / OO / MM / FV) from blendshape thresholds —
  enough to drive face animation rigs and give a plausible lip-shape readout
  for the GUI.

That's exactly what this module does. It is intentionally simple and rule-
based; an upgrade path to `auto-AVSR` ONNX is documented in the README.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import (
    Blink, EventType, FaceDistance, Gaze, HeadPose, MouthActivity,
)
from faceview.core.logger import get_logger


log = get_logger("mouth")


VISEMES = {
    "AA": ("jawOpen", "mouthFunnel"),       # open vowels (father)
    "EE": ("mouthSmileLeft", "mouthSmileRight"),  # tight-lipped vowels
    "OO": ("mouthPucker", "mouthFunnel"),    # rounded vowels
    "MM": ("mouthClose", "mouthRollLower"),  # bilabial
    "FV": ("mouthFunnel", "mouthRollLower"),  # labio-dental
}


class MouthAnalyzer:
    def __init__(self) -> None:
        self._lm = None
        self._mp = None
        self._lock = threading.Lock()
        self._last_emit = 0.0
        self._jaw_window: deque[float] = deque(maxlen=8)
        # Rolling blink-rate window: timestamps of detected blink-closures.
        self._blink_times: deque[float] = deque()
        self._was_eye_closed = False

    def start(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise MissingDependency("mediapipe", "vision") from exc

        self._mp = mp
        self._lm = mp.solutions.face_mesh.FaceMesh(  # type: ignore[attr-defined]
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        get_bus().subscribe(EventType.FRAME, self._on_frame)
        log.info("mouth.started")

    def _on_frame(self, frame) -> None:
        if frame is None or self._lm is None:
            return
        now = time.time()
        if now - self._last_emit < 0.1:
            return
        self._last_emit = now

        # FaceMesh doesn't return ARKit blendshapes directly, but we can
        # derive proxies from the 478 landmark points: jaw-open from chin↔nose
        # distance, mouth pucker from outer-lip width, etc.
        try:
            res = self._lm.process(frame[:, :, ::-1])
        except Exception as exc:  # noqa: BLE001
            log.warning("mouth.error", error=str(exc))
            return
        if not res.multi_face_landmarks:
            return

        lms = res.multi_face_landmarks[0].landmark
        # Indices for upper/lower lip and corners come from MediaPipe's stable
        # 468-point face-mesh ordering.
        upper = lms[13]   # upper inner lip
        lower = lms[14]   # lower inner lip
        left = lms[78]    # left corner
        right = lms[308]  # right corner

        # Normalise by face scale: nose tip (1) to chin (152) distance.
        nose = lms[1]
        chin = lms[152]
        face_h = abs(chin.y - nose.y) + 1e-6

        jaw_open = abs(upper.y - lower.y) / face_h
        mouth_w = abs(left.x - right.x) / face_h
        # Pucker proxy: when lips push forward, width shrinks while jaw_open is small.
        mouth_pucker = max(0.0, 0.55 - mouth_w) / 0.5
        mouth_funnel = max(0.0, jaw_open - 0.05)

        self._jaw_window.append(jaw_open)
        avg = sum(self._jaw_window) / len(self._jaw_window)
        speaking = avg > 0.10

        viseme: Optional[str] = None
        if speaking:
            if jaw_open > 0.22:
                viseme = "AA"
            elif mouth_pucker > 0.4:
                viseme = "OO"
            elif mouth_w > 0.55:
                viseme = "EE"
            elif jaw_open < 0.04:
                viseme = "MM"
            else:
                viseme = "FV"

        get_bus().publish(
            EventType.MOUTH_ACTIVITY,
            MouthActivity(
                speaking=speaking,
                jaw_open=jaw_open,
                mouth_funnel=mouth_funnel,
                mouth_pucker=mouth_pucker,
                viseme=viseme,
            ),
        )

        # ── head pose (approximation from landmark geometry) ──
        # MediaPipe FaceMesh landmarks are in normalised image coords
        # (x, y in [0, 1]). We approximate head pose without solvePnP:
        # - yaw   from nose-x vs face midline (between ear-anchor lms)
        # - pitch from nose-y vs eye line
        # - roll  from the slope of the eye line
        # These are good enough for "mirror my head" feedback.
        try:
            left_eye = lms[33]   # outer corner of left eye
            right_eye = lms[263]  # outer corner of right eye
            mid_x = (left_eye.x + right_eye.x) * 0.5
            mid_y = (left_eye.y + right_eye.y) * 0.5
            eye_w = max(1e-3, abs(right_eye.x - left_eye.x))
            # Yaw: how far is the nose tip from the eye midline,
            # normalised by inter-eye distance. Mirror left/right.
            yaw = -float((nose.x - mid_x) / eye_w) * 2.0
            # Pitch: nose tip y vs eye y, normalised by chin distance.
            chin_dist = max(1e-3, abs(chin.y - mid_y))
            pitch = -float((nose.y - mid_y) / chin_dist - 0.6) * 2.0
            # Roll: slope of the eye line.
            import math
            roll = float(math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
            roll = roll / (math.pi * 0.25)  # ±45° → ±1
            # Clamp to [-1, 1].
            yaw = max(-1.0, min(1.0, yaw))
            pitch = max(-1.0, min(1.0, pitch))
            roll = max(-1.0, min(1.0, roll))
            get_bus().publish(
                EventType.HEAD_POSE,
                HeadPose(yaw=yaw, pitch=pitch, roll=roll),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("mouth.head_pose_error", error=str(exc))

        # ── face distance, gaze, blink ──────────────────────────────────
        # We reuse the face mesh already running above so these signals
        # come for free. All published at the same ~10 Hz cadence as the
        # mouth/head emits — PerceptionStore samples whichever is most
        # recent.
        try:
            self._emit_distance(lms)
        except Exception as exc:  # noqa: BLE001
            log.warning("mouth.distance_error", error=str(exc))
        try:
            self._emit_gaze(lms)
        except Exception as exc:  # noqa: BLE001
            log.warning("mouth.gaze_error", error=str(exc))
        try:
            self._emit_blink(lms)
        except Exception as exc:  # noqa: BLE001
            log.warning("mouth.blink_error", error=str(exc))

    # ── derived signals ──────────────────────────────────────────────────

    def _emit_distance(self, lms) -> None:
        """Coarse face distance from inter-eye width in normalised coords."""
        left_eye = lms[33]
        right_eye = lms[263]
        eye_w = abs(right_eye.x - left_eye.x)
        # Approximate bbox area share: square of eye width × a constant.
        # Calibrated by eye on a 1280×720 webcam at typical distances.
        bbox_ratio = max(0.0, min(1.0, (eye_w * 3.2) ** 2))
        if eye_w > 0.16:
            label = "close"
        elif eye_w > 0.10:
            label = "near"
        elif eye_w > 0.06:
            label = "normal"
        else:
            label = "far"
        get_bus().publish(
            EventType.FACE_DISTANCE,
            FaceDistance(label=label, bbox_ratio=bbox_ratio),
        )

    def _emit_gaze(self, lms) -> None:
        """Iris-relative gaze direction (requires ``refine_landmarks=True``)."""
        # Iris landmarks: left iris 468–472, right iris 473–477 (refined mesh).
        if len(lms) < 478:
            return
        l_iris = lms[468]
        r_iris = lms[473]
        # Eye corners
        l_outer = lms[33]
        l_inner = lms[133]
        r_inner = lms[362]
        r_outer = lms[263]
        # Eye vertical bounds
        l_top = lms[159]
        l_bot = lms[145]
        r_top = lms[386]
        r_bot = lms[374]

        def _norm_offset(iris, inner, outer, top, bot) -> tuple[float, float]:
            ex = (inner.x + outer.x) * 0.5
            ey = (top.y + bot.y) * 0.5
            w = max(1e-3, abs(outer.x - inner.x))
            h = max(1e-3, abs(bot.y - top.y))
            return (iris.x - ex) / w, (iris.y - ey) / h

        lx, ly = _norm_offset(l_iris, l_inner, l_outer, l_top, l_bot)
        rx, ry = _norm_offset(r_iris, r_inner, r_outer, r_top, r_bot)
        yaw = max(-1.0, min(1.0, (lx + rx) * 1.6))
        pitch = max(-1.0, min(1.0, (ly + ry) * 1.6))
        # Attention: 1 when iris is centred in both eyes.
        attention = max(0.0, 1.0 - (abs(yaw) + abs(pitch)) * 0.6)
        attention = max(0.0, min(1.0, attention))
        if attention > 0.65:
            direction = "camera"
        elif yaw > 0.35:
            direction = "right"
        elif yaw < -0.35:
            direction = "left"
        elif pitch > 0.35:
            direction = "down"
        elif pitch < -0.35:
            direction = "up"
        else:
            direction = "away"
        get_bus().publish(
            EventType.GAZE,
            Gaze(direction=direction, yaw=yaw, pitch=pitch,
                 attention=attention),
        )

    def _emit_blink(self, lms) -> None:
        """Eye-aspect-ratio (Soukupová & Čech) + rolling blink rate."""
        def _ear(top1, top2, bot1, bot2, outer, inner) -> float:
            vert = (abs(top1.y - bot1.y) + abs(top2.y - bot2.y)) * 0.5
            horiz = max(1e-4, abs(outer.x - inner.x))
            return vert / horiz

        left = _ear(lms[159], lms[158], lms[145], lms[153], lms[33], lms[133])
        right = _ear(lms[386], lms[385], lms[374], lms[380], lms[263], lms[362])
        ear = (left + right) * 0.5
        # EAR ~0.30 open, ~0.10 closed. State threshold below.
        state = "open"
        closed = ear < 0.18
        if closed:
            state = "closed"
        elif ear < 0.22:
            state = "drowsy"

        now = time.time()
        if closed and not self._was_eye_closed:
            self._blink_times.append(now)
        self._was_eye_closed = closed
        # Trim to last 30 s.
        while self._blink_times and now - self._blink_times[0] > 30.0:
            self._blink_times.popleft()
        rate_per_min = (len(self._blink_times) / 30.0) * 60.0

        get_bus().publish(
            EventType.BLINK,
            Blink(eye_open=ear, state=state, rate_per_min=rate_per_min),
        )
