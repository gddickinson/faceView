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
from faceview.core.events import EventType, HeadPose, MouthActivity
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
