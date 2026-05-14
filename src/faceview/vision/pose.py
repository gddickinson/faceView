"""Pose / posture analysis via MediaPipe Pose.

On-demand only — invoked by the ``describe_pose`` LLM tool. Loads
MediaPipe Pose lazily (the model file ships with the ``mediapipe``
wheel; no separate download). Returns a short natural-language
description: "sitting, leaning forward, arms crossed".

If MediaPipe isn't installed we surface the standard MissingDependency
error pointing at the ``vision`` extra.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from faceview.core.errors import MissingDependency
from faceview.core.logger import get_logger


log = get_logger("pose")


# MediaPipe Pose landmark indices we use:
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE = 25, 26
_L_ANKLE, _R_ANKLE = 27, 28


class PoseAnalyzer:
    """Singleton MP Pose wrapper."""

    _instance: "PoseAnalyzer | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "PoseAnalyzer":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = PoseAnalyzer()
        return cls._instance

    def __init__(self) -> None:
        self._pose = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._pose is not None:
            return self._pose
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise MissingDependency("mediapipe", "vision") from exc
        with self._lock:
            if self._pose is None:
                self._pose = mp.solutions.pose.Pose(  # type: ignore[attr-defined]
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                )
                log.info("pose.loaded")
        return self._pose

    def analyse(self, frame: np.ndarray) -> Optional[dict]:
        """Return a dict of features or ``None`` if no body detected."""
        pose = self._ensure()
        rgb = frame[:, :, ::-1].copy()
        try:
            res = pose.process(rgb)
        except Exception as exc:  # noqa: BLE001
            log.warning("pose.error", error=str(exc))
            return None
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks.landmark
        out: dict = {"visible": True}

        # Standing-vs-sitting heuristic: are knees and hips below the
        # bottom of the frame? If hips are visible but knees aren't,
        # likely sitting.
        knees_visible = (lms[_L_KNEE].visibility > 0.6
                         or lms[_R_KNEE].visibility > 0.6)
        hips_visible = (lms[_L_HIP].visibility > 0.6
                        or lms[_R_HIP].visibility > 0.6)
        if knees_visible and hips_visible:
            hip_y = (lms[_L_HIP].y + lms[_R_HIP].y) * 0.5
            knee_y = (lms[_L_KNEE].y + lms[_R_KNEE].y) * 0.5
            out["posture"] = ("sitting" if knee_y - hip_y < 0.15
                              else "standing")
        elif hips_visible and not knees_visible:
            out["posture"] = "sitting"
        else:
            out["posture"] = "framing only above the waist"

        # Leaning forward/backward — head x relative to hip x
        if hips_visible and lms[_NOSE].visibility > 0.6:
            hip_x = (lms[_L_HIP].x + lms[_R_HIP].x) * 0.5
            dx = lms[_NOSE].x - hip_x
            if abs(dx) > 0.10:
                out["lean"] = "right" if dx > 0 else "left"
            # Forward/back from y: nose well above hip means upright,
            # nose level with shoulders means leaning forward.
            if (lms[_L_SHOULDER].visibility > 0.6
                    and lms[_R_SHOULDER].visibility > 0.6):
                sh_y = (lms[_L_SHOULDER].y + lms[_R_SHOULDER].y) * 0.5
                if lms[_NOSE].y - sh_y > 0.05:
                    out["forward_lean"] = True

        # Arms crossed: wrist on the OPPOSITE side of the body centre.
        if (lms[_L_WRIST].visibility > 0.5
                and lms[_R_WRIST].visibility > 0.5
                and hips_visible):
            mid = (lms[_L_HIP].x + lms[_R_HIP].x) * 0.5
            l_wrist_right = lms[_L_WRIST].x > mid
            r_wrist_left = lms[_R_WRIST].x < mid
            out["arms_crossed"] = l_wrist_right and r_wrist_left

        # Hand raised: wrist y above shoulder y (smaller = higher in
        # image coords).
        if (lms[_L_WRIST].visibility > 0.5
                and lms[_L_SHOULDER].visibility > 0.5
                and lms[_L_WRIST].y < lms[_L_SHOULDER].y - 0.05):
            out["left_hand_raised"] = True
        if (lms[_R_WRIST].visibility > 0.5
                and lms[_R_SHOULDER].visibility > 0.5
                and lms[_R_WRIST].y < lms[_R_SHOULDER].y - 0.05):
            out["right_hand_raised"] = True
        return out


def describe_pose(frame: np.ndarray) -> str:
    """One-paragraph posture summary the LLM can relay."""
    if frame is None:
        return "No camera frame is available right now."
    try:
        feats = PoseAnalyzer.shared().analyse(frame)
    except MissingDependency:
        return "Pose analysis isn't available (MediaPipe not installed)."
    except Exception as exc:  # noqa: BLE001
        log.warning("pose.tool_error", error=str(exc))
        return f"Pose analysis failed: {exc}"
    if feats is None:
        return ("I don't see a clearly-visible body in the frame "
                "to analyse posture from.")
    bits: list[str] = []
    posture = feats.get("posture")
    if posture:
        bits.append(posture)
    if feats.get("forward_lean"):
        bits.append("leaning forward")
    lean = feats.get("lean")
    if lean:
        bits.append(f"leaning to the {lean}")
    if feats.get("arms_crossed"):
        bits.append("arms crossed")
    if feats.get("left_hand_raised") and feats.get("right_hand_raised"):
        bits.append("both hands raised")
    elif feats.get("left_hand_raised"):
        bits.append("left hand raised")
    elif feats.get("right_hand_raised"):
        bits.append("right hand raised")
    if not bits:
        return "The person's posture looks neutral and upright."
    return "Posture read: " + ", ".join(bits) + "."
