"""Gaze-target heuristic.

We already have a high-quality gaze direction (iris vs eye-corners
from the refined face mesh, exposed as :data:`EventType.GAZE`) plus a
head-pose signal. This module turns those into a more semantic
*target* label: ``camera`` / ``screen`` / ``away`` / ``down`` / ...

No new model — just a small lookup over the live PerceptionStore
state. Cheap enough to be its own on-demand tool but light enough
that we could promote it to ambient narration later if the call
pattern justifies it.
"""

from __future__ import annotations

from faceview.core.logger import get_logger


log = get_logger("gaze_target")


def gaze_target() -> str:
    """Return a one-sentence read of what the user is looking at."""
    try:
        from faceview.vision.perception import PerceptionStore
        store = PerceptionStore.shared()
        snap = store.snapshot_dict()
    except Exception as exc:  # noqa: BLE001
        return f"Gaze read unavailable: {exc}"
    gaze = snap.get("gaze")
    head = snap.get("head_pose")
    pres = snap.get("presence")
    if pres is None or (pres.get("face_count") or 0) == 0:
        return "I don't see a face in the camera to track gaze for."
    if gaze is None or not gaze.get("fresh"):
        return "No fresh gaze signal — wait a moment for face mesh to settle."

    direction = gaze.get("direction", "away")
    attention = float(gaze.get("attention", 0.0))
    yaw = float(head.get("yaw", 0.0)) if head else 0.0
    pitch = float(head.get("pitch", 0.0)) if head else 0.0

    # Combine head pose with iris direction for a finer label.
    if direction == "camera" and attention > 0.65:
        target = "the camera"
    elif direction == "down" or pitch < -0.35:
        target = ("something below the camera (phone, "
                  "keyboard, or notes)")
    elif direction == "up" or pitch > 0.35:
        target = "something above the camera (probably the screen / monitor)"
    elif direction in ("left", "right"):
        side = "left" if direction == "left" else "right"
        # Strong head turn → looking off-screen on that side.
        if abs(yaw) > 0.4:
            target = f"something off-screen to the {side}"
        else:
            target = f"something on the {side} side of the screen"
    elif direction == "away":
        target = "no clear target — head turned away"
    else:
        target = "something the heuristic can't classify"
    log.info("gaze_target.read", direction=direction, target=target[:60])
    return (f"Gaze target: {target} "
            f"(attention {attention:.2f}, head yaw {yaw:+.2f}, "
            f"pitch {pitch:+.2f}).")
