"""FACS-based face state used by the animation pipeline.

The 12 Action Units (AUs) here are the same set used by the FaceForge anatomy
project and align with Ekman's Facial Action Coding System. Compared to the
hand-rolled :class:`FaceParams` they offer a meaningful, anatomically-grounded
parameter space: expression presets and visemes are defined as AU activations
rather than as raw smile/jaw numbers.

The renderer in :mod:`faceview.vision.sim_face` still consumes
:class:`FaceParams`, so :func:`face_state_to_params` is the bridge between the
two: it boils a 19-DoF AU-style FaceState down to the 8 fields the renderer
understands. New AUs can be added without changing the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

from faceview.vision.sim_face import FaceParams


# Canonical AU identifier list (matches assets/config/au_definitions.json).
AU_IDS: list[str] = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU9",
    "AU12", "AU15", "AU20", "AU22", "AU25", "AU26",
]


@dataclass
class FaceState:
    """Animation state expressed as FACS AU activations + head pose + gaze.

    All AUs are 0..1 (clamped on apply). Head pose is in normalised units
    where ±1 corresponds to a noticeable but not extreme rotation.
    """

    # Action Units (12)
    AU1: float = 0.0   # Inner Brow Raise
    AU2: float = 0.0   # Outer Brow Raise
    AU4: float = 0.0   # Brow Lower
    AU5: float = 0.0   # Upper Lid Raise
    AU6: float = 0.0   # Cheek Raise
    AU9: float = 0.0   # Nose Wrinkle
    AU12: float = 0.0  # Lip Corner Pull
    AU15: float = 0.0  # Lip Corner Drop
    AU20: float = 0.0  # Lip Stretch
    AU22: float = 0.0  # Lip Funneler
    AU25: float = 0.0  # Lips Part
    AU26: float = 0.0  # Jaw Drop

    # Head pose
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0

    # Eye
    eye_look_x: float = 0.0
    eye_look_y: float = 0.0
    blink_amount: float = 0.0   # 0 = fully open, 1 = fully closed

    # Skin
    skin_hue: float = 28.0
    background: str = "#0c0f14"

    # ── helpers ──────────────────────────────────────────────────────

    def get(self, au_id: str) -> float:
        return float(getattr(self, au_id, 0.0))

    def set(self, au_id: str, value: float) -> None:
        setattr(self, au_id, max(0.0, min(1.0, value)))

    def reset_aus(self) -> None:
        for au in AU_IDS:
            setattr(self, au, 0.0)

    def copy(self) -> "FaceState":
        return FaceState(**self.__dict__)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def face_state_to_params(s: FaceState) -> FaceParams:
    """Translate a FACS-based :class:`FaceState` into renderer-ready FaceParams.

    Mappings (informed by FACS interpretations):
    - smile     = AU12 (corner pull) − AU15 (corner drop)
                  + 0.3·AU20 (stretch widens mouth horizontally)
                  − 0.5·AU22 (funneler tightens corners inward)
    - jaw_open  = 0.35·AU25 (lips part, small) + AU26 (jaw drop)
    - brow_raise = (AU1+AU2)/2 (raise) − AU4 (lower)
    - eye_open   = 1 − blink_amount, then boosted by AU5 (lid raise)
    """
    smile = s.AU12 - s.AU15 + 0.3 * s.AU20 - 0.5 * s.AU22
    jaw_open = max(0.0, 0.35 * s.AU25 + s.AU26)
    brow_raise = (s.AU1 + s.AU2) * 0.5 - s.AU4
    eye_open = max(0.05, 1.0 - s.blink_amount)
    eye_open = min(1.05, eye_open + 0.10 * s.AU5)

    return FaceParams(
        yaw=_clip(s.head_yaw),
        pitch=_clip(s.head_pitch),
        eye_open=eye_open,
        jaw_open=max(0.0, min(1.0, jaw_open)),
        smile=_clip(smile),
        brow_raise=_clip(brow_raise),
        pupil_x=_clip(s.eye_look_x),
        pupil_y=_clip(s.eye_look_y),
        skin_hue=s.skin_hue,
        background=s.background,
    )
