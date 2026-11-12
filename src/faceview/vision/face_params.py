"""FaceParams: the renderer-input dataclass shared by every face renderer.

Qt-free (moved out of :mod:`faceview.vision.sim_face`, which re-exports it) so
the animation pipeline (:mod:`face_state`, :mod:`personas`, :mod:`avatar`) and
the web interface can use it without importing PySide6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FaceParams:
    """Renderer input. Most fields are 0..1 unless noted.

    Coarse fields (``smile``, ``brow_raise``, ``jaw_open``) preserve backward
    compatibility with hand-built scenes. The "AU-grade" fields below let the
    renderer distinguish visemes (``mouth_pucker`` vs ``mouth_stretch``) and
    show subtle expression cues (``cheek_raise``, ``upper_lid_raise``, the
    individual brow components). When in doubt :func:`face_state_to_params`
    populates them all from FACS Action Units.
    """

    # Pose
    yaw: float = 0.0          # -1 (left) .. 1 (right)
    pitch: float = 0.0        # -1 (up) .. 1 (down)

    # Coarse expression knobs
    eye_open: float = 1.0     # 0 = closed (blink)
    jaw_open: float = 0.0     # 0..1
    smile: float = 0.0        # -1 (frown) .. 1 (smile)
    brow_raise: float = 0.0   # -1 (frown) .. 1 (raised) — master fallback
    pupil_x: float = 0.0      # -1 .. 1
    pupil_y: float = 0.0      # -1 .. 1

    # Appearance
    skin_hue: float = 28.0    # HSV-style hue 0..360
    background: str = "#0c0f14"
    hair_color: str = "#2c1810"
    lip_color: str = "#a44a4a"

    # Renderer selection. "stylised" = the layered cartoony pipeline below.
    # "anatomical" / "anatomy_overlay" / "wireframe" route to
    # ``vision.sim_face_anatomical.render_face_anatomical`` instead.
    render_mode: str = "stylised"

    # ICT-FaceKit identity coefficients (PCA modes that vary base
    # face shape). Empty dict = use the neutral mean face.
    identity_weights: dict = None

    # AU-grade detail (all 0..1, defaults 0 → ignored by old call-sites)
    mouth_pucker: float = 0.0       # AU22 lip funneler — pulls lips inward and forward
    mouth_stretch: float = 0.0      # AU20 lip stretch — widens the mouth horizontally
    cheek_raise: float = 0.0        # AU6 cheek raise — lifts cheeks, narrows lower-eye
    nose_wrinkle: float = 0.0       # AU9 nose wrinkle — bunches skin around nose
    upper_lid_raise: float = 0.0    # AU5 upper lid raise — eye-widening
    inner_brow_raise: float = 0.0   # AU1 — inner brow tips lift (sad/surprised)
    outer_brow_raise: float = 0.0   # AU2 — outer brow tips lift (surprise)
    brow_lower: float = 0.0         # AU4 — both brows down + together (anger/concentration)
    lip_corner_drop: float = 0.0    # AU15 — lip corners drop (sadness)
    chin_raise: float = 0.0         # AU17 — chin raises, lower lip up (pout)
    upper_lip_raise: float = 0.0    # AU10 — upper lip pulled up (snarl / disgust)
    dimpler: float = 0.0            # AU14 — corner pulled inward (smirk)
    lip_tighten: float = 0.0        # AU23 — lips drawn tight, narrowed
    lip_press: float = 0.0          # AU24 — lips pressed together

    # Direct ICT blendshape pass-through. Whatever name → value pairs
    # are in this dict get added to the ARKit coefficient stream
    # before rendering. Used for blendshapes that don't fit the
    # 12-AU vocabulary cleanly (PupilDilate_L/R, jawForward, mouthLeft,
    # mouthRight, mouthFunnel, mouthClose, cheekPuff_L/R, etc.).
    direct_blendshapes: dict = None

    # Convenience constructors
    @classmethod
    def neutral(cls) -> "FaceParams":
        return cls()

    @classmethod
    def happy(cls) -> "FaceParams":
        return cls(smile=0.85, eye_open=0.9, brow_raise=0.15, cheek_raise=0.7)

    @classmethod
    def surprised(cls) -> "FaceParams":
        return cls(
            brow_raise=0.9, eye_open=1.0, jaw_open=0.55,
            inner_brow_raise=0.9, outer_brow_raise=0.9, upper_lid_raise=0.8,
        )

    @classmethod
    def sad(cls) -> "FaceParams":
        return cls(
            smile=-0.6, brow_raise=-0.3, eye_open=0.7,
            inner_brow_raise=0.7, lip_corner_drop=0.8,
        )

    @classmethod
    def speaking(cls, t: float) -> "FaceParams":
        """Animated talking-mouth based on a time scalar (seconds)."""
        return cls(
            jaw_open=0.18 + 0.18 * (1 + math.sin(t * 8)) / 2,
            smile=0.12 + 0.06 * math.sin(t * 1.7),
            brow_raise=0.05 * math.sin(t * 0.5),
            pupil_x=0.15 * math.sin(t * 0.7),
        )
