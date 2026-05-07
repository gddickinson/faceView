"""PreFX effects — modify ``FaceParams`` before the renderer runs.

Used for shape morphs (eyes huge, head squish, mouth O), pose
perturbations, and AU drives that interact with the rest of the
animation pipeline.

Each function takes ``(params, u, intensity)`` where ``u ∈ [0, 1]``
is the normalised time into the effect's duration and ``intensity
∈ [0, 1]`` is the trigger amplitude.
"""
from __future__ import annotations

import math


def pre_eyes_huge(params, u: float, intensity: float) -> None:
    """Eyes widen dramatically (surprise)."""
    params.upper_lid_raise = max(getattr(params, "upper_lid_raise", 0.0),
                                    intensity * (0.6 + 0.4 * math.sin(u * math.pi)))
    params.eye_open = min(1.4, getattr(params, "eye_open", 1.0)
                            + 0.4 * intensity * math.sin(u * math.pi))


def pre_eyes_closed(params, u: float, intensity: float) -> None:
    """Eyes squeeze shut (focus / discomfort)."""
    params.eye_open = max(0.0, getattr(params, "eye_open", 1.0)
                            - intensity * math.sin(u * math.pi))


def pre_mouth_o(params, u: float, intensity: float) -> None:
    """Mouth opens to a round 'O' shape (gasp)."""
    params.jaw_open = max(getattr(params, "jaw_open", 0.0),
                            intensity * 0.7 * math.sin(u * math.pi))
    params.mouth_pucker = max(getattr(params, "mouth_pucker", 0.0),
                                intensity * 0.6 * math.sin(u * math.pi))


def pre_mouth_grin(params, u: float, intensity: float) -> None:
    """Wide grin."""
    params.smile = min(1.0, getattr(params, "smile", 0.0)
                        + intensity * 0.9 * math.sin(u * math.pi))
    params.cheek_raise = max(getattr(params, "cheek_raise", 0.0),
                                intensity * 0.7 * math.sin(u * math.pi))


def pre_brow_furrow(params, u: float, intensity: float) -> None:
    """Brows pull inward + down (concentration / anger)."""
    params.brow_lower = max(getattr(params, "brow_lower", 0.0),
                            intensity * 0.85 * math.sin(u * math.pi))


def pre_head_shake(params, u: float, intensity: float) -> None:
    """Head shakes left-right (no)."""
    params.yaw = float(getattr(params, "yaw", 0.0)) + \
        0.45 * intensity * math.sin(u * 2 * math.pi * 3)


def pre_head_nod(params, u: float, intensity: float) -> None:
    """Head nods up-down (yes)."""
    params.pitch = float(getattr(params, "pitch", 0.0)) + \
        0.35 * intensity * math.sin(u * 2 * math.pi * 2)


def pre_head_recoil(params, u: float, intensity: float) -> None:
    """Head jerks backward in shock."""
    params.pitch = float(getattr(params, "pitch", 0.0)) - \
        0.55 * intensity * math.sin(u * math.pi)


HANDLERS = {
    "eyes_huge":   pre_eyes_huge,
    "eyes_closed": pre_eyes_closed,
    "mouth_o":     pre_mouth_o,
    "mouth_grin":  pre_mouth_grin,
    "brow_furrow": pre_brow_furrow,
    "head_shake":  pre_head_shake,
    "head_nod":    pre_head_nod,
    "head_recoil": pre_head_recoil,
}
