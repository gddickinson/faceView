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


def _env(u: float) -> float:
    """Half-sine envelope — ramp up then down over the duration."""
    return math.sin(u * math.pi)


def pre_mouth_pout(params, u: float, intensity: float) -> None:
    """Lower lip pushed out (pout / sulk)."""
    e = _env(u) * intensity
    params.chin_raise = max(getattr(params, "chin_raise", 0.0), e * 0.85)
    params.lip_corner_drop = max(getattr(params, "lip_corner_drop", 0.0),
                                    e * 0.4)


def pre_mouth_sad(params, u: float, intensity: float) -> None:
    """Lip corners drop (sad / disappointed)."""
    e = _env(u) * intensity
    params.lip_corner_drop = max(getattr(params, "lip_corner_drop", 0.0),
                                    e * 0.85)
    params.smile = min(getattr(params, "smile", 0.0), -e * 0.6)
    params.inner_brow_raise = max(getattr(params, "inner_brow_raise", 0.0),
                                      e * 0.45)


def pre_lips_pursed(params, u: float, intensity: float) -> None:
    """Lips pulled together into a small purse (concentration / kiss)."""
    e = _env(u) * intensity
    params.mouth_pucker = max(getattr(params, "mouth_pucker", 0.0), e * 0.85)
    params.lip_press = max(getattr(params, "lip_press", 0.0), e * 0.5)


def pre_lips_tight(params, u: float, intensity: float) -> None:
    """Lips drawn tight, narrowed (tension / containment)."""
    e = _env(u) * intensity
    params.lip_tighten = max(getattr(params, "lip_tighten", 0.0), e * 0.95)


def pre_lips_pressed(params, u: float, intensity: float) -> None:
    """Lips pressed together flat (worry / restraint)."""
    e = _env(u) * intensity
    params.lip_press = max(getattr(params, "lip_press", 0.0), e * 0.95)


def pre_mouth_smirk(params, u: float, intensity: float) -> None:
    """Asymmetric one-sided smile."""
    e = _env(u) * intensity
    params.dimpler = max(getattr(params, "dimpler", 0.0), e * 0.7)
    params.smile = min(1.0, getattr(params, "smile", 0.0) + e * 0.4)


def pre_mouth_snarl(params, u: float, intensity: float) -> None:
    """Upper lip raised, nose wrinkled (disgust / contempt)."""
    e = _env(u) * intensity
    params.upper_lip_raise = max(getattr(params, "upper_lip_raise", 0.0),
                                    e * 0.9)
    params.nose_wrinkle = max(getattr(params, "nose_wrinkle", 0.0), e * 0.7)


def pre_brow_arch_one(params, u: float, intensity: float) -> None:
    """Single-brow raise (skepticism)."""
    e = _env(u) * intensity
    # Asymmetric: lift right brow only.
    params.outer_brow_raise = max(getattr(params, "outer_brow_raise", 0.0),
                                      e * 0.7)


def pre_brow_high(params, u: float, intensity: float) -> None:
    """Both brows raised high (curiosity / surprise-light)."""
    e = _env(u) * intensity
    params.inner_brow_raise = max(getattr(params, "inner_brow_raise", 0.0),
                                      e * 0.85)
    params.outer_brow_raise = max(getattr(params, "outer_brow_raise", 0.0),
                                      e * 0.85)


def pre_squint(params, u: float, intensity: float) -> None:
    """Cheeks lift + lids partially close (suspicion / focus)."""
    e = _env(u) * intensity
    params.cheek_raise = max(getattr(params, "cheek_raise", 0.0), e * 0.7)
    params.eye_open = min(getattr(params, "eye_open", 1.0),
                            1.0 - e * 0.4)


def pre_surprise_combo(params, u: float, intensity: float) -> None:
    """Eyes wide + brows up + jaw drop (full-body surprise)."""
    e = _env(u) * intensity
    pre_eyes_huge(params, u, intensity)
    pre_brow_high(params, u, intensity)
    params.jaw_open = max(getattr(params, "jaw_open", 0.0), e * 0.45)


def pre_disgust_combo(params, u: float, intensity: float) -> None:
    """Snarl + brow down + squint."""
    pre_mouth_snarl(params, u, intensity)
    pre_brow_furrow(params, u, intensity * 0.8)
    pre_squint(params, u, intensity * 0.6)


def pre_tongue_out(params, u: float, intensity: float) -> None:
    """Mouth opens to let the tongue stick out (paired with the
    PostFX overlay that actually draws the tongue)."""
    e = _env(u) * intensity
    params.jaw_open = max(getattr(params, "jaw_open", 0.0), e * 0.55)
    params.smile = max(getattr(params, "smile", 0.0), e * 0.25)


def _set_direct(params, name: str, value: float) -> None:
    """Helper: set/raise an entry in params.direct_blendshapes."""
    direct = getattr(params, "direct_blendshapes", None) or {}
    direct[name] = max(direct.get(name, 0.0), value)
    params.direct_blendshapes = direct


def pre_pupils_huge(params, u: float, intensity: float) -> None:
    """Drive PupilDilate_L/R blendshapes — pupils enlarge dramatically."""
    e = _env(u) * intensity
    _set_direct(params, "PupilDilate_L", e)
    _set_direct(params, "PupilDilate_R", e)


def pre_pupils_pinpoint(params, u: float, intensity: float) -> None:
    """Pupils constrict tiny (shock / bright light). Dilation negative
    isn't a thing — we widen the eyes via squint suppression instead."""
    e = _env(u) * intensity
    params.upper_lid_raise = max(getattr(params, "upper_lid_raise", 0.0),
                                    e * 0.6)


def pre_jaw_forward(params, u: float, intensity: float) -> None:
    """Underbite / aggressive jaw thrust."""
    _set_direct(params, "jawForward", _env(u) * intensity * 0.85)


def pre_mouth_to_left(params, u: float, intensity: float) -> None:
    """Mouth slides to the screen-left (annoyance / thinking)."""
    _set_direct(params, "mouthLeft", _env(u) * intensity * 0.85)


def pre_mouth_to_right(params, u: float, intensity: float) -> None:
    _set_direct(params, "mouthRight", _env(u) * intensity * 0.85)


def pre_mouth_funnel(params, u: float, intensity: float) -> None:
    """Lip funnel — horn-shape forward."""
    _set_direct(params, "mouthFunnel", _env(u) * intensity * 0.9)


def pre_cheeks_puff(params, u: float, intensity: float) -> None:
    """Both cheeks puff out (holding breath / chubby cheek emote)."""
    e = _env(u) * intensity * 0.85
    _set_direct(params, "cheekPuff_L", e)
    _set_direct(params, "cheekPuff_R", e)
    params.lip_press = max(getattr(params, "lip_press", 0.0), e * 0.6)


HANDLERS = {
    "eyes_huge":      pre_eyes_huge,
    "eyes_closed":    pre_eyes_closed,
    "mouth_o":        pre_mouth_o,
    "mouth_grin":     pre_mouth_grin,
    "mouth_pout":     pre_mouth_pout,
    "mouth_sad":      pre_mouth_sad,
    "lips_pursed":    pre_lips_pursed,
    "lips_tight":     pre_lips_tight,
    "lips_pressed":   pre_lips_pressed,
    "mouth_smirk":    pre_mouth_smirk,
    "mouth_snarl":    pre_mouth_snarl,
    "brow_furrow":    pre_brow_furrow,
    "brow_arch_one":  pre_brow_arch_one,
    "brow_high":      pre_brow_high,
    "squint":         pre_squint,
    "surprise_combo": pre_surprise_combo,
    "disgust_combo":  pre_disgust_combo,
    "head_shake":     pre_head_shake,
    "head_nod":       pre_head_nod,
    "head_recoil":    pre_head_recoil,
    "tongue_out":     pre_tongue_out,
    "pupils_huge":    pre_pupils_huge,
    "pupils_pinpoint":pre_pupils_pinpoint,
    "jaw_forward":    pre_jaw_forward,
    "mouth_to_left":  pre_mouth_to_left,
    "mouth_to_right": pre_mouth_to_right,
    "mouth_funnel":   pre_mouth_funnel,
    "cheeks_puff":    pre_cheeks_puff,
}
