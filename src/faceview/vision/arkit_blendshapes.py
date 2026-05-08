"""ARKit-compatible 52 blendshapes — industry-standard FACS-derived set.

Used by Apple ARKit, Google MediaPipe FaceLandmarker, MetaHumans,
Ready Player Me, and most VR/AR avatar systems. Each blendshape is a
named float coefficient in [0, 1] driving a specific facial action.

This module exposes the 52 names as a canonical list and provides
two-way mapping between ARKit blendshapes and our 12 FACS Action
Units. That lets:

- An external face-tracker (MediaPipe FaceLandmarker output, an
  iPhone Face ID app, etc.) drive our avatar via the standard
  vocabulary.
- Our internal AU pipeline emit ARKit-compatible coefficients for
  rigging external character models (Unity/Unreal MetaHumans, RPM
  avatars, etc.).

References:
- ARKit ARFaceAnchor.BlendShapeLocation enumeration.
- MediaPipe blendshape model output.
"""

from __future__ import annotations

from dataclasses import dataclass


# The canonical 52 ARKit blendshapes, in the order Apple ARKit and
# MediaPipe FaceLandmarker emit them.
ARKIT_BLENDSHAPES: list[str] = [
    # Brow (5)
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    # Cheek (3)
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    # Eye (14)
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    # Jaw (4)
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    # Mouth (23)
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    # Nose (2)
    "noseSneerLeft", "noseSneerRight",
    # Tongue (1)
    "tongueOut",
]
assert len(ARKIT_BLENDSHAPES) == 52, "ARKit canonical set is 52 shapes"


# ── ARKit → our 12 AUs ────────────────────────────────────────────
# Many ARKit shapes overlap onto the same AU. We sum (clamped to 1).

ARKIT_TO_AU: dict[str, list[tuple[str, float]]] = {
    # Brows
    "browInnerUp": [("AU1", 1.0)],
    "browOuterUpLeft": [("AU2", 1.0)],
    "browOuterUpRight": [("AU2", 1.0)],
    "browDownLeft": [("AU4", 1.0)],
    "browDownRight": [("AU4", 1.0)],
    # Eyes
    "eyeWideLeft": [("AU5", 1.0)],
    "eyeWideRight": [("AU5", 1.0)],
    "eyeSquintLeft": [("AU6", 0.7)],
    "eyeSquintRight": [("AU6", 0.7)],
    # Cheek
    "cheekSquintLeft": [("AU6", 1.0)],
    "cheekSquintRight": [("AU6", 1.0)],
    "cheekPuff": [("AU22", 0.5)],
    # Nose
    "noseSneerLeft": [("AU9", 1.0)],
    "noseSneerRight": [("AU9", 1.0)],
    # Mouth — smile / frown / lip drop / corner pull
    "mouthSmileLeft": [("AU12", 1.0)],
    "mouthSmileRight": [("AU12", 1.0)],
    "mouthFrownLeft": [("AU15", 1.0)],
    "mouthFrownRight": [("AU15", 1.0)],
    "mouthDimpleLeft": [("AU12", 0.4)],
    "mouthDimpleRight": [("AU12", 0.4)],
    # Mouth — stretch / pucker / funnel
    "mouthStretchLeft": [("AU20", 1.0)],
    "mouthStretchRight": [("AU20", 1.0)],
    "mouthPucker": [("AU22", 1.0)],
    "mouthFunnel": [("AU22", 0.7)],
    # Lip part / jaw
    "jawOpen": [("AU26", 1.0), ("AU25", 0.4)],
    "mouthClose": [("AU25", -0.6)],   # negative = close lips
    # Lower-face actions mapped roughly to AU25/26
    "mouthLowerDownLeft": [("AU25", 0.5)],
    "mouthLowerDownRight": [("AU25", 0.5)],
    "mouthUpperUpLeft": [("AU25", 0.4)],
    "mouthUpperUpRight": [("AU25", 0.4)],
}


# ── Our 12 AUs → ARKit ────────────────────────────────────────────
# Inverse mapping; for export. Many AUs split across L/R ARKit pairs.

AU_TO_ARKIT: dict[str, list[tuple[str, float]]] = {
    "AU1": [("browInnerUp", 1.0)],
    "AU2": [("browOuterUpLeft", 1.0), ("browOuterUpRight", 1.0)],
    "AU4": [("browDownLeft", 1.0), ("browDownRight", 1.0)],
    "AU5": [("eyeWideLeft", 1.0), ("eyeWideRight", 1.0)],
    "AU6": [("cheekSquintLeft", 1.0), ("cheekSquintRight", 1.0)],
    "AU9": [("noseSneerLeft", 1.0), ("noseSneerRight", 1.0)],
    "AU12": [("mouthSmileLeft", 1.0), ("mouthSmileRight", 1.0)],
    "AU15": [("mouthFrownLeft", 1.0), ("mouthFrownRight", 1.0)],
    "AU20": [("mouthStretchLeft", 1.0), ("mouthStretchRight", 1.0)],
    "AU22": [("mouthPucker", 1.0)],
    "AU25": [("mouthLowerDownLeft", 0.5), ("mouthLowerDownRight", 0.5),
              ("mouthUpperUpLeft", 0.5), ("mouthUpperUpRight", 0.5)],
    "AU26": [("jawOpen", 1.0)],
    "AU45": [("eyeBlinkLeft", 1.0), ("eyeBlinkRight", 1.0)],
    "AU17": [("mouthShrugLower", 1.0), ("mouthShrugUpper", 0.5)],
    "AU23": [("mouthRollLower", 0.6), ("mouthRollUpper", 0.6),
              ("mouthPressLeft", 0.4), ("mouthPressRight", 0.4)],
    "AU24": [("mouthPressLeft", 1.0), ("mouthPressRight", 1.0)],
    "AU10": [("mouthUpperUpLeft", 1.0), ("mouthUpperUpRight", 1.0)],
    "AU14": [("mouthDimpleLeft", 1.0), ("mouthDimpleRight", 1.0)],
}


# ── conversion ─────────────────────────────────────────────────────


def arkit_to_au_values(coefficients: dict[str, float]) -> dict[str, float]:
    """Convert a 52-shape ARKit dict to our 12-AU dict.

    Coefficients can be a sparse subset; missing names are treated as 0.
    Multiple ARKit shapes mapping to the same AU are summed and clamped.
    """
    out: dict[str, float] = {}
    for name, weight in coefficients.items():
        for au, gain in ARKIT_TO_AU.get(name, []):
            out[au] = out.get(au, 0.0) + float(weight) * gain
    # Clamp.
    return {au: max(-1.0, min(1.0, v)) for au, v in out.items()}


def au_to_arkit_values(au_values: dict[str, float]) -> dict[str, float]:
    """Convert our 12-AU dict to a 52-shape ARKit dict (most fields 0)."""
    out: dict[str, float] = {name: 0.0 for name in ARKIT_BLENDSHAPES}
    for au, value in au_values.items():
        for name, gain in AU_TO_ARKIT.get(au, []):
            out[name] = max(out[name], float(value) * gain)
    return out


@dataclass
class ARKitFrame:
    """One frame of ARKit-compatible facial capture data."""
    coefficients: dict[str, float]

    @classmethod
    def from_au_values(cls, au_values: dict[str, float]) -> "ARKitFrame":
        return cls(au_to_arkit_values(au_values))

    def to_au_values(self) -> dict[str, float]:
        return arkit_to_au_values(self.coefficients)
