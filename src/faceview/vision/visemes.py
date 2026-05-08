"""Viseme alphabet defined as FACS AU activations.

A viseme is a visually-distinguishable mouth shape. Where a phoneme captures
how a speech sound is articulated, a viseme captures what it *looks* like
from the outside. Several phonemes share a viseme (``b/p/m`` all look like
closed lips), which is why mouth shapes are a coarser signal than audio.

This module ships the 15-class viseme alphabet from the FaceForge animation
system (in turn adapted from common SAPI/Disney sets):

==============  =====================================================
Viseme          Phoneme class
==============  =====================================================
``REST``        silence
``PP``          P / B / M  — bilabial closure
``FF``          F / V       — labiodental
``TH``          TH / DH    — dental
``DD``          T / D / N / L — alveolar
``SS``          S / Z      — sibilant
``SH``          SH / ZH / CH / JH — palatal
``KK``          K / G / NG  — velar
``RR``          R / ER     — retroflex
``AA``          AA / AE / AH — open vowel
``EH``          EH / EY    — mid vowel
``IH``          IH / IY    — close-front vowel
``OH``          AO / OW / OY — round-back vowel
``UH``          UH / UW    — close-round vowel
``WW``          W / Y / HH  — glide
==============  =====================================================

Each viseme is an :class:`AUTarget` — five AUs that move the mouth. Other
AUs are unaffected, so a viseme blends with the current expression baseline
(a smiling speaker still smiles between syllables).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AUTarget:
    """AU activations a viseme drives directly."""
    AU25: float = 0.0   # Lips Part
    AU26: float = 0.0   # Jaw Drop
    AU22: float = 0.0   # Lip Funneler
    AU20: float = 0.0   # Lip Stretch
    AU12: float = 0.0   # Lip Corner Pull (smile gives EE/IH the right shape)

    def to_dict(self) -> dict[str, float]:
        return {"AU25": self.AU25, "AU26": self.AU26,
                "AU22": self.AU22, "AU20": self.AU20, "AU12": self.AU12}


# Tongue-position targets per viseme. Each tuple is
# (extend, vertical, lateral, taper). Values are in [-1, 1] to
# match the slider ranges. extend ≈ -0.4 = "just barely visible
# behind the teeth"; +0.05 = "tip at the lip line"; +0.4 = "tip
# protrudes" (only for /th/ — between-teeth sounds).
# vertical: +1 = tip up to alveolar/palate, -1 = down on floor.
# lateral: side-shift (most consonants are midline).
TONGUE_POSE: dict[str, tuple[float, float, float, float]] = {
    "REST": (-0.85,  0.0, 0.0, 0.30),
    # Bilabials — tongue rests low on floor.
    "PP":   (-0.85,  -0.15, 0.0, 0.30),
    # Labio-dentals — tongue at lower-front, doesn't help much.
    "FF":   (-0.55,  -0.10, 0.0, 0.40),
    # Inter-dentals — tongue tip BETWEEN teeth (visible).
    "TH":   ( 0.10,   0.20, 0.0, 0.65),
    # Alveolars — tip touches the ridge behind upper teeth.
    "DD":   (-0.30,   0.55, 0.0, 0.55),
    # Sibilants — tip near alveolar, slightly retracted.
    "SS":   (-0.40,   0.45, 0.0, 0.50),
    # Post-alveolars — tip back of alveolar, body bunched.
    "SH":   (-0.45,   0.30, 0.0, 0.55),
    # Velars — tongue back arched up, tip stays low.
    "KK":   (-0.55,  -0.15, 0.0, 0.30),
    # Rhotic — tip curled back.
    "RR":   (-0.45,   0.20, 0.0, 0.55),
    # Open vowels — tongue low, central.
    "AA":   (-0.65,  -0.40, 0.0, 0.30),
    # Mid front vowels — tongue mid-front.
    "EH":   (-0.55,   0.10, 0.0, 0.30),
    # High front vowels — tongue high-front.
    "IH":   (-0.50,   0.50, 0.0, 0.30),
    # Mid-back rounded — tongue mid-back.
    "OH":   (-0.60,  -0.20, 0.0, 0.30),
    # High back rounded — tongue high-back.
    "UH":   (-0.55,   0.20, 0.0, 0.30),
    # Glides — tongue mid + rounded lips.
    "WW":   (-0.55,   0.10, 0.0, 0.35),
}


VISEMES: dict[str, AUTarget] = {
    "REST": AUTarget(0.0, 0.0, 0.0, 0.0, 0.0),
    "PP":   AUTarget(0.0, 0.05, 0.0, 0.0, 0.0),
    "FF":   AUTarget(0.20, 0.05, 0.0, 0.10, 0.0),
    "TH":   AUTarget(0.30, 0.10, 0.0, 0.0, 0.0),
    "DD":   AUTarget(0.30, 0.10, 0.0, 0.0, 0.0),
    "SS":   AUTarget(0.20, 0.05, 0.0, 0.20, 0.0),
    "SH":   AUTarget(0.30, 0.10, 0.40, 0.0, 0.0),
    "KK":   AUTarget(0.20, 0.15, 0.0, 0.0, 0.0),
    "RR":   AUTarget(0.30, 0.10, 0.30, 0.0, 0.0),
    "AA":   AUTarget(0.60, 0.40, 0.0, 0.0, 0.0),
    "EH":   AUTarget(0.40, 0.20, 0.0, 0.20, 0.0),
    "IH":   AUTarget(0.30, 0.10, 0.0, 0.30, 0.10),
    "OH":   AUTarget(0.50, 0.30, 0.50, 0.0, 0.0),
    "UH":   AUTarget(0.30, 0.15, 0.60, 0.0, 0.0),
    "WW":   AUTarget(0.20, 0.10, 0.30, 0.0, 0.0),
}


# ARPAbet phoneme → viseme mapping. Stress-marker variants (`AA0/1/2`) are
# enumerated so we don't need to strip in hot paths.
PHONEME_TO_VISEME: dict[str, str] = {
    "P": "PP", "B": "PP", "M": "PP",
    "F": "FF", "V": "FF",
    "TH": "TH", "DH": "TH",
    "T": "DD", "D": "DD", "N": "DD", "L": "DD",
    "S": "SS", "Z": "SS",
    "SH": "SH", "ZH": "SH", "CH": "SH", "JH": "SH",
    "K": "KK", "G": "KK", "NG": "KK",
    "R": "RR",
    "W": "WW", "Y": "WW", "HH": "WW",
    "SIL": "REST",
}
for _vow, _vis in [
    ("ER", "RR"),
    ("AA", "AA"), ("AE", "AA"), ("AH", "AA"), ("AW", "AA"), ("AY", "AA"),
    ("EH", "EH"), ("EY", "EH"),
    ("IH", "IH"), ("IY", "IH"),
    ("AO", "OH"), ("OW", "OH"), ("OY", "OH"),
    ("UH", "UH"), ("UW", "UH"),
]:
    for _suffix in ("", "0", "1", "2"):
        PHONEME_TO_VISEME[f"{_vow}{_suffix}"] = _vis
del _vow, _vis, _suffix


def viseme_for_phoneme(phoneme: str) -> str:
    """Return the viseme name for an ARPAbet phoneme; ``REST`` for unknown."""
    if phoneme in PHONEME_TO_VISEME:
        return PHONEME_TO_VISEME[phoneme]
    # Strip stress digit and try again (covers e.g. EY3 if ever encountered).
    base = "".join(c for c in phoneme if not c.isdigit())
    return PHONEME_TO_VISEME.get(base, "REST")


def viseme_au_targets(name: str) -> dict[str, float]:
    """Return AU dict for viseme ``name`` (empty for unknown)."""
    v = VISEMES.get(name)
    return v.to_dict() if v else {}
