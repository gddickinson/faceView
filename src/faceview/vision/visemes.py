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
