"""Text → phoneme → viseme → AU sequence pipeline for lip-sync.

Adapted from the FaceForge animation system. Uses a small bundled CMU
pronouncing dictionary for common words and a simple letter-rule fallback
for everything else. Output is a list of :class:`TimedViseme` records that
:class:`TalkingAvatar` plays back over time.

Usage::

    eng = SpeechEngine()
    timeline = eng.generate_au_sequence("Hello world.", speed=1.0)
    duration = eng.get_total_duration("Hello world.")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from faceview.assets import assets_dir
from faceview.vision.visemes import VISEMES, viseme_au_targets, viseme_for_phoneme


# Letter → ARPAbet phoneme rules — used when the CMU dict has no entry.
_LETTER_RULES: dict[str, list[str]] = {
    "a": ["AE1"], "b": ["B"],   "c": ["K"],     "d": ["D"],
    "e": ["EH1"], "f": ["F"],   "g": ["G"],     "h": ["HH"],
    "i": ["IH1"], "j": ["JH"],  "k": ["K"],     "l": ["L"],
    "m": ["M"],   "n": ["N"],   "o": ["OW1"],   "p": ["P"],
    "q": ["K", "W"],            "r": ["R"],     "s": ["S"],
    "t": ["T"],   "u": ["AH1"], "v": ["V"],     "w": ["W"],
    "x": ["K", "S"],            "y": ["Y"],     "z": ["Z"],
}


@dataclass
class TimedViseme:
    """A viseme scheduled with a start/end time and AU targets."""
    viseme: str
    start_time: float
    end_time: float
    au_targets: dict[str, float]


@lru_cache(maxsize=1)
def _load_cmu_dict() -> dict[str, list[str]]:
    path = assets_dir() / "data" / "cmu_dict_compact.json"
    if not path.exists():
        return {}
    return json.loads(Path(path).read_text())


class SpeechEngine:
    """Text-to-viseme pipeline with timing.

    Args:
        phoneme_duration: Base seconds per phoneme (default 0.085 ≈ 12 phon/s).
        word_gap: Silent pause inserted between words.
    """

    def __init__(
        self,
        phoneme_duration: float = 0.085,
        word_gap: float = 0.04,
    ) -> None:
        self.phoneme_duration = phoneme_duration
        self.word_gap = word_gap
        self._dict = _load_cmu_dict()

    # ── pipeline stages ────────────────────────────────────────────

    def text_to_phonemes(self, text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        out: list[str] = []
        for w in words:
            phonemes = self._dict.get(w) or self._rule_based(w)
            out.extend(phonemes)
            out.append("SIL")
        if out and out[-1] == "SIL":
            out.pop()
        return out

    def _rule_based(self, word: str) -> list[str]:
        out: list[str] = []
        for ch in word:
            r = _LETTER_RULES.get(ch)
            if r:
                out.extend(r)
        return out or ["AH1"]

    def phonemes_to_visemes(
        self, phonemes: list[str], *, speed: float = 1.0
    ) -> list[TimedViseme]:
        dur = self.phoneme_duration / max(speed, 0.1)
        gap = self.word_gap / max(speed, 0.1)
        t = 0.0
        out: list[TimedViseme] = []
        for ph in phonemes:
            if ph == "SIL":
                out.append(TimedViseme(
                    viseme="REST",
                    start_time=t,
                    end_time=t + gap,
                    au_targets=viseme_au_targets("REST"),
                ))
                t += gap
                continue
            v = viseme_for_phoneme(ph)
            out.append(TimedViseme(
                viseme=v,
                start_time=t,
                end_time=t + dur,
                au_targets=viseme_au_targets(v),
            ))
            t += dur
        return out

    def generate_au_sequence(
        self, text: str, *, speed: float = 1.0
    ) -> list[TimedViseme]:
        return self.phonemes_to_visemes(self.text_to_phonemes(text), speed=speed)

    def get_total_duration(self, text: str, *, speed: float = 1.0) -> float:
        seq = self.generate_au_sequence(text, speed=speed)
        return seq[-1].end_time if seq else 0.0


def viseme_at(timeline: list[TimedViseme], t: float) -> Optional[TimedViseme]:
    """Return the active viseme at time ``t`` from a precomputed timeline."""
    if not timeline:
        return None
    if t < timeline[0].start_time:
        return None
    if t > timeline[-1].end_time:
        return None
    # Linear scan — timelines are short (a few hundred entries even for long replies).
    for tv in timeline:
        if tv.start_time <= t <= tv.end_time:
            return tv
    return None


def _viseme_weight(tv: TimedViseme, t: float, attack: float, release: float) -> float:
    """Triangular activation envelope around a viseme's [start, end] window.

    Rises linearly over [start-attack, start], holds at 1 in [start, end],
    falls linearly over [end, end+release]. Outside that window: 0.
    """
    if t <= tv.start_time - attack or t >= tv.end_time + release:
        return 0.0
    if t < tv.start_time:
        return max(0.0, (t - (tv.start_time - attack)) / max(1e-6, attack))
    if t > tv.end_time:
        return max(0.0, 1.0 - (t - tv.end_time) / max(1e-6, release))
    return 1.0


def tongue_pose_at(
    timeline: list[TimedViseme],
    t: float,
    *,
    attack: float = 0.040,
    release: float = 0.060,
) -> tuple[float, float, float, float] | None:
    """Blend tongue (extend, vertical, lateral, taper) per viseme.

    Returns the weighted sum of TONGUE_POSE entries for the active
    visemes (same envelope as viseme_blend_at). When no viseme is
    active, returns None — caller can decide REST or hide.
    """
    from faceview.vision.visemes import TONGUE_POSE
    if not timeline:
        return None
    weights = []
    poses = []
    for tv in timeline:
        if t <= tv.start_time - attack:
            break
        if t >= tv.end_time + release:
            continue
        w = _viseme_weight(tv, t, attack, release)
        if w <= 0.0:
            continue
        pose = TONGUE_POSE.get(tv.viseme, TONGUE_POSE["REST"])
        weights.append(w)
        poses.append(pose)
    if not weights:
        return None
    total = sum(weights)
    if total <= 1e-6:
        return None
    e = sum(w * p[0] for w, p in zip(weights, poses)) / total
    v = sum(w * p[1] for w, p in zip(weights, poses)) / total
    l = sum(w * p[2] for w, p in zip(weights, poses)) / total
    tp = sum(w * p[3] for w, p in zip(weights, poses)) / total
    return float(e), float(v), float(l), float(tp)


def viseme_blend_at(
    timeline: list[TimedViseme],
    t: float,
    *,
    attack: float = 0.040,
    release: float = 0.060,
) -> dict[str, float]:
    """Blend overlapping viseme activations into a single AU target dict.

    Each viseme contributes during a triangular envelope around its slot.
    For each AU touched by any active viseme we take the weighted max
    (``weight * au_value``) — that produces a smooth crossfade between
    adjacent visemes (the new one ramps up as the old ramps down) without
    summing to unrealistic openings when both share an AU.

    Returns ``{}`` when ``t`` is outside the timeline. AUs missing from the
    returned dict should be treated as 0 by the caller (they are *released*).
    """
    if not timeline:
        return {}
    blend: dict[str, float] = {}
    for tv in timeline:
        if t <= tv.start_time - attack:
            break  # timelines are monotonic — no later viseme can be active yet
        if t >= tv.end_time + release:
            continue
        w = _viseme_weight(tv, t, attack, release)
        if w <= 0.0:
            continue
        for au, val in tv.au_targets.items():
            contrib = w * float(val)
            if contrib > blend.get(au, 0.0):
                blend[au] = contrib
    return blend
