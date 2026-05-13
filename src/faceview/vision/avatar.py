"""TalkingAvatar — anatomically-grounded animated speaking face.

Built on the FACS Action Unit model in :mod:`faceview.vision.face_state`. A
:class:`TalkingAvatar` owns a :class:`FaceState` and advances it on every
:meth:`tick`. Three layers compose each frame:

1. **Baseline expression** — an FACS preset from
   :mod:`faceview.vision.expressions` (``neutral`` / ``happy`` / ``sad`` /
   ``surprised`` / etc.). This is the "resting" face when nobody is speaking.

2. **Idle behaviour** — :class:`AutoBlink` (~3-5 s blink interval, 0.08 s
   close + 0.12 s open), :class:`AutoBreathing` (slow sinusoidal AU9/AU25
   bias), and :class:`AutoSaccade` (occasional gaze shifts).

3. **Active utterance** — a :class:`SpeechEngine` timeline whose viseme
   targets override the mouth AUs (AU25 / AU26 / AU22 / AU20 / AU12). Other
   AUs continue to express the baseline emotion, so a smiling speaker still
   smiles between syllables.

Smoothing is exponential approach (``rate * dt``) toward the target AU
values — each tick moves a fraction of the remaining distance, which
produces natural-feeling transitions without per-AU velocity tracking.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from faceview.vision.expressions import apply_expression
from faceview.vision.face_state import AU_IDS, FaceState, face_state_to_params
from faceview.vision.personas import Persona, apply_persona, load_persona
from faceview.vision.sim_face import FaceParams
from faceview.vision.speech import (
    SpeechEngine, TimedViseme, tongue_pose_at, viseme_at, viseme_blend_at,
)


# ── Idle systems ─────────────────────────────────────────────────────────


class AutoBlink:
    CLOSE_DUR = 0.08
    OPEN_DUR = 0.12
    MIN_INTERVAL = 2.5
    MAX_INTERVAL = 5.0

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self._timer = 0.0
        self._next = self._roll()
        self._blinking = False
        self._phase = 0.0

    def _roll(self) -> float:
        return self.rng.uniform(self.MIN_INTERVAL, self.MAX_INTERVAL)

    def update(self, state: FaceState, dt: float) -> None:
        if self._blinking:
            self._phase += dt
            total = self.CLOSE_DUR + self.OPEN_DUR
            if self._phase < self.CLOSE_DUR:
                state.blink_amount = self._phase / self.CLOSE_DUR
            elif self._phase < total:
                t = (self._phase - self.CLOSE_DUR) / self.OPEN_DUR
                state.blink_amount = max(0.0, 1.0 - t)
            else:
                state.blink_amount = 0.0
                self._blinking = False
                self._timer = 0.0
                self._next = self._roll()
        else:
            self._timer += dt
            if self._timer >= self._next:
                self._blinking = True
                self._phase = 0.0


class AutoBreathing:
    """Subtle nostril-flare + lip-part bias on a slow sinusoid (~0.4 Hz)."""
    NOSTRIL_AMP = 0.06   # AU9
    LIP_PART_AMP = 0.03  # AU25

    def __init__(self) -> None:
        self._phase = 0.0

    def update(self, state: FaceState, dt: float) -> None:
        self._phase += dt * 2.5
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
        breath = (math.sin(self._phase) + 1.0) * 0.5  # 0..1
        # max() so we don't dampen an active expression
        state.AU9 = max(state.AU9, breath * self.NOSTRIL_AMP)
        state.AU25 = max(state.AU25, breath * self.LIP_PART_AMP)


class AutoSaccade:
    """Periodic small gaze shifts (max ±0.3 in eye_look_x/y)."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self._next_at = self.rng.uniform(1.0, 2.5)
        self._timer = 0.0
        self._target_x = 0.0
        self._target_y = 0.0

    def update(self, state: FaceState, dt: float) -> None:
        self._timer += dt
        if self._timer >= self._next_at:
            self._target_x = self.rng.uniform(-0.30, 0.30)
            self._target_y = self.rng.uniform(-0.18, 0.18)
            self._timer = 0.0
            self._next_at = self.rng.uniform(1.2, 2.6)
        # Ease toward the target — fast settling so saccades look saccadic.
        k = min(1.0, dt * 6.0)
        state.eye_look_x += (self._target_x - state.eye_look_x) * k
        state.eye_look_y += (self._target_y - state.eye_look_y) * k


# ── Utterance ────────────────────────────────────────────────────────────


@dataclass
class Utterance:
    text: str
    start_t: float
    timeline: list[TimedViseme]

    @property
    def duration(self) -> float:
        return self.timeline[-1].end_time if self.timeline else 0.0


# ── TalkingAvatar ────────────────────────────────────────────────────────


_MOUTH_AUS = ("AU25", "AU26", "AU22", "AU20", "AU12")


class TalkingAvatar:
    """Animated face built on FACS AUs, with lip-sync from text."""

    def __init__(
        self,
        *,
        emotion: str = "neutral",
        persona: str | Persona = "default",
        seed: Optional[int] = None,
        speech_engine: Optional[SpeechEngine] = None,
        smoothing_rate: float = 12.0,
    ) -> None:
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self.state = FaceState()
        self.target = FaceState()
        self.baseline = FaceState()
        apply_expression(self.baseline, emotion)
        self.target = self.baseline.copy()

        self.blink = AutoBlink(self._rng)
        self.breath = AutoBreathing()
        self.saccade = AutoSaccade(self._rng)
        self.engine = speech_engine or SpeechEngine()
        self.smoothing_rate = smoothing_rate

        self._t0 = time.monotonic()
        self._last_t: Optional[float] = None
        self._utterance: Optional[Utterance] = None
        self._emotion = emotion
        self.persona: Persona = persona if isinstance(persona, Persona) else load_persona(persona)

    # ── public ──────────────────────────────────────────────────────

    @property
    def t(self) -> float:
        return time.monotonic() - self._t0

    @property
    def emotion(self) -> str:
        return self._emotion

    def reset_clock(self) -> None:
        self._t0 = time.monotonic()
        self._last_t = None
        self._utterance = None

    def set_emotion(self, name: str) -> None:
        self._emotion = name
        self.baseline = FaceState()
        apply_expression(self.baseline, name)

    def set_persona(self, name: str | Persona) -> None:
        self.persona = name if isinstance(name, Persona) else load_persona(name)

    def say(self, text: str, *, speed: float = 1.0) -> Utterance:
        timeline = self.engine.generate_au_sequence(text, speed=speed)
        u = Utterance(text=text, start_t=self.t, timeline=timeline)
        self._utterance = u
        return u

    def is_speaking(self) -> bool:
        return self._utterance is not None and self.t < self._utterance.start_t + self._utterance.duration

    # ── per-frame update ────────────────────────────────────────────

    def tick(self, t: Optional[float] = None) -> FaceParams:
        """Advance internal state and return the rendered :class:`FaceParams`."""
        if t is None:
            t = self.t
        if self._last_t is None:
            dt = 1.0 / 60.0
        else:
            dt = max(0.0, min(0.2, t - self._last_t))
        self._last_t = t

        # 1. Build a target FaceState from the baseline expression.
        self.target = self.baseline.copy()

        # 2. Active utterance overrides only the mouth AUs. Use the
        # coarticulation-aware blend so AUs ramp continuously across
        # viseme boundaries instead of stepping.
        if self._utterance is not None:
            rel = t - self._utterance.start_t
            blend = viseme_blend_at(self._utterance.timeline, rel)
            if blend:
                for au, val in blend.items():
                    setattr(self.target, au, max(getattr(self.target, au), float(val)))
            # Speech-driven tongue. Each viseme has a target tongue
            # pose (from visemes.TONGUE_POSE); coarticulation between
            # adjacent visemes is smoothed by tongue_pose_at.
            tongue = tongue_pose_at(self._utterance.timeline, rel)
            if tongue is not None:
                e, v, l, tp = tongue
                self._talking_tongue = (e, v, l, tp)
            else:
                self._talking_tongue = (-0.85, 0.0, 0.0, 0.30)
            # Subtle head sway while speaking — three out-of-phase
            # sinusoids on yaw / pitch / roll.
            self.target.head_yaw += 0.13 * math.sin(rel * 2.1)
            self.target.head_pitch += 0.07 * math.sin(rel * 1.4 + 0.7)
            self.target.head_roll += 0.05 * math.sin(rel * 1.7 + 2.3)
            if rel >= self._utterance.duration + 0.05:
                self._utterance = None
                self._talking_tongue = None
        else:
            self._talking_tongue = None

        # 3. Idle behaviours act on the *target*, not the rendered state, so
        # blinks/breathing/saccades don't get re-smoothed away on each tick.
        self.blink.update(self.target, dt)
        self.breath.update(self.target, dt)
        self.saccade.update(self.target, dt)

        # 4. Smooth approach for AUs and pose. blink_amount transitions are
        # already short, so use a faster rate for it.
        k = min(1.0, dt * self.smoothing_rate)
        for au in AU_IDS:
            cur = getattr(self.state, au)
            tgt = getattr(self.target, au)
            setattr(self.state, au, cur + (tgt - cur) * k)
        for f in ("head_yaw", "head_pitch", "head_roll", "eye_look_x", "eye_look_y"):
            cur = getattr(self.state, f)
            tgt = getattr(self.target, f)
            setattr(self.state, f, cur + (tgt - cur) * k)
        # Eye blink is animated directly by AutoBlink — copy through.
        self.state.blink_amount = self.target.blink_amount

        params = face_state_to_params(self.state)
        # Forward the speech-driven tongue pose to the renderer so
        # the EffectsRuntime can pick it up (only applied when the
        # talking_tongue slider is enabled and no manual tongue
        # slider has overridden it).
        params._talking_tongue = getattr(self, "_talking_tongue", None)
        return apply_persona(params, self.persona)
