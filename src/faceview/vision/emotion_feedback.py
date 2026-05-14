"""User-emotion → persona-emotion feedback loop (C8).

The persona's emotional ledger (in :class:`CognitionStore`) used to
evolve only from the persona's own replies. This module closes the
loop: when the *user* shows a strong, sustained emotion via
DeepFace's classification, we bump the persona's emotional state
accordingly. A persona that's "kind" notices when you're sad and
the system prompt reflects it on the next turn.

Heuristic, on purpose:
- Subscribe to :data:`EventType.EMOTION` (user-side deepface).
- Maintain a rolling sample over the last ~20 s.
- If the same non-neutral emotion dominates with mean confidence
  ≥ 0.55 AND the persona's ledger hasn't been bumped for that
  emotion in the last ~60 s, nudge the persona's emotional state.

Mapping (user → persona bump):

    happy     → joy        (intensity 0.3, "user looked happy")
    sad       → tenderness (intensity 0.4, "user looked sad")
    angry     → no bump (don't escalate; staying calm is the
                          better persona response)
    fear      → tenderness (intensity 0.3)
    surprise  → surprise   (intensity 0.25)
    disgust   → no bump (also kept calm)

The loop is *opt-in for downstream effects*: CognitionStore
``set_emotion`` is called, decay parameters do their job, and the
LLM sees the updated mood through the existing system prompt — no
new event types or coupling required.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import Emotion, EventType
from faceview.core.logger import get_logger


log = get_logger("emotion_feedback")


_WINDOW_S = 20.0
_MIN_CONF = 0.55
_MIN_SAMPLES = 3
_COOLDOWN_S = 60.0


_USER_TO_PERSONA: dict[str, tuple[str, float]] = {
    "happy":    ("joy", 0.3),
    "sad":      ("tenderness", 0.4),
    "fear":     ("tenderness", 0.3),
    "surprise": ("surprise", 0.25),
    # Neutral/angry/disgust: stay out of the way.
}


class EmotionFeedback:
    """Singleton-ish; attached to a ClaudeClient at boot via
    :meth:`attach`."""

    _instance: "EmotionFeedback | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "EmotionFeedback":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = EmotionFeedback()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client = None  # ClaudeClient — set by attach()
        self._samples: deque[tuple[float, str, float]] = deque(
            maxlen=200
        )
        self._last_bumped_at: dict[str, float] = {}
        get_bus().subscribe(EventType.EMOTION, self._on_emotion)

    # ── public API ────────────────────────────────────────────

    def attach(self, claude_client) -> None:
        """ClaudeClient holds the bound memory; we read the latest
        memory store off it before every bump in case the user
        swapped persona since the last sample arrived."""
        with self._lock:
            self._client = claude_client

    # ── bus handler ──────────────────────────────────────────

    def _on_emotion(self, payload) -> None:
        if not isinstance(payload, Emotion):
            return
        label = (payload.label or "").lower()
        conf = float(payload.confidence or 0.0)
        if not label or label == "unknown":
            return
        now = time.time()
        with self._lock:
            self._samples.append((now, label, conf))
            # Drop anything older than _WINDOW_S so dominance
            # calculations are over a rolling tail.
            while (self._samples
                   and now - self._samples[0][0] > _WINDOW_S):
                self._samples.popleft()
        self._maybe_bump()

    def _maybe_bump(self) -> None:
        with self._lock:
            if len(self._samples) < _MIN_SAMPLES:
                return
            counts: dict[str, list[float]] = {}
            for _ts, label, conf in self._samples:
                counts.setdefault(label, []).append(conf)
            dominant_label, confs = max(
                counts.items(), key=lambda kv: len(kv[1]),
            )
            if len(confs) / len(self._samples) < 0.5:
                return
            mean_conf = sum(confs) / len(confs)
            if mean_conf < _MIN_CONF:
                return
            mapping = _USER_TO_PERSONA.get(dominant_label)
            if mapping is None:
                return
            now = time.time()
            last = self._last_bumped_at.get(dominant_label, 0.0)
            if now - last < _COOLDOWN_S:
                return
            self._last_bumped_at[dominant_label] = now
            client = self._client
        if client is None:
            return
        memory = getattr(client, "memory", None)
        if memory is None or not hasattr(memory, "set_emotion"):
            return
        persona_emotion, intensity = mapping
        try:
            memory.set_emotion(
                persona_emotion,
                intensity=intensity,
                trigger=f"user looked {dominant_label}",
            )
            log.info("emotion_feedback.bumped",
                     user=dominant_label,
                     persona=persona_emotion,
                     mean_conf=round(mean_conf, 2))
        except Exception as exc:  # noqa: BLE001
            log.warning("emotion_feedback.bump_failed",
                        error=str(exc))
