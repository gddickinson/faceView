"""TestConversation — ping-pong two bots in the GUI's two windows.

When the user enables Test Mode in the config dialog, MainWindow spawns:

- a second :class:`SimCameraWorker` that publishes to ``EventType.FRAME``
  (so it appears in the *camera* window, where the user normally sits);
- a :class:`TestConversation` that alternates "speakers" by publishing
  ``CHAT_USER_MESSAGE`` / ``LLM_REPLY`` events and driving the two
  avatars' lip-sync via direct ``avatar.say(...)`` calls.

The conversation runs on a background thread, paced slowly enough that
the lip-sync and emotion changes are clearly visible. The two bots are
intentionally simple: they ECHO each other with a small prompt-style
mutation, so the demo is bounded and does not consume real API tokens
even when a key is set.
"""

from __future__ import annotations

import itertools
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import ChatMessage, Emotion, EventType
from faceview.core.logger import get_logger

if TYPE_CHECKING:
    from faceview.vision.sim_camera import SimCameraWorker


log = get_logger("test_conversation")


SEED_PROMPTS = [
    "Hi! Can you see me on your side?",
    "Yes — and can you see me? What expression am I making?",
    "I think you look curious. Want to play a word game?",
    "Sure. Say a word and I'll respond with an emotion.",
    "Sunshine.",
    "Happy. Storm.",
    "Anxious. Library.",
    "Calm. Your turn — give me a feeling.",
    "Hope. What does hope look like for you?",
    "A small light at the end of a long hallway. Yours?",
    "The first sentence of a book you can't put down.",
]


_EMOTION_KEYWORDS = {
    "happy":   ["sunshine", "smile", "play", "joy", "hope", "calm", "fun"],
    "sad":     ["alone", "lost", "sorry", "miss", "rain", "dark"],
    "surprise":["wow", "really", "?", "what", "sudden", "storm"],
    "neutral": [],
}


def _guess_emotion(text: str) -> str:
    t = text.lower()
    for label, words in _EMOTION_KEYWORDS.items():
        for w in words:
            if w in t:
                return label
    return "neutral"


@dataclass
class TestConversation:
    """Drive two bots in alternating turns.

    The orchestrator owns no UI — it just publishes events that the
    avatars and chat panel already react to.
    """

    avatar_worker: "SimCameraWorker"
    user_worker:   "SimCameraWorker"
    period_s:      float = 6.0

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lines = list(SEED_PROMPTS)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="test-conversation", daemon=True
        )
        self._thread.start()
        log.info("test_conversation.started", period_s=self.period_s)

    def stop(self) -> None:
        self._stop.set()
        log.info("test_conversation.stopped")

    def _loop(self) -> None:
        bus = get_bus()
        # Alternate (user-side, avatar-side) turns.
        sides = itertools.cycle(["user", "claude"])
        line_iter = itertools.cycle(self._lines)
        # Stagger the very first turn so the windows have time to paint.
        time.sleep(1.0)
        while not self._stop.is_set():
            side = next(sides)
            text = next(line_iter)
            mood = _guess_emotion(text)
            if side == "user":
                # The user-window bot speaks; the chat panel renders this
                # as a "You" message; the avatar (Claude) is the listener
                # and will respond on the next tick.
                self.user_worker.avatar.say(text)
                bus.publish(EventType.CHAT_USER_MESSAGE, ChatMessage("user", text))
                bus.publish(EventType.EMOTION, Emotion(label=mood, confidence=0.7))
            else:
                self.avatar_worker.avatar.say(text)
                bus.publish(EventType.LLM_REPLY, ChatMessage("assistant", text))
                bus.publish(EventType.EMOTION, Emotion(label=mood, confidence=0.7))
            # Pace by talking time plus a small idle gap so visemes finish.
            wait = max(2.0, min(self.period_s, 0.07 * len(text) + 2.0))
            slept = 0.0
            while slept < wait and not self._stop.is_set():
                time.sleep(0.1)
                slept += 0.1
