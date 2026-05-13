"""TestConversation — ping-pong two bots in the GUI's two windows.

When the user enables Test Mode in the config dialog, MainWindow spawns:

- a second :class:`SimCameraWorker` that publishes to ``EventType.FRAME``
  (so it appears in the *camera* window, where the user normally sits);
- a :class:`TestConversation` that alternates "speakers", driving the two
  avatars' lip-sync via ``avatar.say(...)`` and rendering both sides in
  the chat panel.

Two modes:

- **canned** — cycles through :data:`SEED_PROMPTS` (bounded, no network).
  This is the default when no engines are supplied. Publishes
  ``CHAT_USER_MESSAGE`` and ``LLM_REPLY`` so the existing chat panel
  + status pills react as in normal use.

- **LLM** — when ``engine_a`` and ``engine_b`` are provided, each bot
  has its own :class:`Conversation` history and system prompt. The
  orchestrator routes each reply to the other bot as its next user
  message. Chat-panel rendering goes through
  :meth:`ChatPanel.append_external_message` so the real
  ``ClaudeClient`` subscription is not re-triggered.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import Emotion, EventType
from faceview.core.logger import get_logger
from faceview.llm.conversation import Conversation

if TYPE_CHECKING:
    from faceview.gui.chat_panel import ChatPanel
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
    "happy":    ["sunshine", "smile", "play", "joy", "hope", "calm", "fun", "great"],
    "sad":      ["alone", "lost", "sorry", "miss", "rain", "dark", "regret"],
    "surprise": ["wow", "really", "?", "what", "sudden", "storm", "amazing"],
    "neutral":  [],
}


def _guess_emotion(text: str) -> str:
    t = text.lower()
    for label, words in _EMOTION_KEYWORDS.items():
        for w in words:
            if w in t:
                return label
    return "neutral"


_DEFAULT_SYSTEM_A = (
    "You are a curious, plain-spoken student. You're chatting with a mentor "
    "via a real-time avatar interface. Keep every reply to 1–2 short "
    "sentences. Ask one follow-up question."
)
_DEFAULT_SYSTEM_B = (
    "You are a thoughtful, slightly witty mentor. You're chatting with a "
    "student via a real-time avatar interface. Keep every reply to 1–2 "
    "short sentences. Don't lecture."
)
_DEFAULT_SEED = "Hi! What's something interesting you've thought about today?"

_TURN_RULES = (
    "\n\nThis is a face-to-face avatar chat with another character. "
    "Keep every reply to 1–3 short sentences. Stay completely in character; "
    "do not break the fourth wall or mention being an AI."
)


def _system_for_persona(persona: str, *, fallback: str) -> str:
    """Build a bot system prompt from the persona's Character if one exists."""
    try:
        from faceview.llm.character import character_for
        c = character_for(persona)
        # If we got the default Character (no custom backstory), fall back.
        from faceview.llm.character import _DEFAULT_CHARACTER  # noqa: WPS437
        if c.name == _DEFAULT_CHARACTER.name and c.occupation == _DEFAULT_CHARACTER.occupation:
            return fallback + _TURN_RULES
        return c.narrate_identity() + _TURN_RULES
    except Exception:  # noqa: BLE001
        return fallback + _TURN_RULES


@dataclass
class TestConversation:
    """Drive two bots in alternating turns.

    The orchestrator owns no UI — it publishes events (canned mode) or
    pushes directly into the chat panel (LLM mode), and calls
    ``worker.avatar.say(text)`` for lip-sync on each turn.
    """

    avatar_worker: "SimCameraWorker"
    user_worker:   "SimCameraWorker"
    period_s:      float = 6.0
    engine_a:      Optional[Any] = None       # drives the camera-window bot
    engine_b:      Optional[Any] = None       # drives the avatar-window bot
    chat_panel:    Optional["ChatPanel"] = None
    persona_a:     Optional[str] = None       # camera-side persona → Character
    persona_b:     Optional[str] = None       # avatar-side persona → Character
    system_a:      str = _DEFAULT_SYSTEM_A
    system_b:      str = _DEFAULT_SYSTEM_B
    seed_prompt:   str = _DEFAULT_SEED
    label_a:       str = "Camera bot"
    label_b:       str = "Avatar bot"
    color_a:       str = "#1a73e8"
    color_b:       str = "#9b51e0"
    _stop:         threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread:       Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _conv_a:       Optional[Conversation]      = field(default=None, init=False, repr=False)
    _conv_b:       Optional[Conversation]      = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # If persona names were supplied, build the system prompts from
        # the matching Character. Otherwise fall back to the canned
        # student/mentor strings.
        sys_a = (_system_for_persona(self.persona_a, fallback=self.system_a)
                 if self.persona_a else self.system_a + _TURN_RULES)
        sys_b = (_system_for_persona(self.persona_b, fallback=self.system_b)
                 if self.persona_b else self.system_b + _TURN_RULES)
        self._conv_a = Conversation(system=sys_a)
        self._conv_b = Conversation(system=sys_b)
        # Surface the characters' display names if they exist.
        try:
            from faceview.llm.character import character_for, _DEFAULT_CHARACTER  # noqa: WPS437
            if self.persona_a:
                ca = character_for(self.persona_a)
                if ca.name != _DEFAULT_CHARACTER.name:
                    self.label_a = ca.name
            if self.persona_b:
                cb = character_for(self.persona_b)
                if cb.name != _DEFAULT_CHARACTER.name:
                    self.label_b = cb.name
        except Exception:  # noqa: BLE001
            pass

    # ── lifecycle ────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return "llm" if (self.engine_a is not None and self.engine_b is not None) else "canned"

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="test-conversation", daemon=True,
        )
        self._thread.start()
        log.info("test_conversation.started", mode=self.mode, period_s=self.period_s)

    def stop(self) -> None:
        self._stop.set()
        log.info("test_conversation.stopped")

    # ── loop dispatch ────────────────────────────────────────────────

    def _loop(self) -> None:
        time.sleep(1.0)  # let the windows paint
        try:
            if self.mode == "llm":
                self._llm_loop()
            else:
                self._canned_loop()
        except Exception as exc:  # noqa: BLE001
            log.warning("test_conversation.crashed", error=str(exc))

    # ── canned: original behaviour ───────────────────────────────────

    def _canned_loop(self) -> None:
        # Same display path as LLM mode — do NOT publish
        # CHAT_USER_MESSAGE / LLM_REPLY because those would re-trigger
        # the main ClaudeClient subscription and produce a real Claude
        # reply interleaved with the canned ping-pong.
        bus = get_bus()
        sides = itertools.cycle(["user", "claude"])
        line_iter = itertools.cycle(list(SEED_PROMPTS))
        while not self._stop.is_set():
            side = next(sides)
            text = next(line_iter)
            mood = _guess_emotion(text)
            if side == "user":
                self.user_worker.avatar.say(text)
                who, color = self.label_a, self.color_a
            else:
                self.avatar_worker.avatar.say(text)
                who, color = self.label_b, self.color_b
            if self.chat_panel is not None:
                self.chat_panel.append_external_message(who, text, color=color)
            bus.publish(EventType.EMOTION, Emotion(label=mood, confidence=0.7))
            self._wait_paced(text)

    # ── LLM-driven: each bot has its own engine + history ────────────

    def _llm_loop(self) -> None:
        bus = get_bus()
        # Bot A speaks first using the seed prompt as the incoming line.
        speaker = "a"
        incoming = self.seed_prompt
        while not self._stop.is_set():
            if speaker == "a":
                engine, worker, conv = self.engine_a, self.user_worker, self._conv_a
                who, color = self.label_a, self.color_a
            else:
                engine, worker, conv = self.engine_b, self.avatar_worker, self._conv_b
                who, color = self.label_b, self.color_b

            assert conv is not None  # for the type-checker
            conv.add_user(incoming)
            try:
                tokens = list(engine.stream_reply(conv, incoming))
            except Exception as exc:  # noqa: BLE001
                log.warning("test.engine_error", speaker=speaker, error=str(exc))
                if self._stop.wait(2.0):
                    return
                continue
            text = ("".join(tokens)).strip()
            if not text:
                text = "(no reply)"
            text = _trim_reply(text, max_chars=320)
            conv.add_assistant(text)

            worker.avatar.say(text)
            if self.chat_panel is not None:
                self.chat_panel.append_external_message(who, text, color=color)
            mood = _guess_emotion(text)
            bus.publish(EventType.EMOTION, Emotion(label=mood, confidence=0.7))

            incoming = text
            speaker = "b" if speaker == "a" else "a"
            self._wait_paced(text)

    # ── pacing ───────────────────────────────────────────────────────

    def _wait_paced(self, text: str) -> None:
        # Roughly proportional to talking time plus a small idle gap so
        # visemes finish before the next bot starts.
        wait = max(2.5, min(self.period_s + 4.0, 0.05 * len(text) + 2.5))
        slept = 0.0
        while slept < wait and not self._stop.is_set():
            time.sleep(0.1)
            slept += 0.1


# ── helpers ──────────────────────────────────────────────────────────


def _trim_reply(text: str, *, max_chars: int = 320) -> str:
    """Trim long replies so the dialogue stays paceable. Prefers a clean
    sentence boundary, falls back to a hard cut."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Try to cut at the last sentence end before max_chars.
    cut = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n"):
        idx = cut.rfind(sep)
        if idx > max_chars // 2:
            return cut[: idx + 1].strip()
    return cut.rstrip() + "…"
