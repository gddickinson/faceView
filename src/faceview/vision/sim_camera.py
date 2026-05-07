"""Simulated camera: drives :func:`render_face` on a thread and posts frames.

Public API matches :class:`vision.camera.CameraWorker` so the rest of the
pipeline doesn't care which one is wired in.

This is what tests, headless smoke runs, and README screenshot capture use
to exercise the full chain (frame → presence → mouth → status panel) without
needing a webcam or microphone.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import (
    ChatMessage,
    Emotion,
    EventType,
    Identity,
    MouthActivity,
    Presence,
)
from faceview.core.logger import get_logger
from faceview.vision.avatar import TalkingAvatar
from faceview.vision.sim_face import FaceParams, render_face


log = get_logger("sim_camera")


class SimCameraWorker:
    """Generates synthetic frames + matching presence/mouth/emotion events.

    By posting *both* the frames and the analysed events, the full event
    surface looks identical to a live pipeline — which is exactly what we
    want for end-to-end tests and screenshots.
    """

    def __init__(
        self,
        size: tuple[int, int] = (640, 480),
        fps: int = 24,
        scenario: str = "talking",
        *,
        emotion: str = "neutral",
        persona: str = "default",
        wire_to_llm: bool = False,
    ) -> None:
        self.size = size
        self.fps = fps
        self.scenario = scenario
        self.wire_to_llm = wire_to_llm
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._params = FaceParams.neutral()
        self._t0 = 0.0
        self._last_emotion = ""

        # The talking avatar is created up front so external callers can call
        # ``worker.avatar.say(text)`` even before ``start()``. Persona drives
        # appearance + render mode (ICT face when ICT data is built locally).
        self.avatar = TalkingAvatar(emotion=emotion, persona=persona, seed=42)
        if wire_to_llm:
            self._wire_llm_chat()

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="sim-camera", daemon=True)
        self._thread.start()
        log.info("sim_camera.started", scenario=self.scenario, fps=self.fps)

    def stop(self) -> None:
        self._stop.set()
        log.info("sim_camera.stopped")

    # ── manual override ─────────────────────────────────────────────

    def set_params(self, params: FaceParams) -> None:
        self._params = params

    def say(self, text: str, *, speed: float = 1.0) -> None:
        """Drive the avatar to mouth ``text`` (only used in the avatar scenario)."""
        self.avatar.say(text, speed=speed)

    def _wire_llm_chat(self) -> None:
        bus = get_bus()
        # Mouth Claude's full reply when it's complete (one say() call so the
        # mouth animation aligns with the text rather than per-token jitter).
        bus.subscribe(EventType.LLM_REPLY, self._on_llm_reply)

    def _on_llm_reply(self, msg) -> None:
        text = getattr(msg, "content", "") if isinstance(msg, ChatMessage) else str(msg)
        if text:
            self.avatar.say(text)

    # ── loop ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        bus = get_bus()
        period = 1.0 / max(1, self.fps)
        while not self._stop.is_set():
            t = time.time() - self._t0
            params = self._scenario_params(t)
            self._params = params

            frame = render_face(params, self.size)
            bus.publish(EventType.FRAME, frame)

            # Post derived events so the status panel updates as if the
            # vision pipeline had analysed the frame.
            bus.publish(
                EventType.PRESENCE,
                Presence(
                    face_count=1,
                    bboxes=[(self.size[0] // 4, self.size[1] // 4, self.size[0] // 2, self.size[1] // 2)],
                ),
            )
            speaking = params.jaw_open > 0.07
            viseme = self._viseme_for(params)
            bus.publish(
                EventType.MOUTH_ACTIVITY,
                MouthActivity(
                    speaking=speaking,
                    jaw_open=params.jaw_open,
                    mouth_funnel=max(0.0, params.jaw_open - 0.05),
                    mouth_pucker=max(0.0, -params.smile) * 0.6,
                    viseme=viseme,
                ),
            )
            label, conf = self._emotion_for(params)
            if label != self._last_emotion:
                bus.publish(
                    EventType.EMOTION,
                    Emotion(label=label, confidence=conf, scores={label: conf}),
                )
                self._last_emotion = label

            # Identity heartbeat (sim → "owner" with a stable similarity).
            if int(t * 2) != int((t - period) * 2):
                bus.publish(
                    EventType.IDENTITY,
                    Identity(is_owner=True, similarity=0.71, label="sim-owner"),
                )

            time.sleep(period)

    # ── scenarios ──────────────────────────────────────────────────

    def _scenario_params(self, t: float) -> FaceParams:
        s = self.scenario
        # AU-based avatar — handles its own idle behaviour and any active
        # utterance scheduled via avatar.say().
        if s == "avatar":
            return self.avatar.tick(t)
        if s == "neutral":
            return FaceParams.neutral()
        if s == "happy":
            return FaceParams.happy()
        if s == "surprised":
            return FaceParams.surprised()
        if s == "sad":
            return FaceParams.sad()
        if s == "blink":
            blink = 1.0
            phase = t % 4.0
            if 0.0 <= phase < 0.18:
                blink = max(0.05, 1.0 - phase / 0.09)
            elif 0.18 <= phase < 0.36:
                blink = max(0.05, (phase - 0.18) / 0.18)
            return FaceParams(eye_open=blink, smile=0.2)
        # default: talking with subtle gaze drift
        p = FaceParams.speaking(t)
        p.smile = 0.25 + 0.05 * (1 if (t // 4) % 2 == 0 else -1)
        return p

    def _viseme_for(self, params: FaceParams) -> Optional[str]:
        if params.jaw_open < 0.07:
            return None
        if params.jaw_open > 0.28:
            return "AA"
        if params.smile < -0.1:
            return "OO"
        if params.smile > 0.3:
            return "EE"
        return "FV"

    def _emotion_for(self, params: FaceParams) -> tuple[str, float]:
        if params.brow_raise > 0.6 and params.jaw_open > 0.3:
            return "surprise", 0.82
        if params.smile > 0.5:
            return "happy", 0.78
        if params.smile < -0.3:
            return "sad", 0.65
        return "neutral", 0.72
