"""Ambient VLM scene captioner.

Runs a small vision-language model (``moondream`` by default) on the
latest webcam frame at a slow cadence (~15 s) so the LLM has an
always-fresh natural-language description of the camera view to read
from the perception block. This is the **continuous monitoring**
half of the two-tier vision system:

* Fast, light, ambient → this module (moondream, ~2 s round-trip)
* Slow, rich, on-demand → ``llm/vision_tool.py`` ``look_at_camera``
  tool with optional ``question`` + ``region`` args (preferring
  ``llama3.2-vision`` or other heavy VLMs when installed).

Throttling rules:

* Skip cycles when ``presence.face_count == 0`` for the last 30 s
  (the camera is staring at an empty room — no caption needed).
* Skip when the scene has been "still" for the last interval (no
  motion → caption hasn't changed).
* Hard floor of :attr:`interval_s` between caption attempts so we
  never overrun a busy CPU.

Disable with ``FACEVIEW_AMBIENT_VLM=0``. Tune interval via
``FACEVIEW_AMBIENT_VLM_INTERVAL`` (seconds).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import (
    EventType, Presence, SceneCaption, SceneInfo,
)
from faceview.core.logger import get_logger


log = get_logger("scene.caption")


DEFAULT_INTERVAL_S = 15.0


def ambient_vlm_enabled() -> bool:
    raw = os.environ.get("FACEVIEW_AMBIENT_VLM")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _interval_from_env() -> float:
    raw = os.environ.get("FACEVIEW_AMBIENT_VLM_INTERVAL")
    if raw:
        try:
            return max(3.0, float(raw))
        except ValueError:
            pass
    return DEFAULT_INTERVAL_S


class SceneCaptioner:
    """Background thread: periodic VLM caption of the latest frame."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: str = "http://127.0.0.1:11434",
        interval_s: Optional[float] = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.host = host
        self.model = model or ""
        self.interval_s = interval_s or _interval_from_env()
        self.timeout_s = timeout_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Throttling state.
        self._last_attempt_ts = 0.0
        self._last_presence_at = 0.0
        self._last_motion: float = 0.0
        bus = get_bus()
        bus.subscribe(EventType.PRESENCE, self._on_presence)
        bus.subscribe(EventType.SCENE, self._on_scene)

    def _on_presence(self, p: Presence) -> None:
        if p is None:
            return
        if p.face_count > 0:
            self._last_presence_at = time.time()

    def _on_scene(self, s: SceneInfo) -> None:
        if s is None:
            return
        self._last_motion = float(s.motion)

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> bool:
        if not ambient_vlm_enabled():
            log.info("scene.caption.disabled")
            return False
        if self._thread is not None:
            return True
        # Pick a model — late so it can use the live Ollama install.
        if not self.model:
            self.model = self._pick_model()
        if not self.model:
            log.info("scene.caption.no_model")
            return False
        self._thread = threading.Thread(
            target=self._loop, name="scene-caption", daemon=True,
        )
        self._thread.start()
        log.info("scene.caption.started",
                 model=self.model, interval_s=self.interval_s)
        return True

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None

    # ── pick the right VLM ─────────────────────────────────────────

    def _pick_model(self) -> str:
        env = os.environ.get("FACEVIEW_AMBIENT_VLM_MODEL")
        if env:
            return env
        # Prefer fast / small captioners for the ambient cadence.
        try:
            from faceview.llm.ollama_client import list_ollama_models
            models = list_ollama_models(self.host)
        except Exception:  # noqa: BLE001
            models = []
        for needle in ("moondream", "llava-phi", "llava:7b", "llava"):
            for m in models:
                if needle in m.lower():
                    return m
        return ""

    # ── inner loop ─────────────────────────────────────────────────

    def _loop(self) -> None:
        from faceview.llm.vision_tool import FrameGrabber
        grabber = FrameGrabber.shared()
        bus = get_bus()
        # First caption: small delay so the camera has a chance to
        # produce a frame.
        time.sleep(min(self.interval_s, 5.0))
        while not self._stop.is_set():
            try:
                self._tick(grabber, bus)
            except Exception as exc:  # noqa: BLE001
                log.warning("scene.caption.tick_error", error=str(exc))
            # Wait for the next interval; check stop flag every second
            # so shutdown is responsive.
            for _ in range(int(self.interval_s)):
                if self._stop.is_set():
                    return
                time.sleep(1.0)

    def _tick(self, grabber, bus) -> None:
        now = time.time()
        # 1) Skip if no recent presence.
        if (self._last_presence_at == 0.0
                or now - self._last_presence_at > 30.0):
            return
        # 2) Skip if scene is still (motion < 0.05 for the whole gap).
        # We allow at least one caption per 5×interval regardless so
        # the model isn't stuck on an outdated description.
        force = now - self._last_attempt_ts > self.interval_s * 5
        if self._last_motion < 0.05 and not force:
            return

        pair = grabber.latest_jpeg_b64(max_dim=512, quality=70)
        if pair is None:
            return
        b64, _source = pair
        prompt = (
            "In one short sentence (under 25 words), describe what's "
            "currently happening in this webcam view. Focus on people, "
            "what they're doing, and visible objects. Be specific."
        )
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"num_predict": 80},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read())
            text = (data.get("response") or "").strip()
        except (urllib.error.URLError, ConnectionError, TimeoutError,
                OSError, ValueError) as exc:
            log.warning("scene.caption.vlm_failed", error=str(exc))
            return
        latency = time.time() - t0
        self._last_attempt_ts = time.time()
        if not text:
            return
        bus.publish(
            EventType.SCENE_CAPTION,
            SceneCaption(text=text, model=self.model, latency_s=latency),
        )
        log.info("scene.caption.published",
                 chars=len(text), latency_s=round(latency, 2))
