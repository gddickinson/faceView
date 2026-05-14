"""Dual-LLM test-mode orchestrator lifecycle.

Test mode stands up a second avatar worker (in the *user*-camera
slot) plus a :class:`TestConversation` that drives two LLM clients
talking to each other. Engine choice is env-driven so the same UI
toggle works for canned, anthropic, ollama, and demo backends.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from faceview.config import settings
from faceview.core.events import EventType
from faceview.gui.controllers.base import BaseController

if TYPE_CHECKING:
    from faceview.llm.test_conversation import TestConversation
    from faceview.vision.sim_camera import SimCameraWorker


_SAFE_RENDER_MODES = {
    "stylised", "anatomical", "anatomy_overlay", "wireframe",
}


class TestModeController(BaseController):
    log_name = "test_mode_ctrl"

    def __init__(self, window) -> None:
        super().__init__(window)
        self._orchestrator: Optional["TestConversation"] = None
        self._user_avatar_worker: Optional["SimCameraWorker"] = None

    # ── public API ────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._orchestrator is not None

    def set_enabled(self, on: bool) -> None:
        if on and self._orchestrator is None:
            self._start()
        elif not on and self._orchestrator is not None:
            self._stop()

    def restart(self) -> None:
        """Stop and re-start so engine-config changes take effect."""
        if self._orchestrator is None:
            return
        self._stop()
        self._start()

    # ── internals ─────────────────────────────────────────────────

    def _start(self) -> None:
        from faceview.llm.test_conversation import TestConversation
        from faceview.vision.sim_camera import SimCameraWorker

        avatar_ctrl = self.window.avatar_ctrl
        # Ensure the Claude-side avatar is running.
        if not avatar_ctrl.is_running():
            avatar_ctrl.set_enabled(True)
            self.window.show_avatar_window()
        # Free the FRAME channel for the second bot avatar.
        self.window.camera_ctrl.set_enabled(False)
        partner_persona = self._partner_persona(avatar_ctrl.current_persona())
        self._user_avatar_worker = SimCameraWorker(
            scenario="avatar",
            emotion="neutral",
            persona=partner_persona,
            wire_to_llm=False,
            frame_channel=EventType.FRAME,
            publish_user_events=False,
        )
        self._user_avatar_worker.start()

        engine_a, engine_b, engine_name = self._build_engines()
        self._orchestrator = TestConversation(
            avatar_worker=avatar_ctrl.worker,
            user_worker=self._user_avatar_worker,
            engine_a=engine_a,
            engine_b=engine_b,
            chat_panel=self.window.chat,
            persona_a=partner_persona,
            persona_b=avatar_ctrl.current_persona(),
        )
        self._orchestrator.start()
        mode = ("LLM (" + engine_name + ")"
                if engine_a is not None else "canned")
        self.status(f"Test mode: two bots conversing — {mode}")
        self.window.refresh_llm_pill()

    def _stop(self) -> None:
        if self._orchestrator is not None:
            try:
                self._orchestrator.stop()
            except Exception:  # noqa: BLE001
                pass
            self._orchestrator = None
        if self._user_avatar_worker is not None:
            try:
                self._user_avatar_worker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._user_avatar_worker = None
        self.status("Test mode stopped")
        self.window.refresh_llm_pill()

    def _build_engines(self) -> tuple[object | None, object | None, str]:
        """Construct two engines based on env vars / settings.

        Honours ``FACEVIEW_TEST_ENGINE`` (canned / anthropic / ollama
        / demo) and ``FACEVIEW_TEST_MODEL`` (model id for the chosen
        engine)."""
        engine_name = (os.environ.get("FACEVIEW_TEST_ENGINE")
                       or "canned").lower()
        model = os.environ.get("FACEVIEW_TEST_MODEL") or None
        if engine_name in ("", "canned", "seed", "off"):
            return None, None, "canned"
        try:
            if engine_name == "anthropic":
                if not settings.has_claude_key:
                    raise RuntimeError("ANTHROPIC_API_KEY not set")
                from faceview.llm.claude_client import AnthropicEngine
                m = model or settings.anthropic_model
                return (
                    AnthropicEngine(
                        api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
                        model=m),
                    AnthropicEngine(
                        api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
                        model=m),
                    f"anthropic:{m}",
                )
            if engine_name == "ollama":
                from faceview.llm.ollama_client import (
                    OllamaEngine, pick_default_model,
                )
                m = model or pick_default_model()
                if not m:
                    raise RuntimeError("no ollama models installed")
                return (OllamaEngine(model=m), OllamaEngine(model=m),
                        f"ollama:{m}")
            if engine_name == "demo":
                from faceview.llm.claude_client import EchoEngine
                return EchoEngine(), EchoEngine(), "demo"
        except Exception as exc:  # noqa: BLE001
            self.log.warning("test_mode.engine_build_failed",
                             engine=engine_name, error=str(exc))
            self.status(
                f"Test mode: {engine_name} unavailable — falling back to canned"
            )
        return None, None, "canned"

    def _partner_persona(self, current: str) -> str:
        """Pick a persona for the partner bot.

        Explicit user override via ``FACEVIEW_TEST_PARTNER_PERSONA``
        wins. Otherwise prefer lightweight (non-ICT-3D, non-GPU)
        personas because running two heavy renderers in parallel
        races on moderngl's GL context and segfaults the process.
        """
        forced = os.environ.get("FACEVIEW_TEST_PARTNER_PERSONA")
        if forced and forced != current:
            return forced
        try:
            from faceview.llm.character import list_character_keys
            keys = [k for k in list_character_keys()
                    if k != current and not k.endswith("_fallback")]
        except Exception:  # noqa: BLE001
            keys = []
        safe: list[str] = []
        try:
            from faceview.vision.personas import load_persona
            for k in keys:
                try:
                    mode = (getattr(load_persona(k), "render_mode",
                                    "stylised") or "stylised")
                except Exception:  # noqa: BLE001
                    mode = "stylised"
                if mode in _SAFE_RENDER_MODES:
                    safe.append(k)
        except Exception:  # noqa: BLE001
            safe = keys
        pool = safe or keys
        if not pool:
            return "warm_tan" if current != "warm_tan" else "claude"
        idx = abs(hash(current)) % len(pool)
        return pool[idx]
