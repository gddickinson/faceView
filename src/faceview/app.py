"""Application wiring.

Constructs the QApplication, MainWindow (the user-facing webcam side),
the AvatarWindow (Claude's face), and the optional speech / vision /
LLM / server workers. Heavy ML modules are imported lazily so this file
imports cleanly even when only the minimum extras are installed.

Run with::

    faceview                 # GUI
    python -m faceview       # equivalent
    FACEVIEW_HEADLESS=1 faceview   # offscreen
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from faceview.config import settings
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, ChatMessage
from faceview.core.logger import configure as configure_logging, get_logger
from faceview.utils.headless import enable_offscreen


log = get_logger("app")


def _maybe_enable_offscreen() -> None:
    if settings.headless:
        enable_offscreen()


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main(argv: Optional[list[str]] = None) -> int:
    configure_logging()
    _maybe_enable_offscreen()

    from PySide6.QtWidgets import QApplication
    from faceview.gui.main_window import MainWindow

    argv = argv if argv is not None else sys.argv
    app = QApplication.instance() or QApplication(argv)

    # U6 — restore persisted theme before any widget paints.
    try:
        from faceview.gui.theme import apply_theme, load_persisted
        apply_theme(load_persisted())
    except Exception as exc:  # noqa: BLE001
        log.warning("theme.apply_failed", error=str(exc))

    # I4 — discover user plugins under ~/.faceview/plugins.
    try:
        from faceview.llm.plugins import discover_and_load_plugins
        n = discover_and_load_plugins()
        if n:
            log.info("plugins.discovered", count=n)
    except Exception as exc:  # noqa: BLE001
        log.warning("plugins.discover_failed", error=str(exc))

    window = MainWindow()

    # Wire LLM client to chat events. Stash on the window so the config
    # dialog can swap engines / models live.
    from faceview.llm.claude_client import ClaudeClient
    client = ClaudeClient()
    window.llm_client = client
    # Live perception (presence/emotion/gaze/gestures/scene/...) — gets
    # prepended to the system prompt on every turn so chat bots have
    # ambient awareness of what the camera sees. Added BEFORE
    # bind_memory so the persona swap path doesn't disturb it.
    try:
        from faceview.vision.perception import PerceptionStore
        store = PerceptionStore.shared()
        client.conversation.add_system_extras_provider(store.narrate_now)
    except Exception as exc:  # noqa: BLE001
        log.warning("perception.attach_failed", error=str(exc))
    # Load persistent memory for the current persona so chats build up
    # context across sessions (and across LLM engines).
    window._bind_memory_for_current_persona()
    # C8 — user-emotion → persona-emotion feedback loop. Singleton
    # subscribes to EMOTION events; we attach the client so it can
    # reach the live cognition store.
    try:
        from faceview.vision.emotion_feedback import EmotionFeedback
        EmotionFeedback.shared().attach(client)
    except Exception as exc:  # noqa: BLE001
        log.warning("emotion_feedback.attach_failed", error=str(exc))
    # Show the actual engine on the LLM pill from boot, not just the
    # initial label baked into StatusPanel.__init__ (which assumes
    # anthropic-or-demo).
    window.refresh_llm_pill()

    bus = get_bus()
    bus.subscribe(
        EventType.CHAT_USER_MESSAGE,
        lambda msg: client.send_async(msg) if isinstance(msg, ChatMessage) and msg.role == "user" else None,
    )

    # Start the real webcam (the user side) by default. Toggleable via
    # FACEVIEW_AUTOCAM and from Tools → Configuration… while running.
    if not settings.headless and settings.auto_start_camera:
        window.set_camera_enabled(True)

    # Optional: microphone / STT. Off by default — speech extras are
    # heavy. Users opt in via FACEVIEW_AUTOMIC or the config dialog.
    if not settings.headless and settings.auto_start_audio:
        window.set_audio_enabled(True)

    # The Claude avatar (separate window) is on by default — that is the
    # whole point of this GUI. Disable with FACEVIEW_AVATAR=0.
    avatar_on = _env_truthy("FACEVIEW_AVATAR", default=True)
    if avatar_on and not settings.headless:
        window.set_avatar_enabled(True)

    # Optional dual-Claude test mode at boot: FACEVIEW_TEST_MODE=1.
    if not settings.headless and _env_truthy("FACEVIEW_TEST_MODE", default=False):
        window.set_test_mode_enabled(True)

    # Optional control server (FastAPI on 127.0.0.1).
    if settings.api_enabled and not settings.headless:
        try:
            from faceview.server.api import start_api_server
            start_api_server(window)
            log.info("api.started", url=settings.api_url)
        except Exception as exc:  # noqa: BLE001
            log.warning("api.start_failed", error=str(exc))

    log.info(
        "boot",
        headless=settings.headless,
        claude_key=settings.has_claude_key,
        camera_index=settings.camera_index,
        avatar=avatar_on,
    )

    if not settings.headless:
        window.show()

    return app.exec()
