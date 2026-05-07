"""Application wiring.

Constructs the QApplication, MainWindow, optional speech/vision/llm/server
workers, and runs the Qt event loop. Heavy ML modules are imported lazily so
this file imports cleanly even when only the minimum extras are installed.

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


def main(argv: Optional[list[str]] = None) -> int:
    configure_logging()
    _maybe_enable_offscreen()

    # Import PySide6 only after the offscreen flag is set.
    from PySide6.QtWidgets import QApplication
    from faceview.gui.main_window import MainWindow

    argv = argv if argv is not None else sys.argv
    app = QApplication.instance() or QApplication(argv)

    window = MainWindow()

    # Wire LLM client to chat events. The client itself decides whether to
    # call Anthropic or fall back to demo mode.
    from faceview.llm.claude_client import ClaudeClient
    client = ClaudeClient()

    bus = get_bus()
    bus.subscribe(
        EventType.CHAT_USER_MESSAGE,
        lambda msg: client.send_async(msg) if isinstance(msg, ChatMessage) and msg.role == "user" else None,
    )

    # Optional: avatar mode — talking head driven by Claude replies, shown
    # in the camera panel. Toggle with ``FACEVIEW_AVATAR=1``. Created before
    # the API server so the Service can bind it for set_emotion / set_persona
    # / avatar_say.
    avatar_worker = None
    if os.environ.get("FACEVIEW_AVATAR", "").strip().lower() in {"1", "true", "yes", "on"}:
        from faceview.vision.sim_camera import SimCameraWorker
        # Default to xray-male-young (sci-fi look without anatomy
        # underlay) when ICT-FaceKit data is present; fall back to
        # stylised 2D otherwise. Mood-driven skin tone + glowing
        # eye pulse + bloom halo + subtle talking sway are wired in
        # automatically.
        from faceview.vision.ict_face import _data_path as _ict_data
        default_persona = "ict_xray_young" if _ict_data().exists() else "claude"
        avatar_worker = SimCameraWorker(
            scenario="avatar", emotion="happy", persona=default_persona,
            wire_to_llm=True,
        )
        avatar_worker.start()
        log.info("avatar.started")

    # Optional control server (FastAPI on 127.0.0.1).
    if settings.api_enabled and not settings.headless:
        try:
            from faceview.server.api import start_api_server
            from faceview.server.service import get_service
            start_api_server(window)
            if avatar_worker is not None:
                get_service().bind_camera_worker(avatar_worker)
            log.info("api.started", url=settings.api_url)
        except Exception as exc:  # noqa: BLE001
            log.warning("api.start_failed", error=str(exc))

    log.info(
        "boot",
        headless=settings.headless,
        claude_key=settings.has_claude_key,
        camera_index=settings.camera_index,
        avatar=avatar_worker is not None,
    )

    if not settings.headless:
        window.show()

    return app.exec()
