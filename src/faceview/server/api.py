"""FastAPI HTTP control surface.

Bound to 127.0.0.1 only — this is a local control plane for a Claude Code
session running on the same machine, not a public API. Endpoints intentionally
mirror the :class:`Service` ops 1:1 so the MCP server can stay just as thin.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from faceview.config import settings
from faceview.core.logger import get_logger
from faceview.server.service import Service, init_service


log = get_logger("api")


class ChatRequest(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str


class ScreenshotRequest(BaseModel):
    name: str = "shot.png"
    encode_b64: bool = False


def build_app(service: Service) -> FastAPI:
    app = FastAPI(title="faceView Control API", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "service": "faceview"}

    @app.get("/state")
    def state() -> dict[str, Any]:
        return service.get_camera_state()

    @app.get("/events")
    def events(n: int = 50) -> list[dict[str, Any]]:
        return service.list_events(n=n)

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        return service.send_chat(req.text)

    @app.post("/speak")
    def speak(req: SpeakRequest) -> dict[str, Any]:
        return service.speak(req.text)

    @app.post("/screenshot")
    def screenshot(req: ScreenshotRequest) -> dict[str, Any]:
        return service.screenshot(req.name, encode_b64=req.encode_b64)

    return app


def start_api_server(window) -> threading.Thread:
    """Launch uvicorn in a background thread bound to 127.0.0.1.

    Returns the server thread. The thread is daemonised so it dies with the
    GUI; we don't bother gracefully shutting uvicorn down.
    """
    service = init_service(window)
    app = build_app(service)

    import uvicorn

    config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        try:
            server.run()
        except Exception as exc:  # noqa: BLE001
            log.warning("api.run_failed", error=str(exc))

    t = threading.Thread(target=_run, name="faceview-api", daemon=True)
    t.start()
    return t
