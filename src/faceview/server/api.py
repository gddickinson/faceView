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


class EmotionRequest(BaseModel):
    name: str


class PersonaRequest(BaseModel):
    name: str


class EffectRequest(BaseModel):
    name: str
    intensity: float = 1.0
    duration: float | None = None


class SliderRequest(BaseModel):
    key: str
    value: float | str


class AvatarSayRequest(BaseModel):
    text: str
    speed: float = 1.0


class EngineRequest(BaseModel):
    engine: str               # "auto" | "anthropic" | "ollama" | "demo"
    model: str | None = None


class TestEngineRequest(BaseModel):
    engine: str               # "canned" | "ollama" | "anthropic" | "demo"
    model: str | None = None


class LifecycleRequest(BaseModel):
    name: str   # "camera" | "mic" | "tts" | "avatar" | "test_mode" | "mirror"
    on: bool


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

    @app.get("/chat/log")
    def chat_log(n: int = 50) -> dict[str, Any]:
        return {"ok": True, "chat": service.list_chat_log(n=n)}

    @app.get("/monitor")
    def monitor(chat_n: int = 20, events_n: int = 30) -> dict[str, Any]:
        return service.monitor_snapshot(chat_n=chat_n, events_n=events_n)

    @app.post("/llm/engine")
    def set_engine(req: EngineRequest) -> dict[str, Any]:
        return service.set_engine(req.engine, model=req.model)

    @app.post("/test/engine")
    def set_test_engine(req: TestEngineRequest) -> dict[str, Any]:
        return service.set_test_engine(req.engine, model=req.model)

    @app.post("/lifecycle")
    def lifecycle(req: LifecycleRequest) -> dict[str, Any]:
        return service.set_lifecycle(req.name, req.on)

    @app.post("/shutdown")
    def shutdown() -> dict[str, Any]:
        return service.shutdown()

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        return service.send_chat(req.text)

    @app.post("/speak")
    def speak(req: SpeakRequest) -> dict[str, Any]:
        return service.speak(req.text)

    @app.post("/screenshot")
    def screenshot(req: ScreenshotRequest) -> dict[str, Any]:
        return service.screenshot(req.name, encode_b64=req.encode_b64)

    @app.post("/avatar/emotion")
    def set_emotion(req: EmotionRequest) -> dict[str, Any]:
        return service.set_emotion(req.name)

    @app.post("/avatar/persona")
    def set_persona(req: PersonaRequest) -> dict[str, Any]:
        return service.set_persona(req.name)

    @app.post("/avatar/say")
    def avatar_say(req: AvatarSayRequest) -> dict[str, Any]:
        return service.avatar_say(req.text, speed=req.speed)

    @app.get("/avatar/personas")
    def list_personas() -> dict[str, Any]:
        return {"ok": True, "personas": service.list_personas()}

    @app.get("/effects")
    def list_effects() -> dict[str, Any]:
        return {"ok": True, "effects": service.list_effects()}

    @app.get("/effects/active")
    def list_active_effects() -> dict[str, Any]:
        return {"ok": True, "active": service.list_active_effects()}

    @app.post("/effects/trigger")
    def trigger_effect(req: EffectRequest) -> dict[str, Any]:
        return service.trigger_effect(
            req.name, intensity=req.intensity, duration=req.duration,
        )

    @app.post("/effects/stop")
    def stop_effect(req: EffectRequest) -> dict[str, Any]:
        return service.stop_effect(req.name)

    @app.post("/effects/stop_all")
    def stop_all_effects() -> dict[str, Any]:
        return service.stop_all_effects()

    @app.get("/effects/sliders")
    def get_sliders() -> dict[str, Any]:
        return {"ok": True, "sliders": service.get_sliders()}

    @app.post("/effects/slider")
    def set_slider(req: SliderRequest) -> dict[str, Any]:
        return service.set_slider(req.key, req.value)

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
