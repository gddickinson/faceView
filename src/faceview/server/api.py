"""FastAPI HTTP control surface.

Bound to 127.0.0.1 only — this is a local control plane for a Claude Code
session running on the same machine, not a public API. Endpoints intentionally
mirror the :class:`Service` ops 1:1 so the MCP server can stay just as thin.

Optional token auth (S7): set ``FACEVIEW_API_TOKEN`` and every request
must carry ``X-API-Token: <token>`` (or ``Authorization: Bearer <token>``).
Unset → no auth, same as before. Local-only by default so this is
defence-in-depth for shared-host scenarios.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from faceview.config import settings
from faceview.core.logger import get_logger
from faceview.server.openai_compat import build_router as _openai_router
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


class WebhookRegisterRequest(BaseModel):
    url: str
    events: list[str] = []


class ConsentToolRequest(BaseModel):
    tool: str
    decision: str  # "allow" | "block" | "prompt"


class ConsentTrustRequest(BaseModel):
    on: bool


def _expected_token() -> str | None:
    """Read FACEVIEW_API_TOKEN at request time so toggling it doesn't
    need a restart. Empty / unset → no auth."""
    raw = os.environ.get("FACEVIEW_API_TOKEN") or ""
    raw = raw.strip()
    return raw or None


# Endpoints that bypass auth (so SDKs can probe + shutdown stays
# reachable from the GUI process even if it forgets the token).
_AUTH_BYPASS = {"/healthz"}


def build_app(service: Service) -> FastAPI:
    app = FastAPI(title="faceView Control API", version="0.1.0")

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        expected = _expected_token()
        if expected is None:
            # Auth not configured — open as before.
            return await call_next(request)
        if request.url.path in _AUTH_BYPASS:
            return await call_next(request)
        # Accept either header form.
        header = (request.headers.get("x-api-token")
                  or request.headers.get("authorization") or "")
        if header.lower().startswith("bearer "):
            header = header[7:]
        if header.strip() != expected:
            return _json_response(
                {"ok": False, "error": "missing or invalid API token"},
                status=401,
            )
        return await call_next(request)

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

    @app.get("/memory")
    def memory(recent_n: int = 20) -> dict[str, Any]:
        return service.get_memory(recent_n=recent_n)

    @app.post("/memory/clear")
    def memory_clear() -> dict[str, Any]:
        return service.clear_memory()

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

    @app.get("/chat/export")
    def chat_export(n: int = 10_000) -> dict[str, Any]:
        return service.export_chat(n=n)

    # ── webhooks (I3) ────────────────────────────────────────

    @app.get("/webhooks")
    def webhooks_list() -> dict[str, Any]:
        from faceview.core.webhooks import WebhookManager
        subs = WebhookManager.shared().list_subs()
        return {"ok": True, "subs": [
            {"id": s.id, "url": s.url, "events": s.events,
             "created_at": s.created_at}
            for s in subs
        ]}

    @app.post("/webhooks")
    def webhooks_register(req: WebhookRegisterRequest) -> dict[str, Any]:
        from faceview.core.webhooks import WebhookManager
        try:
            sub = WebhookManager.shared().register(req.url, req.events)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True, "id": sub.id, "url": sub.url,
                "events": sub.events}

    @app.delete("/webhooks/{sub_id}")
    def webhooks_unregister(sub_id: str) -> dict[str, Any]:
        from faceview.core.webhooks import WebhookManager
        ok = WebhookManager.shared().unregister(sub_id)
        if not ok:
            raise HTTPException(status_code=404,
                                 detail=f"no webhook with id {sub_id}")
        return {"ok": True, "removed": sub_id}

    # ── consent (PR2) ────────────────────────────────────────

    @app.get("/consent")
    def consent_get() -> dict[str, Any]:
        from faceview.core.consent import ConsentStore
        c = ConsentStore.shared()
        return {
            "ok": True,
            "trust_remote": c.trust_remote(),
            "tools": dict(c._tool_decisions),
        }

    @app.post("/consent/tool")
    def consent_set_tool(req: ConsentToolRequest) -> dict[str, Any]:
        from faceview.core.consent import ConsentStore
        if req.decision not in ("allow", "block", "prompt"):
            raise HTTPException(status_code=400,
                                 detail="decision must be allow/block/prompt")
        ConsentStore.shared().set_tool_decision(req.tool, req.decision)
        return {"ok": True}

    @app.post("/consent/trust_remote")
    def consent_trust_remote(req: ConsentTrustRequest) -> dict[str, Any]:
        from faceview.core.consent import ConsentStore
        ConsentStore.shared().set_trust_remote(bool(req.on))
        return {"ok": True, "trust_remote": req.on}

    # OpenAI-compat /v1/chat/completions + /v1/models.
    app.include_router(_openai_router(service))

    return app


def _json_response(body: dict[str, Any], status: int = 200):
    """Tiny helper since the middleware returns before FastAPI's
    auto-serialisation kicks in."""
    from fastapi.responses import JSONResponse
    return JSONResponse(content=body, status_code=status)


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
