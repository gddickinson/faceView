"""I3 — webhook subscriptions for bus events.

Third-party tools register a callback URL + a list of event types;
faceView POSTs the JSON-serialised payload on each matching event.
Enables Home Assistant, Stream Deck, custom dashboards, etc.

Subscriptions persist in ``~/.faceview/webhooks.json`` as a list of:

    {"url": "...", "events": ["EMOTION", "GESTURE"], "id": "..."}

To wire from outside the GUI: POST to ``/webhooks`` with the same
dict shape. DELETE ``/webhooks/<id>`` to remove.

This module just owns the subscriber list + dispatcher. The HTTP
endpoints live in ``server/api.py``; they marshal requests onto a
shared :class:`WebhookManager` singleton."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from faceview.config import settings
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType
from faceview.core.logger import get_logger


log = get_logger("webhooks")


# Events we agree to forward — exclude high-volume ones (FRAME,
# AVATAR_FRAME, SCREEN_FRAME, AUDIO_CHUNK, LLM_TOKEN, AUDIO_AMPLITUDE)
# so a misconfigured subscriber can't ddos itself.
_DEFAULT_ALLOWED = {
    "PRESENCE", "IDENTITY", "IDENTITIES_MULTI", "EMOTION",
    "MOUTH_ACTIVITY", "HEAD_POSE", "GAZE", "FACE_DISTANCE",
    "BLINK", "GESTURE", "SCENE", "SCENE_CAPTION", "OBJECTS",
    "CHAT_USER_MESSAGE", "LLM_REPLY", "LLM_ERROR", "CHAT_LOG",
    "TTS_STARTED", "TTS_FINISHED", "ROOM_MAP", "PIXELS_LEAVING",
    "TURN_RECORDED", "VAD_SPEECH_START", "VAD_SPEECH_END",
    "TRANSCRIPT_FINAL", "STATUS",
}


@dataclass
class WebhookSub:
    id: str
    url: str
    events: list[str]
    created_at: float = field(default_factory=time.time)


class WebhookManager:
    """Singleton — owns subscriptions + dispatch loop."""

    _instance: "WebhookManager | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "WebhookManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = WebhookManager()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: dict[str, WebhookSub] = {}
        self._load()
        # Subscribe to every allowed event type once so we can
        # forward. The dispatcher itself checks per-sub interest
        # before POSTing so high-fanout events are still cheap.
        bus = get_bus()
        for name in _DEFAULT_ALLOWED:
            et = getattr(EventType, name, None)
            if et is not None:
                bus.subscribe(et, self._dispatch_for(name))

    # ── persistence ─────────────────────────────────────────

    def _path(self) -> Path:
        return settings.data_dir / "webhooks.json"

    def _load(self) -> None:
        p = self._path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            for entry in data.get("subs") or []:
                sub = WebhookSub(
                    id=str(entry.get("id") or uuid.uuid4().hex),
                    url=str(entry.get("url") or ""),
                    events=list(entry.get("events") or []),
                    created_at=float(entry.get("created_at")
                                     or time.time()),
                )
                if sub.url:
                    self._subs[sub.id] = sub
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            log.warning("webhooks.load_failed", error=str(exc))

    def _persist(self) -> None:
        try:
            p = self._path()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            with self._lock:
                payload = {
                    "subs": [asdict(s) for s in self._subs.values()],
                    "saved_at": time.time(),
                }
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(p)
        except OSError as exc:
            log.warning("webhooks.save_failed", error=str(exc))

    # ── public API ──────────────────────────────────────────

    def register(self, url: str, events: list[str]) -> WebhookSub:
        if not url or not url.startswith(("http://", "https://")):
            raise ValueError("webhook url must be http:// or https://")
        events = [e.upper() for e in (events or []) if e]
        events = [e for e in events if e in _DEFAULT_ALLOWED]
        sub = WebhookSub(
            id=uuid.uuid4().hex[:12], url=url, events=events,
        )
        with self._lock:
            self._subs[sub.id] = sub
        self._persist()
        log.info("webhooks.registered", id=sub.id, url=url,
                 events=events)
        return sub

    def unregister(self, sub_id: str) -> bool:
        with self._lock:
            existed = self._subs.pop(sub_id, None) is not None
        if existed:
            self._persist()
        return existed

    def list_subs(self) -> list[WebhookSub]:
        with self._lock:
            return list(self._subs.values())

    # ── dispatch ────────────────────────────────────────────

    def _dispatch_for(self, event_name: str):
        """Return a handler closure for the given event name. The
        bus calls it; we filter by sub interest and POST in a thread
        so a slow webhook doesn't block the GUI."""
        def _handler(payload) -> None:
            with self._lock:
                interested = [
                    s for s in self._subs.values()
                    if event_name in s.events or not s.events
                ]
            if not interested:
                return
            body = _serialise(payload)
            for sub in interested:
                t = threading.Thread(
                    target=_post,
                    args=(sub.url, event_name, body),
                    daemon=True,
                    name=f"webhook-{sub.id}",
                )
                t.start()
        return _handler


# ── helpers ───────────────────────────────────────────────────────


def _serialise(payload: Any) -> dict:
    """Best-effort: payload is usually a dataclass; fall back to
    str() for opaque types."""
    try:
        if hasattr(payload, "__dataclass_fields__"):
            return asdict(payload)
    except Exception:  # noqa: BLE001
        pass
    if isinstance(payload, dict):
        return payload
    return {"value": str(payload)}


def _post(url: str, event_name: str, body: dict) -> None:
    try:
        data = json.dumps({
            "event": event_name,
            "ts": time.time(),
            "payload": body,
        }, default=str).encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("webhooks.serialise_failed", error=str(exc))
        return
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=2.0)
    except (urllib.error.URLError, ConnectionError, TimeoutError,
            OSError) as exc:
        log.warning("webhooks.delivery_failed", url=url,
                    event=event_name, error=str(exc))
