"""Shared service layer used by both the HTTP API and the MCP server.

All ops are designed to be safe to call from any thread: anything that
touches Qt widgets is marshalled to the GUI thread via ``QMetaObject.invokeMethod``
or by routing through the event bus.
"""

from __future__ import annotations

import base64
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QMetaObject, Qt, QThread, Q_ARG, QObject, Slot
from PySide6.QtGui import QPixmap

from faceview.core.event_bus import get_bus
from faceview.core.events import ChatMessage, EventType
from faceview.core.logger import get_logger
from faceview.utils.paths import docs_image_dir


log = get_logger("service")


@dataclass
class CameraState:
    presence: dict[str, Any] = field(default_factory=lambda: {"face_count": 0})
    identity: dict[str, Any] = field(default_factory=lambda: {"is_owner": False, "label": "unknown", "similarity": 0.0})
    emotion: dict[str, Any] = field(default_factory=lambda: {"label": "unknown", "confidence": 0.0})
    mouth: dict[str, Any] = field(default_factory=lambda: {"speaking": False, "viseme": None})
    last_frame_ts: float = 0.0


@dataclass
class EventLogEntry:
    type: str
    payload: Any
    ts: float


class _GuiBridge(QObject):
    """Helper that lives on the GUI thread; receives invoke calls."""

    def __init__(self, window) -> None:
        super().__init__()
        self._window = window
        self._last_pix: QPixmap | None = None
        self._evt = threading.Event()

    @Slot(str)
    def take_screenshot(self, name: str) -> None:
        try:
            path = self._window.take_screenshot(name)
            self._last_pix = QPixmap(str(path))
        finally:
            self._evt.set()

    def reset(self) -> None:
        self._last_pix = None
        self._evt.clear()


class Service:
    """Process-global controller wrapping the GUI for HTTP/MCP adapters."""

    def __init__(self, window) -> None:
        self.window = window
        self.bridge = _GuiBridge(window)
        self.bridge.moveToThread(window.thread())  # GUI thread

        self.bus = get_bus()
        self.camera_state = CameraState()
        self.events: deque[EventLogEntry] = deque(maxlen=500)
        self._camera_worker = None  # bound later if a SimCameraWorker is in use

        self._wire()

    def bind_camera_worker(self, worker) -> None:
        """Attach the running camera worker so avatar ops can reach it.

        Currently only :class:`SimCameraWorker` exposes an ``avatar`` attribute
        suitable for ``set_emotion`` / ``set_persona`` / ``say``. Real-camera
        workers leave this unset and the avatar ops return a clean error.
        """
        self._camera_worker = worker

    def _wire(self) -> None:
        b = self.bus
        b.subscribe(EventType.PRESENCE, self._on_presence)
        b.subscribe(EventType.IDENTITY, self._on_identity)
        b.subscribe(EventType.EMOTION, self._on_emotion)
        b.subscribe(EventType.MOUTH_ACTIVITY, self._on_mouth)
        b.subscribe(EventType.FRAME, self._on_frame)
        for et in EventType:
            b.subscribe(et, lambda p, t=et: self._log_event(t, p))

    # ── ops ─────────────────────────────────────────────────────────

    def send_chat(self, text: str) -> dict[str, Any]:
        if not text or not text.strip():
            return {"ok": False, "error": "empty message"}
        self.bus.publish(EventType.CHAT_USER_MESSAGE, ChatMessage("user", text))
        return {"ok": True, "queued": True}

    def speak(self, text: str) -> dict[str, Any]:
        self.bus.publish(EventType.TTS_SPEAK, text)
        return {"ok": True, "queued": True}

    # ── avatar ops ─────────────────────────────────────────────────

    def _avatar(self):
        worker = self._camera_worker
        if worker is None:
            return None
        return getattr(worker, "avatar", None)

    def set_emotion(self, name: str) -> dict[str, Any]:
        """Switch the avatar's baseline expression to ``name``."""
        avatar = self._avatar()
        if avatar is None:
            return {"ok": False, "error": "no avatar bound (set FACEVIEW_AVATAR=1)"}
        try:
            avatar.set_emotion(str(name))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "emotion": str(name)}

    def set_persona(self, name: str) -> dict[str, Any]:
        """Switch the avatar's appearance preset to ``name``."""
        avatar = self._avatar()
        if avatar is None:
            return {"ok": False, "error": "no avatar bound (set FACEVIEW_AVATAR=1)"}
        try:
            avatar.set_persona(str(name))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "persona": str(name)}

    def avatar_say(self, text: str, speed: float = 1.0) -> dict[str, Any]:
        """Drive the avatar to mouth ``text`` (visemes only — no TTS audio)."""
        if not text or not text.strip():
            return {"ok": False, "error": "empty text"}
        avatar = self._avatar()
        if avatar is None:
            return {"ok": False, "error": "no avatar bound (set FACEVIEW_AVATAR=1)"}
        try:
            u = avatar.say(str(text), speed=float(speed))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "duration": u.duration, "phonemes": len(u.timeline)}

    def list_personas(self) -> list[str]:
        from faceview.vision.personas import list_personas as _ls
        return _ls()

    def get_camera_state(self) -> dict[str, Any]:
        return asdict(self.camera_state)

    def list_events(self, n: int = 50) -> list[dict[str, Any]]:
        out = []
        for entry in list(self.events)[-n:]:
            out.append(
                {
                    "type": entry.type,
                    "ts": entry.ts,
                    "payload": _to_jsonable(entry.payload),
                }
            )
        return out

    def screenshot(self, name: str = "shot.png", *, encode_b64: bool = False) -> dict[str, Any]:
        """Save a window screenshot. Optionally inline as base64 (for MCP)."""
        if not name.endswith(".png"):
            name = f"{name}.png"

        # If we're already on the GUI thread, call directly. Otherwise marshal.
        if QThread.currentThread() is self.window.thread():
            try:
                path_obj = self.window.take_screenshot(name)
                path = Path(path_obj)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        else:
            self.bridge.reset()
            QMetaObject.invokeMethod(
                self.bridge,
                "take_screenshot",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, name),
            )
            if not self.bridge._evt.wait(timeout=5.0):
                return {"ok": False, "error": "screenshot timeout"}
            path = docs_image_dir() / name

        result: dict[str, Any] = {"ok": True, "path": str(path)}
        if encode_b64 and path.exists():
            result["png_b64"] = base64.b64encode(path.read_bytes()).decode("ascii")
        return result

    # ── bus handlers ────────────────────────────────────────────────

    def _on_presence(self, p) -> None:
        self.camera_state.presence = {
            "face_count": getattr(p, "face_count", 0),
        }

    def _on_identity(self, i) -> None:
        self.camera_state.identity = {
            "is_owner": getattr(i, "is_owner", False),
            "similarity": getattr(i, "similarity", 0.0),
            "label": getattr(i, "label", "unknown"),
        }

    def _on_emotion(self, e) -> None:
        self.camera_state.emotion = {
            "label": getattr(e, "label", "unknown"),
            "confidence": getattr(e, "confidence", 0.0),
        }

    def _on_mouth(self, m) -> None:
        self.camera_state.mouth = {
            "speaking": getattr(m, "speaking", False),
            "viseme": getattr(m, "viseme", None),
        }

    def _on_frame(self, _payload) -> None:
        self.camera_state.last_frame_ts = time.time()

    def _log_event(self, et: EventType, payload: Any) -> None:
        # Don't log frame-rate-y events into the public log.
        if et in (EventType.FRAME, EventType.AUDIO_CHUNK):
            return
        self.events.append(
            EventLogEntry(type=et.name, ts=time.time(), payload=payload)
        )


def _to_jsonable(payload: Any) -> Any:
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if hasattr(payload, "__dataclass_fields__"):
        return asdict(payload)
    if isinstance(payload, (list, tuple)):
        return [_to_jsonable(x) for x in payload]
    if isinstance(payload, dict):
        return {k: _to_jsonable(v) for k, v in payload.items()}
    return repr(payload)


# Module-level handle so adapters can find it.
_service: Optional[Service] = None


def init_service(window) -> Service:
    global _service
    _service = Service(window)
    return _service


def get_service() -> Service:
    if _service is None:
        raise RuntimeError("Service not initialised — call init_service(window) first")
    return _service
