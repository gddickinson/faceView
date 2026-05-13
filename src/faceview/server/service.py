"""Shared service layer used by both the HTTP API and the MCP server.

Thread-safe: Qt-touching ops route through ``_GuiBridge`` on the GUI
thread; everything else just reads cached state.
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

    @Slot(str, bool)
    def set_lifecycle(self, name: str, on: bool) -> None:
        slots = Service._LIFECYCLE_SLOTS.get(name)
        if not slots:
            return
        fn = getattr(self._window, slots[1], None)
        if callable(fn):
            try:
                fn(bool(on))
            except Exception:  # noqa: BLE001
                pass

    @Slot()
    def restart_test_mode(self) -> None:
        fn = getattr(self._window, "restart_test_mode", None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    @Slot()
    def close_window(self) -> None:
        try:
            self._window.close()
        except Exception:  # noqa: BLE001
            pass

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
        self.chat_log: deque = deque(maxlen=200)
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
        b.subscribe(EventType.CHAT_LOG, self._on_chat_log)
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

    # ── effects ─────────────────────────────────────────────────

    def trigger_effect(
        self, name: str, *,
        intensity: float = 1.0,
        duration: float | None = None,
    ) -> dict[str, Any]:
        """Trigger an avatar effect by name. Returns whether known."""
        from faceview.vision.effects_runtime import get_runtime
        ok = get_runtime().trigger(name, intensity=intensity, duration=duration)
        return {"ok": ok, "name": name}

    def list_effects(self) -> list[dict[str, Any]]:
        from faceview.vision.effects_runtime import get_runtime
        return get_runtime().list_specs()

    def list_active_effects(self) -> list[dict[str, Any]]:
        from faceview.vision.effects_runtime import get_runtime
        return get_runtime().list_active()

    def stop_effect(self, name: str) -> dict[str, Any]:
        from faceview.vision.effects_runtime import get_runtime
        n = get_runtime().stop(name)
        return {"ok": True, "stopped": n}

    def stop_all_effects(self) -> dict[str, Any]:
        from faceview.vision.effects_runtime import get_runtime
        n = get_runtime().stop_all()
        return {"ok": True, "stopped": n}

    def set_slider(self, key: str, value) -> dict[str, Any]:
        from faceview.vision.effects_runtime import get_runtime
        ok = get_runtime().set_slider(key, value)
        return {"ok": ok, "key": key}

    def get_sliders(self) -> dict[str, Any]:
        from faceview.vision.effects_runtime import get_runtime
        return get_runtime().get_sliders()

    def get_camera_state(self) -> dict[str, Any]:
        return asdict(self.camera_state)

    # ── monitoring ──────────────────────────────────────────────────

    # ── control surface (write) ─────────────────────────────────────

    def set_engine(self, engine: str, *, model: Optional[str] = None) -> dict[str, Any]:
        """Live-swap the main ClaudeClient engine."""
        client = getattr(self.window, "llm_client", None)
        if client is None or not hasattr(client, "select_engine"):
            return {"ok": False, "error": "no llm_client bound"}
        try:
            actual = client.select_engine(engine, model=model)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "engine": actual, "model": model}

    def set_test_engine(self, engine: str, *, model: Optional[str] = None) -> dict[str, Any]:
        """Configure the test-mode bot engine (restarts test mode if running)."""
        import os
        engine = (engine or "canned").lower()
        os.environ["FACEVIEW_TEST_ENGINE"] = engine
        if model:
            os.environ["FACEVIEW_TEST_MODEL"] = model
        else:
            os.environ.pop("FACEVIEW_TEST_MODEL", None)
        restarted = False
        try:
            if (hasattr(self.window, "test_mode_running")
                    and self.window.test_mode_running()):
                QMetaObject.invokeMethod(
                    self.bridge, "restart_test_mode",
                    Qt.ConnectionType.QueuedConnection,
                )
                restarted = True
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "engine": engine, "model": model, "restarted": restarted}

    _LIFECYCLE_SLOTS = {
        "camera":    ("camera_running",    "set_camera_enabled"),
        "mic":       ("audio_running",     "set_audio_enabled"),
        "tts":       ("tts_running",       "set_tts_enabled"),
        "avatar":    ("avatar_running",    "set_avatar_enabled"),
        "test_mode": ("test_mode_running", "set_test_mode_enabled"),
        "mirror":    ("mirror_running",    "set_mirror_mode_enabled"),
    }

    def set_lifecycle(self, name: str, on: bool) -> dict[str, Any]:
        slots = self._LIFECYCLE_SLOTS.get(name)
        if slots is None:
            return {"ok": False, "error": f"unknown worker {name!r}"}
        try:
            QMetaObject.invokeMethod(
                self.bridge, "set_lifecycle",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, name),
                Q_ARG(bool, bool(on)),
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "name": name, "on": bool(on)}

    def shutdown(self) -> dict[str, Any]:
        try:
            QMetaObject.invokeMethod(
                self.bridge, "close_window", Qt.ConnectionType.QueuedConnection,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "queued": True}

    # ── monitoring (read) ───────────────────────────────────────────

    def list_chat_log(self, n: int = 50) -> list[dict[str, Any]]:
        out = []
        for entry in list(self.chat_log)[-n:]:
            out.append({
                "who":   getattr(entry, "who", ""),
                "text":  getattr(entry, "text", ""),
                "color": getattr(entry, "color", "#666"),
                "ts":    getattr(entry, "ts", 0.0),
            })
        return out

    def monitor_snapshot(self, *, chat_n: int = 20, events_n: int = 30) -> dict[str, Any]:
        """One-shot view for the /monitor endpoint — engines, workers,
        recent chat, recent events. All fields are best-effort: missing
        attributes degrade to None rather than raising."""
        import os
        w = self.window

        # LLM client + engine
        client = getattr(w, "llm_client", None)
        if client is not None and hasattr(client, "current_engine"):
            engine = client.current_engine()
        else:
            engine = None
        try:
            from faceview.config import settings
            anthropic_model = settings.anthropic_model
            has_key = settings.has_claude_key
        except Exception:  # noqa: BLE001
            anthropic_model, has_key = None, False
        ollama_model = os.environ.get("FACEVIEW_OLLAMA_MODEL")
        if engine == "anthropic":
            active_model: Optional[str] = anthropic_model
        elif engine == "ollama":
            active_model = ollama_model or "(default)"
        else:
            active_model = None

        # Worker states
        def _safe_bool(attr: str) -> Optional[bool]:
            fn = getattr(w, attr, None)
            try:
                return bool(fn()) if callable(fn) else None
            except Exception:  # noqa: BLE001
                return None

        workers = {
            "camera":    _safe_bool("camera_running"),
            "mic":       _safe_bool("audio_running"),
            "tts":       _safe_bool("tts_running"),
            "avatar":    _safe_bool("avatar_running"),
            "test_mode": _safe_bool("test_mode_running"),
            "mirror":    _safe_bool("mirror_running"),
        }

        # Test-mode detail (if running)
        test: dict[str, Any] = {
            "engine": os.environ.get("FACEVIEW_TEST_ENGINE") or "canned",
            "model":  os.environ.get("FACEVIEW_TEST_MODEL"),
            "mode":   None,
        }
        orch = getattr(w, "_test_orchestrator", None)
        if orch is not None and hasattr(orch, "mode"):
            test["mode"] = orch.mode

        persona = None
        if hasattr(w, "current_persona"):
            try:
                persona = w.current_persona()
            except Exception:  # noqa: BLE001
                persona = None

        return {
            "ok": True,
            "ts": time.time(),
            "engine": engine,
            "model": active_model,
            "anthropic_key": has_key,
            "anthropic_model": anthropic_model,
            "ollama_model": ollama_model,
            "persona": persona,
            "workers": workers,
            "test": test,
            "camera_state": asdict(self.camera_state),
            "chat": self.list_chat_log(n=chat_n),
            "events": self.list_events(n=events_n),
        }

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

    def _on_chat_log(self, entry) -> None:
        self.chat_log.append(entry)

    def _log_event(self, et: EventType, payload: Any) -> None:
        # Drop frame-rate-y events so /events stays signal not noise.
        if et in (EventType.FRAME, EventType.AVATAR_FRAME, EventType.AUDIO_CHUNK):
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
    # If the avatar worker started before the API server (the usual order
    # in app.py), retroactively bind it so /avatar/say etc. work.
    worker = getattr(window, "_avatar_worker", None)
    if worker is not None:
        _service.bind_camera_worker(worker)
    return _service


def get_service() -> Service:
    if _service is None:
        raise RuntimeError("Service not initialised — call init_service(window) first")
    return _service
