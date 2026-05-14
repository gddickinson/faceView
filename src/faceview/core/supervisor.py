"""Worker-thread health supervisor (R7).

The faceView GUI runs a dozen background workers (camera capture,
each MediaPipe analyser, the room-map worker, ambient VLM captioner,
audio + STT, etc.). A single segfault in any one of them used to
leave the GUI looking healthy while a signal silently stopped
flowing. This module gives every worker a place to register so we
can detect and surface those silent deaths.

API is intentionally minimal:

* `register(name, worker, *, restart=None)` — call after `worker.start()`
* `unregister(name)` — call from `worker.stop()`
* `WorkerSupervisor.shared().status()` — snapshot of current health

Every ~5 s the supervisor walks its registry. For each entry whose
thread is no longer alive:

1. Log a warning (`supervisor.worker_died`).
2. Publish a STATUS event so the GUI status bar surfaces it.
3. If a `restart` callable was provided, call it (up to 3 attempts
   per session — beyond that we leave the worker dead and assume
   the cause is structural, not transient).

The supervisor is intentionally **opt-in** for restart. Most
workers (CameraWorker, MouthAnalyzer, …) are restarted by their
owning controller when the user toggles them back on; only workers
that should auto-recover from a transient crash should pass a
`restart` callback.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, StatusEvent
from faceview.core.logger import get_logger


log = get_logger("supervisor")


# How often to walk the registry checking for dead threads.
_CHECK_INTERVAL_S = 5.0

# Cap how many auto-restart attempts we make per (process, worker)
# before giving up.
_MAX_RESTARTS = 3


@dataclass
class _WorkerEntry:
    name: str
    thread_provider: Callable[[], Optional[threading.Thread]]
    restart: Optional[Callable[[], None]] = None
    restart_attempts: int = 0
    died_at: float = 0.0
    history: list[float] = field(default_factory=list)


class WorkerSupervisor:
    """Singleton — one per process."""

    _instance: "WorkerSupervisor | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "WorkerSupervisor":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = WorkerSupervisor()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            inst = cls._instance
            cls._instance = None
        if inst is not None:
            inst._stop.set()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[str, _WorkerEntry] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start()

    # ── public API ───────────────────────────────────────────

    def register(
        self,
        name: str,
        worker_or_thread,
        *,
        restart: Optional[Callable[[], None]] = None,
    ) -> None:
        """Track a worker. ``worker_or_thread`` is either a
        ``threading.Thread`` or any object exposing a ``_thread``
        attribute (the convention most of our workers follow)."""
        provider = _thread_provider_for(worker_or_thread)
        with self._lock:
            self._workers[name] = _WorkerEntry(
                name=name, thread_provider=provider, restart=restart,
            )
        log.info("supervisor.registered", name=name)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._workers.pop(name, None)

    def status(self) -> dict:
        """Snapshot — used by HTTP /health and the GUI dashboards."""
        out: dict[str, dict] = {}
        with self._lock:
            for name, entry in self._workers.items():
                thread = entry.thread_provider()
                out[name] = {
                    "alive": bool(thread and thread.is_alive()),
                    "thread_name": thread.name if thread else None,
                    "restart_attempts": entry.restart_attempts,
                    "history": list(entry.history),
                }
        return out

    # ── internals ────────────────────────────────────────────

    def _start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="worker-supervisor", daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                log.warning("supervisor.tick_error", error=str(exc))
            for _ in range(int(_CHECK_INTERVAL_S * 2)):
                if self._stop.is_set():
                    return
                time.sleep(0.5)

    def _tick(self) -> None:
        with self._lock:
            entries = list(self._workers.values())
        for entry in entries:
            thread = entry.thread_provider()
            alive = bool(thread and thread.is_alive())
            if alive:
                # Reset death timestamp on a recovery so the next
                # death starts a fresh attempt counter.
                continue
            now = time.time()
            # Brand-new entry where the thread just hasn't started
            # yet (e.g. registered before .start()) — give it a free
            # pass for one cycle.
            if entry.died_at == 0.0:
                entry.died_at = now
                continue
            log.warning("supervisor.worker_died", name=entry.name,
                        attempts=entry.restart_attempts)
            entry.history.append(now)
            try:
                get_bus().publish(
                    EventType.STATUS,
                    StatusEvent(
                        source="supervisor",
                        message=f"worker '{entry.name}' died",
                        level="warning",
                    ),
                )
            except Exception:  # noqa: BLE001
                pass
            if (entry.restart is not None
                    and entry.restart_attempts < _MAX_RESTARTS):
                entry.restart_attempts += 1
                try:
                    entry.restart()
                    log.info("supervisor.restarted", name=entry.name,
                             attempts=entry.restart_attempts)
                except Exception as exc:  # noqa: BLE001
                    log.warning("supervisor.restart_failed",
                                name=entry.name, error=str(exc))
            # Reset died_at so we don't re-fire the death event
            # every tick — only when the thread newly dies after a
            # successful restart attempt.
            entry.died_at = 0.0


# ── helpers ───────────────────────────────────────────────────────


def _thread_provider_for(obj) -> Callable[[], Optional[threading.Thread]]:
    """Return a callable that fetches the worker's current thread.

    Most faceView workers store the thread on ``self._thread``;
    some expose ``.thread()``; if neither, return ``None`` lambda."""
    if isinstance(obj, threading.Thread):
        return lambda obj=obj: obj
    if hasattr(obj, "_thread"):
        return lambda obj=obj: getattr(obj, "_thread", None)
    if hasattr(obj, "thread"):
        return lambda obj=obj: obj.thread()
    return lambda: None
