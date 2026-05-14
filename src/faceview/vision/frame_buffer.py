"""P7 — rolling video frame buffer.

A singleton that subscribes to ``EventType.FRAME`` and keeps the
last N seconds of frames in memory. Future on-demand tools (action
recognition, "what just happened?" Q&A, motion playback) read short
clips from here without re-grabbing the camera or running their
own capture.

Memory use: a 1280×720 BGR frame is ~2.7 MB; at 30 fps × 5 s ×
2.7 MB ≈ 400 MB. Default is therefore 5 seconds at the camera's
actual frame rate, but bounded by a max-frame count + a max-byte
budget so a high-resolution / high-fps webcam doesn't drown the
process.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType
from faceview.core.logger import get_logger


log = get_logger("frame_buffer")


# Defaults — overridable per-instance.
_DEFAULT_SECONDS = 5.0
_DEFAULT_MAX_FRAMES = 240   # ~8 s at 30 fps, ~16 s at 15 fps
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024  # 256 MB hard ceiling


class FrameBuffer:
    """Singleton rolling buffer of recent FRAME events."""

    _instance: "FrameBuffer | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "FrameBuffer":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = FrameBuffer()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(
        self,
        seconds: float = _DEFAULT_SECONDS,
        max_frames: int = _DEFAULT_MAX_FRAMES,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self.seconds = float(seconds)
        self.max_frames = int(max_frames)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        # (timestamp, frame_ndarray) pairs.
        self._frames: deque[tuple[float, object]] = deque(
            maxlen=self.max_frames,
        )
        self._size_bytes = 0
        get_bus().subscribe(EventType.FRAME, self._on_frame)

    # ── public API ──────────────────────────────────────────

    def push(self, frame, ts: Optional[float] = None) -> None:
        """Manual push — used by tests and future capture sources."""
        self._on_frame(frame, ts=ts)

    def clip_last(self, seconds: float) -> list:
        """Return a list of frames captured in the last ``seconds``.
        Cheap — references into the buffer, no copy."""
        with self._lock:
            if not self._frames:
                return []
            cutoff = time.time() - max(0.0, float(seconds))
            return [
                f for t, f in self._frames if t >= cutoff
            ]

    def clip_all(self) -> list:
        with self._lock:
            return [f for _t, f in self._frames]

    def count(self) -> int:
        with self._lock:
            return len(self._frames)

    def byte_size(self) -> int:
        with self._lock:
            return self._size_bytes

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._size_bytes = 0

    # ── internals ───────────────────────────────────────────

    def _on_frame(self, frame, ts: Optional[float] = None) -> None:
        if frame is None:
            return
        size = 0
        try:
            size = int(getattr(frame, "nbytes", 0) or 0)
        except Exception:  # noqa: BLE001
            size = 0
        now = ts if ts is not None else time.time()
        with self._lock:
            # Pop oldest until we're under the time / byte budget.
            cutoff = now - self.seconds
            while self._frames and self._frames[0][0] < cutoff:
                _t, old = self._frames.popleft()
                self._size_bytes -= int(
                    getattr(old, "nbytes", 0) or 0
                )
            while (self._frames
                   and self._size_bytes + size > self.max_bytes):
                _t, old = self._frames.popleft()
                self._size_bytes -= int(
                    getattr(old, "nbytes", 0) or 0
                )
            self._frames.append((now, frame))
            self._size_bytes += size
            if self._size_bytes < 0:
                self._size_bytes = 0
