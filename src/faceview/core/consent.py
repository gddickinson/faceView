"""PR2 — per-tool consent dial.

Some LLM tools send pixels off-device (Anthropic vision content
blocks). Others stay on-machine (Ollama VLMs, MediaPipe). Users
might want to keep the cheap on-device tools available while
blocking the off-device ones until they explicitly approve.

This module is the single source of truth for "is this tool
allowed right now?". Engine dispatch consults it before invoking
the executor; blocked tools return a short refusal message that
the model relays to the user.

Decision is made from three signals, in priority order:

1. Per-tool override (set via :meth:`set_tool_decision`).
2. Per-destination override — covers Anthropic-vision in bulk
   without listing every tool.
3. Default policy: cheap on-device tools default-allow,
   off-device tools default-prompt unless globally trusted.

Persists choices to ``~/.faceview/tool_consent.json`` so a user's
explicit "always trust this" survives restart.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Literal, Optional

from faceview.config import settings
from faceview.core.logger import get_logger


log = get_logger("consent")


Decision = Literal["allow", "block", "prompt"]


# Tools that stay strictly on this machine.
_LOCAL_TOOLS = {
    "remember_person", "read_text", "track_object", "check_visible",
    "describe_color", "describe_pose", "face_attributes",
    "scan_qr", "estimate_depth", "gaze_target", "segment_object",
    "describe_room_layout", "forget_memory",
}

# Tools that may send raw pixels off-device when the engine is
# Anthropic. Routed through Ollama they stay local but the same
# tool name applies in both — so we err on the side of "sends".
_REMOTE_RISK_TOOLS = {
    "look_at_camera", "look_at_screen",
}


class ConsentStore:
    """Singleton — one dial per process."""

    _instance: "ConsentStore | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "ConsentStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ConsentStore()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tool_decisions: dict[str, Decision] = {}
        self._trust_remote: bool = False  # default-prompt
        self._load()

    # ── persistence ─────────────────────────────────────────

    def _path(self) -> Path:
        return settings.data_dir / "tool_consent.json"

    def _load(self) -> None:
        p = self._path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            with self._lock:
                self._tool_decisions = {
                    str(k): v for k, v in (
                        data.get("tools") or {}
                    ).items()
                    if v in ("allow", "block", "prompt")
                }
                self._trust_remote = bool(data.get("trust_remote"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            log.warning("consent.load_failed", error=str(exc))

    def _persist(self) -> None:
        try:
            p = self._path()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            with self._lock:
                payload = {
                    "tools": dict(self._tool_decisions),
                    "trust_remote": self._trust_remote,
                    "saved_at": time.time(),
                }
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(p)
        except OSError as exc:
            log.warning("consent.save_failed", error=str(exc))

    # ── public API ──────────────────────────────────────────

    def set_tool_decision(self, name: str, decision: Decision) -> None:
        with self._lock:
            self._tool_decisions[name] = decision
        self._persist()

    def get_tool_decision(self, name: str) -> Optional[Decision]:
        with self._lock:
            return self._tool_decisions.get(name)

    def set_trust_remote(self, on: bool) -> None:
        with self._lock:
            self._trust_remote = bool(on)
        self._persist()

    def trust_remote(self) -> bool:
        with self._lock:
            return self._trust_remote

    def is_allowed(self, name: str, *, engine: str = "") -> bool:
        """Top-level decision used by engine dispatch.

        ``engine`` lets us be smart about look_at_* — Ollama routes
        stay local, Anthropic routes leave the host. When engine
        isn't supplied we treat any remote-risk tool as potentially
        sending."""
        explicit = self.get_tool_decision(name)
        if explicit == "allow":
            return True
        if explicit == "block":
            return False
        # No explicit decision → policy defaults.
        if name in _LOCAL_TOOLS:
            return True
        if name in _REMOTE_RISK_TOOLS:
            if engine == "ollama":
                return True  # stays on-machine
            return self.trust_remote()
        # Unknown tool — allow by default; the engine catches
        # bogus names elsewhere.
        return True

    def refuse_message(self, name: str) -> str:
        """Short text the engine returns when a tool is blocked, so
        the LLM can relay the reason to the user."""
        return (
            f"I'd like to call `{name}` but the user has disabled "
            "that tool. Ask them to re-enable it via "
            "Tools → Tool consent if you need it."
        )
