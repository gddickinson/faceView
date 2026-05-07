"""openFACS UDP bridge — emit our AU stream to an external Unreal renderer.

[phuselab/openFACS](https://github.com/phuselab/openFACS) is an MIT-
licensed Unreal Engine 4 facial animation system that listens for
JSON action-unit messages on UDP localhost:5000. This module wraps
that protocol so faceView can drive an external openFACS-rendered
avatar in parallel with its own renderer.

Usage::

    from faceview.vision.openfacs_bridge import OpenFACSBridge
    bridge = OpenFACSBridge()
    bridge.send({"AU12": 1.0, "AU26": 0.5})

The bridge subscribes to our event bus when ``attach_to_bus()`` is
called, so an avatar's FACS pipeline drives the remote renderer
automatically.

No new runtime dependency — pure stdlib socket + json.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Optional


# openFACS' Unreal blueprint listens on this port by default.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


# openFACS uses AU1..AU45 + a "speed" field. Our 12 AUs map directly
# (the openFACS set is a superset of FACS 12).
_AU_PASSTHROUGH = (
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU9",
    "AU12", "AU15", "AU20", "AU22", "AU25", "AU26",
)


@dataclass
class OpenFACSBridge:
    """Send AU values to a running openFACS Unreal instance."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    speed: float = 1.0
    _sock: Optional[socket.socket] = None

    def __post_init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, au_values: dict[str, float]) -> None:
        """Encode AU dict as JSON and emit one UDP packet."""
        payload: dict[str, float] = {"speed": float(self.speed)}
        for au in _AU_PASSTHROUGH:
            payload[au] = float(au_values.get(au, 0.0))
        msg = json.dumps(payload).encode("ascii")
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.sendto(msg, (self.host, self.port))

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    # ── Avatar integration ──────────────────────────────────────

    def attach_to_avatar(self, avatar) -> None:
        """Wire the bridge to a TalkingAvatar so every tick streams.

        Wraps ``avatar.tick`` so each rendered frame also emits an AU
        packet. Reversible via ``detach_from_avatar``.
        """
        from faceview.vision.anatomy import face_params_to_au_values
        original_tick = avatar.tick

        def patched_tick(t=None):
            params = original_tick(t)
            self.send(face_params_to_au_values(params))
            return params

        avatar._original_tick = original_tick
        avatar.tick = patched_tick

    def detach_from_avatar(self, avatar) -> None:
        if hasattr(avatar, "_original_tick"):
            avatar.tick = avatar._original_tick
            delattr(avatar, "_original_tick")
