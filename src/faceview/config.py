"""Runtime configuration: env vars, paths, feature flags.

Reading config:

    from faceview.config import settings
    if settings.anthropic_api_key:
        ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from faceview.utils.paths import data_dir, owner_dir


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    # API keys
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("FACEVIEW_MODEL", "claude-sonnet-4-6")
    )

    # Server
    api_host: str = field(default_factory=lambda: os.environ.get("FACEVIEW_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("FACEVIEW_API_PORT", 8765))
    api_enabled: bool = field(default_factory=lambda: _env_bool("FACEVIEW_API", True))
    mcp_enabled: bool = field(default_factory=lambda: _env_bool("FACEVIEW_MCP", False))

    # Audio
    sample_rate: int = 16_000
    audio_chunk_ms: int = 30

    # Vision
    camera_index: int = field(default_factory=lambda: _env_int("FACEVIEW_CAMERA", 0))
    target_fps: int = 30

    # Headless mode toggles
    headless: bool = field(default_factory=lambda: _env_bool("FACEVIEW_HEADLESS", False))
    auto_start_camera: bool = field(
        default_factory=lambda: _env_bool("FACEVIEW_AUTOCAM", True)
    )
    auto_start_audio: bool = field(
        default_factory=lambda: _env_bool("FACEVIEW_AUTOMIC", False)
    )

    # Paths
    data_dir: Path = field(default_factory=data_dir)
    owner_dir: Path = field(default_factory=owner_dir)

    @property
    def has_claude_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def api_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"


settings = Settings()
