"""Exception hierarchy.

Use :class:`MissingDependency` to surface a clean install hint when an optional
ML extra (mediapipe, insightface, faster-whisper, etc.) is not present.
"""

from __future__ import annotations


class FaceViewError(Exception):
    """Base class for all faceView-specific errors."""


class MissingDependency(FaceViewError):
    """A required optional dependency is not installed."""

    def __init__(self, package: str, extra: str, *, hint: str | None = None) -> None:
        msg = (
            f"Optional dependency '{package}' is not installed. "
            f"Install with: pip install -e \".[{extra}]\""
        )
        if hint:
            msg += f"\nHint: {hint}"
        super().__init__(msg)
        self.package = package
        self.extra = extra


class CameraError(FaceViewError):
    """Webcam open / read failure."""


class AudioError(FaceViewError):
    """Microphone open / read failure."""


class LlmError(FaceViewError):
    """Anthropic API or demo-mode failure."""


class ServiceError(FaceViewError):
    """Service-layer operation failed."""
