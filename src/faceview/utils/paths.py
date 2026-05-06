"""Path helpers: project-local data + owner enrollment dirs."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Repository root — three parents up from this file."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Project-local data dir; honors ``FACEVIEW_DATA_DIR`` env var."""
    env = os.environ.get("FACEVIEW_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / ".faceview"


def owner_dir() -> Path:
    """Where face-enrollment embeddings live."""
    return project_root() / "owner_data"


def docs_image_dir() -> Path:
    """Where README screenshots are written."""
    d = project_root() / "docs" / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d
