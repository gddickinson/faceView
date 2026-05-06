"""Owner-vs-stranger face identification via InsightFace ArcFace.

Enrollment workflow (run via ``tools/enroll_owner.py``): capture N frames,
extract ArcFace embeddings, average → owner template stored at
``owner_data/owner.npy``. At runtime, embed the largest detected face per
frame and report cosine similarity. Above ``threshold`` ⇒ owner.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from faceview.config import settings
from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, Identity
from faceview.core.logger import get_logger


log = get_logger("identity")


OWNER_TEMPLATE_FILE = "owner.npy"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class IdentityRecognizer:
    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold
        self._app = None
        self._owner_template: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._last_emit = 0.0

    def start(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as exc:
            raise MissingDependency("insightface", "identity") from exc

        self._app = FaceAnalysis(name="buffalo_l", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=(640, 640))

        self.load_owner_template()
        get_bus().subscribe(EventType.FRAME, self._on_frame)
        log.info("identity.started", has_owner=self._owner_template is not None)

    def load_owner_template(self) -> bool:
        path = settings.owner_dir / OWNER_TEMPLATE_FILE
        if path.exists():
            self._owner_template = np.load(path)
            return True
        return False

    def save_owner_template(self, template: np.ndarray) -> Path:
        settings.owner_dir.mkdir(parents=True, exist_ok=True)
        path = settings.owner_dir / OWNER_TEMPLATE_FILE
        np.save(path, template)
        self._owner_template = template
        return path

    def embed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self._app is None:
            return None
        faces = self._app.get(frame)
        if not faces:
            return None
        # Largest face (assume the user).
        faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return faces[0].normed_embedding

    # ── frame handler ──────────────────────────────────────────────

    def _on_frame(self, frame) -> None:
        if frame is None or self._app is None:
            return
        now = time.time()
        if now - self._last_emit < 0.5:  # 2 Hz is plenty
            return
        self._last_emit = now
        try:
            emb = self.embed(frame)
        except Exception as exc:  # noqa: BLE001
            log.warning("identity.error", error=str(exc))
            return
        if emb is None:
            return
        if self._owner_template is None:
            sim = 0.0
            label = "unknown"
            is_owner = False
        else:
            sim = _cosine(emb, self._owner_template)
            is_owner = sim >= self.threshold
            label = "owner" if is_owner else "stranger"
        get_bus().publish(
            EventType.IDENTITY,
            Identity(is_owner=is_owner, similarity=sim, label=label),
        )
