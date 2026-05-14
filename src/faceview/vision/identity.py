"""Multi-person face identification via InsightFace ArcFace.

Per frame: embed the largest face, compare against every name in the
process-wide :class:`PeopleStore` (see ``vision/people.py``), and emit
:data:`EventType.IDENTITY` with the best-matching name + similarity.

The legacy owner-only workflow still works: ``tools/enroll_owner.py``
writes ``owner_data/owner.npy`` and the :class:`PeopleStore` picks it
up as a synthetic ``"owner"`` entry. Save-owner-template is preserved
for backwards compatibility.

The LLM-driven enrollment flow:

1. Stranger appears → PerceptionStore reports stranger-since-N-seconds
   in the system prompt.
2. LLM (Claude / Ollama with tool support) asks for the name.
3. User replies; LLM calls ``remember_person(name="…")`` tool.
4. Tool grabs the latest frame, calls ``IdentityRecognizer.embed`` via
   the embed_fn :class:`PeopleStore` was wired with at start-up, and
   persists the new template.

Next frame on, the same person is recognised by name automatically.
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
from faceview.core.events import (
    EventType, IdentitiesMulti, Identity, IdentityHit,
)
from faceview.core.logger import get_logger
from faceview.vision.people import PeopleStore


log = get_logger("identity")


OWNER_TEMPLATE_FILE = "owner.npy"


class IdentityRecognizer:
    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold
        self._app = None
        self._lock = threading.Lock()
        self._last_emit = 0.0
        self._people: Optional[PeopleStore] = None

    def start(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as exc:
            raise MissingDependency("insightface", "identity") from exc

        self._app = FaceAnalysis(name="buffalo_l", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=(640, 640))

        self._people = PeopleStore.shared()
        # Let the LLM ``remember_person`` tool reach back into InsightFace
        # without owning a second model.
        self._people.set_embed_fn(self.embed)
        # Same idea for the face_attributes tool — share the loaded
        # FaceAnalysis handle so it can read age + gender.
        try:
            from faceview.vision.face_attr import register_app
            register_app(self._app)
        except Exception:  # noqa: BLE001
            pass

        get_bus().subscribe(EventType.FRAME, self._on_frame)
        log.info("identity.started", people=self._people.count())

    # ── owner-template back-compat ─────────────────────────────────

    def load_owner_template(self) -> bool:
        """Legacy hook — present so old call sites keep working. The
        owner template is automatically picked up by PeopleStore on
        boot, so this no longer needs to do anything beyond reporting
        whether a template file exists."""
        path = settings.owner_dir / OWNER_TEMPLATE_FILE
        return path.exists()

    def save_owner_template(self, template: np.ndarray) -> Path:
        settings.owner_dir.mkdir(parents=True, exist_ok=True)
        path = settings.owner_dir / OWNER_TEMPLATE_FILE
        np.save(path, template)
        # Drop the cached store so the new template is picked up on
        # next match — cheapest correct way to refresh without a
        # dedicated reload() API.
        PeopleStore.reset_for_tests()
        self._people = PeopleStore.shared()
        self._people.set_embed_fn(self.embed)
        return path

    # ── inference ──────────────────────────────────────────────────

    def embed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self._app is None:
            return None
        faces = self._app.get(frame)
        if not faces:
            return None
        # Largest face (assume the user).
        faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return faces[0].normed_embedding

    def _on_frame(self, frame) -> None:
        if frame is None or self._app is None or self._people is None:
            return
        now = time.time()
        if now - self._last_emit < 0.5:  # 2 Hz is plenty
            return
        self._last_emit = now
        try:
            faces = self._app.get(frame) or []
        except Exception as exc:  # noqa: BLE001
            log.warning("identity.error", error=str(exc))
            return
        if not faces:
            return
        # Largest-first ordering so the back-compat IDENTITY event
        # keeps reporting "the most prominent face".
        faces.sort(
            key=lambda f: -(
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
        )
        hits: list[IdentityHit] = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            name, sim, is_known = self._people.match(emb)
            is_owner = is_known and name.lower() == "owner"
            bbox = (
                int(f.bbox[0]), int(f.bbox[1]),
                int(f.bbox[2] - f.bbox[0]), int(f.bbox[3] - f.bbox[1]),
            )
            hits.append(IdentityHit(
                is_owner=is_owner, similarity=sim, label=name, bbox=bbox,
            ))
        if not hits:
            return
        # P6 — full list of recognised faces.
        get_bus().publish(
            EventType.IDENTITIES_MULTI,
            IdentitiesMulti(hits=list(hits)),
        )
        # Back-compat IDENTITY (single, largest face).
        top = hits[0]
        get_bus().publish(
            EventType.IDENTITY,
            Identity(is_owner=top.is_owner,
                     similarity=top.similarity,
                     label=top.label),
        )
