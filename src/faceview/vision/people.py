"""Per-name face-embedding store.

InsightFace ArcFace gives us a 512-d L2-normalised embedding per face.
This module persists ``name → embedding`` mappings on disk so the GUI
can recognise multiple people across sessions, not just the binary
owner / stranger flag the original :mod:`faceview.vision.identity`
shipped with.

Layout:

    ~/.faceview/people/<slug>.npz
        ├── name     (str — original spelling, with caps and spaces)
        └── embedding (np.float32[512])

The legacy ``owner_data/owner.npy`` file is loaded as a synthetic
"owner" entry if present, so existing enrollments keep working with
zero migration.

Singleton: there's one process-wide :class:`PeopleStore` shared by the
identity recognizer (read side) and the LLM ``remember_person`` tool
(write side). The recogniser injects its embedding function so the
tool can convert a frame → embedding without re-loading InsightFace.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from faceview.config import settings
from faceview.core.logger import get_logger


log = get_logger("people")


# InsightFace buffalo_l with L2-normed embeddings: ~0.4 is the canonical
# same-person threshold. We use 0.42 to lean slightly cautious.
DEFAULT_THRESHOLD = 0.42


def _slug(name: str) -> str:
    """Filesystem-safe slug derived from a display name."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return s or "person"


class PeopleStore:
    """Disk-persisted multi-person face-template store (singleton)."""

    _instance: "PeopleStore | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "PeopleStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = PeopleStore()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._lock = threading.Lock()
        # slug → (display_name, normed_embedding)
        self._people: dict[str, tuple[str, np.ndarray]] = {}
        self._embed_fn: Optional[Callable[[np.ndarray],
                                          Optional[np.ndarray]]] = None
        self._load_from_disk()

    # ── persistence ─────────────────────────────────────────────────────

    def _people_dir(self) -> Path:
        return settings.data_dir / "people"

    def _load_from_disk(self) -> None:
        # 1) Legacy owner.npy — promote to a "owner" entry.
        try:
            legacy = settings.owner_dir / "owner.npy"
            if legacy.exists():
                emb = np.load(legacy)
                self._people["owner"] = ("owner", _l2(emb))
        except Exception as exc:  # noqa: BLE001
            log.warning("people.legacy_load_failed", error=str(exc))
        # 2) New-format .npz files under ~/.faceview/people/
        d = self._people_dir()
        if not d.exists():
            return
        for path in sorted(d.glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    name = str(data["name"].item()) if "name" in data \
                        else path.stem
                    emb = np.asarray(data["embedding"], dtype=np.float32)
                slug = path.stem
                self._people[slug] = (name, _l2(emb))
            except Exception as exc:  # noqa: BLE001
                log.warning("people.load_failed",
                            path=str(path), error=str(exc))

    def _save_one(self, slug: str, name: str, emb: np.ndarray) -> Path:
        d = self._people_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{slug}.npz"
        np.savez(path, name=np.array(name), embedding=emb.astype(np.float32))
        return path

    # ── injection ───────────────────────────────────────────────────────

    def set_embed_fn(
        self, fn: Callable[[np.ndarray], Optional[np.ndarray]],
    ) -> None:
        """Called by IdentityRecognizer once InsightFace is loaded."""
        self._embed_fn = fn

    def have_embed_fn(self) -> bool:
        return self._embed_fn is not None

    # ── reads ───────────────────────────────────────────────────────────

    def list_people(self) -> list[str]:
        with self._lock:
            return sorted(name for (name, _emb) in self._people.values())

    def count(self) -> int:
        with self._lock:
            return len(self._people)

    def match(self, embedding: np.ndarray) -> tuple[str, float, bool]:
        """Compare ``embedding`` to all stored people.

        Returns ``(display_name, similarity, is_known)``.
        When nothing matches above :attr:`threshold`, returns
        ``("stranger", best_sim, False)``.
        """
        emb = _l2(np.asarray(embedding))
        with self._lock:
            if not self._people:
                return "stranger", 0.0, False
            names = list(self._people.keys())
            mat = np.stack([self._people[s][1] for s in names])
            display = [self._people[s][0] for s in names]
        sims = mat @ emb
        i = int(np.argmax(sims))
        sim = float(sims[i])
        if sim < self.threshold:
            return "stranger", sim, False
        return display[i], sim, True

    # ── writes ──────────────────────────────────────────────────────────

    def remember(
        self,
        name: str,
        frame: np.ndarray,
    ) -> tuple[bool, str]:
        """Embed the supplied frame and save it under ``name``.

        Returns ``(ok, human_readable_message)``.
        """
        clean = (name or "").strip()
        if not clean:
            return False, "I need a non-empty name to remember someone."
        if self._embed_fn is None:
            return False, ("Identity recognizer isn't running yet — "
                           "I can't save a face without it.")
        try:
            emb = self._embed_fn(frame)
        except Exception as exc:  # noqa: BLE001
            log.warning("people.embed_error", error=str(exc))
            return False, f"Couldn't embed the current frame: {exc}"
        if emb is None:
            return False, ("I don't see a clear face in the camera right "
                           "now — can you turn toward the camera?")
        emb = _l2(np.asarray(emb))
        slug = _slug(clean)
        with self._lock:
            self._people[slug] = (clean, emb)
        try:
            self._save_one(slug, clean, emb)
        except Exception as exc:  # noqa: BLE001
            log.warning("people.save_failed", error=str(exc), slug=slug)
            return False, f"Saved in memory but disk write failed: {exc}"
        log.info("people.remembered", slug=slug, count=len(self._people))
        return True, f"Got it — I'll remember you as {clean}."

    def forget(self, name_or_slug: str) -> bool:
        """Remove a person by display name or slug. Returns True if removed."""
        target_slug = _slug(name_or_slug)
        with self._lock:
            if target_slug not in self._people:
                # Search by display name
                for slug, (display, _emb) in list(self._people.items()):
                    if display.lower() == name_or_slug.strip().lower():
                        target_slug = slug
                        break
            if target_slug not in self._people:
                return False
            del self._people[target_slug]
        path = self._people_dir() / f"{target_slug}.npz"
        try:
            if path.exists():
                path.unlink()
        except Exception as exc:  # noqa: BLE001
            log.warning("people.forget_failed", error=str(exc))
        return True


# ── helpers ──────────────────────────────────────────────────────────────


def _l2(emb: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(emb)) + 1e-9
    return (emb / n).astype(np.float32)
