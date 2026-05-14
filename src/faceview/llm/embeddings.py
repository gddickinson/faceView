"""Sentence-embedding service for retrieval-augmented memory.

Wraps ``sentence-transformers/all-MiniLM-L6-v2`` — 384-d L2-normed
vectors, ~80 MB model, ~5 ms/embedding on Apple Silicon CPU. Lazy-
loaded singleton so the import (and the model download on first run)
is paid only once per process, and only when retrieval is actually
used.

The dependency is **optional**. If ``sentence-transformers`` isn't
installed, :meth:`embed` returns ``None`` and the cognition layer
falls back to its keyword-based recall scoring. Install with::

    pip install sentence-transformers

No hard failure mode.
"""

from __future__ import annotations

import math
import threading
from typing import Optional

from faceview.core.logger import get_logger


log = get_logger("embeddings")


_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class EmbeddingService:
    """Singleton wrapper around a SentenceTransformer."""

    _instance: "EmbeddingService | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "EmbeddingService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = EmbeddingService()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        # Cached availability: probe once and remember.
        self._available: Optional[bool] = None

    # ── public API ────────────────────────────────────────────────

    def available(self) -> bool:
        """Whether the embedding model can be loaded.

        Falsy means ``sentence-transformers`` isn't installed; callers
        should fall back to non-embedding logic."""
        if self._available is None:
            self._available = _sentence_transformers_available()
            if not self._available:
                log.info("embeddings.unavailable",
                         hint="pip install sentence-transformers")
        return bool(self._available)

    def embed(self, text: str) -> Optional[list[float]]:
        """Return an L2-normed embedding for ``text``.

        ``None`` on missing dep, empty input, or load failure — the
        caller's job to treat that as "no embedding available"."""
        if not text or not text.strip():
            return None
        if not self.available():
            return None
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            log.warning("embeddings.load_failed", error=str(exc))
            self._available = False
            return None
        try:
            vec = self._model.encode(
                [text.strip()], normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            log.warning("embeddings.encode_failed", error=str(exc))
            return None
        return [float(x) for x in vec]

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        if not texts:
            return []
        if not self.available():
            return [None] * len(texts)
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            log.warning("embeddings.load_failed", error=str(exc))
            self._available = False
            return [None] * len(texts)
        try:
            vecs = self._model.encode(
                [(t or "").strip() for t in texts],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("embeddings.encode_failed", error=str(exc))
            return [None] * len(texts)
        return [[float(x) for x in v] for v in vecs]

    # ── internals ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore
        with self._lock:
            if self._model is not None:
                return
            log.info("embeddings.loading", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            log.info("embeddings.loaded")


# ── helpers ───────────────────────────────────────────────────────


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity between two same-length lists. Returns 0.0
    on length mismatch / missing inputs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
