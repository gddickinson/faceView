"""Open-vocabulary visibility check via OpenCLIP.

EfficientDet-Lite0 already gives us ~80 fixed COCO classes — but
"is the user wearing glasses?" / "is there a coffee mug?" / "is the
laptop screen showing code?" aren't in that vocabulary. CLIP closes
that gap: encode the image once, encode the free-form query, return
cosine similarity.

This module is **lazy-loaded** — the ~600 MB ViT-B/32 weights are
only fetched the first time the LLM calls ``check_visible``.

The output is a small structured tuple plus a friendly sentence the
tool returns to the model:

    (is_visible: bool, similarity: float, sentence: str)

Threshold is set conservatively at 0.22 (typical CLIP image-vs-text
similarity for "in vs out of frame" queries). Override via
``FACEVIEW_CLIP_THRESHOLD``.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

from faceview.core.errors import MissingDependency
from faceview.core.logger import get_logger


log = get_logger("clip")


_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "openai"
_DEFAULT_THRESHOLD = 0.22


def _threshold_from_env() -> float:
    raw = os.environ.get("FACEVIEW_CLIP_THRESHOLD")
    if raw:
        try:
            return max(0.10, min(0.40, float(raw)))
        except ValueError:
            pass
    return _DEFAULT_THRESHOLD


class ClipEngine:
    """Singleton OpenCLIP wrapper."""

    _instance: "ClipEngine | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "ClipEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ClipEngine()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import open_clip  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:
            raise MissingDependency("open_clip_torch", "vision-tools") from exc
        with self._lock:
            if self._model is not None:
                return
            log.info("clip.loading", model=_DEFAULT_MODEL,
                     pretrained=_DEFAULT_PRETRAINED)
            model, _, preprocess = open_clip.create_model_and_transforms(
                _DEFAULT_MODEL, pretrained=_DEFAULT_PRETRAINED,
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(_DEFAULT_MODEL)
            self._torch = torch
            log.info("clip.loaded")

    def similarity(self, frame: np.ndarray, query: str) -> float:
        """Cosine similarity between the image and the query (0..1ish)."""
        self._ensure_loaded()
        # OpenCLIP wants a PIL image in RGB.
        try:
            import cv2  # type: ignore
            from PIL import Image  # type: ignore
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not convert frame: {exc}") from exc
        torch = self._torch
        with torch.no_grad():
            img = self._preprocess(pil).unsqueeze(0)
            text = self._tokenizer([query])
            img_emb = self._model.encode_image(img)
            txt_emb = self._model.encode_text(text)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            sim = (img_emb @ txt_emb.T).item()
        return float(sim)


def check_visible(
    frame: np.ndarray,
    query: str,
    region: str = "full",
) -> str:
    """User-facing helper invoked by the LLM tool.

    Returns a one-sentence yes/no string with confidence — designed
    to be relayed straight to the user."""
    from faceview.llm.vision_tool import _crop_to_region  # lazy: cycle

    if not query or not query.strip():
        return "I need a query — what should I look for in the image?"
    if frame is None:
        return "No camera frame is available right now."
    cropped = _crop_to_region(frame, region)
    try:
        engine = ClipEngine.shared()
        sim = engine.similarity(cropped, query.strip())
    except MissingDependency:
        return ("Open-vocabulary check needs open_clip_torch — install "
                "with `pip install open_clip_torch` to enable.")
    except Exception as exc:  # noqa: BLE001
        log.warning("clip.error", error=str(exc))
        return f"Visibility check failed: {exc}"
    threshold = _threshold_from_env()
    yes = sim >= threshold
    log.info("clip.checked", query=query[:60],
             sim=round(sim, 3), yes=yes, region=region)
    if yes:
        return (f"Yes — '{query.strip()}' appears to be visible "
                f"(CLIP similarity {sim:.2f}).")
    return (f"No — I don't see '{query.strip()}' in the frame "
            f"(CLIP similarity {sim:.2f}, threshold {threshold:.2f}).")
