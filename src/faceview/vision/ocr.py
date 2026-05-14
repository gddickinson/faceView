"""On-demand OCR via EasyOCR.

This module is loaded lazily — neither EasyOCR nor torch is imported
until the LLM actually calls the ``read_text`` tool, so booting the
GUI without OCR installed stays cheap.

The reader is cached on the singleton :class:`OcrEngine` so the
~100 MB torch model only loads once per process. First call is slow
(several seconds for the load); subsequent calls are ~200 ms.

English only by default. Set ``FACEVIEW_OCR_LANGS`` to a comma-
separated list (e.g. ``"en,fr,de"``) for multi-language reading;
each extra language costs another model download on first use.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

from faceview.core.errors import MissingDependency
from faceview.core.logger import get_logger


log = get_logger("ocr")


def _langs_from_env() -> list[str]:
    raw = os.environ.get("FACEVIEW_OCR_LANGS")
    if not raw:
        return ["en"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or ["en"]


class OcrEngine:
    """Singleton wrapper around an EasyOCR Reader."""

    _instance: "OcrEngine | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "OcrEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = OcrEngine()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._reader = None
        self._lock = threading.Lock()
        self.langs = _langs_from_env()

    def _ensure_reader(self):
        if self._reader is not None:
            return self._reader
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise MissingDependency("easyocr", "vision-tools") from exc
        with self._lock:
            if self._reader is None:
                log.info("ocr.loading", langs=self.langs)
                # gpu=False — EasyOCR's MPS support is patchy on macOS;
                # CPU runs ~200 ms / frame which is fine for on-demand.
                self._reader = easyocr.Reader(self.langs, gpu=False,
                                              verbose=False)
                log.info("ocr.loaded")
        return self._reader

    def read(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.3,
    ) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Return a list of (text, confidence, (x, y, w, h)) tuples."""
        reader = self._ensure_reader()
        # EasyOCR wants RGB.
        try:
            import cv2  # type: ignore
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:  # noqa: BLE001
            rgb = frame
        result = reader.readtext(rgb)
        out: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for bbox_poly, text, conf in result:
            if conf < min_confidence:
                continue
            # bbox_poly is 4 points; reduce to axis-aligned (x, y, w, h)
            xs = [int(p[0]) for p in bbox_poly]
            ys = [int(p[1]) for p in bbox_poly]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            out.append((text, float(conf), (x, y, w, h)))
        return out


def read_text(frame: np.ndarray, region: str = "full") -> str:
    """User-facing helper used by the LLM tool.

    ``region`` follows the same naming as the ``look_at_camera`` tool
    (full / center / top / bottom / left / right / corners). Returns
    a one-paragraph summary the LLM can relay to the user.
    """
    from faceview.llm.vision_tool import _crop_to_region  # lazy: avoid cycle

    if frame is None:
        return "No camera frame is available right now."
    cropped = _crop_to_region(frame, region)
    try:
        engine = OcrEngine.shared()
        items = engine.read(cropped)
    except MissingDependency:
        return ("OCR isn't available — install with "
                "`pip install easyocr` to enable text reading.")
    except Exception as exc:  # noqa: BLE001
        log.warning("ocr.error", error=str(exc))
        return f"OCR failed: {exc}"
    if not items:
        return f"I don't see any readable text in the {region} region."
    lines = [t for t, _c, _b in items]
    joined = " ".join(lines)
    log.info("ocr.read", chars=len(joined),
             items=len(items), region=region)
    return f'I read: "{joined}"'
