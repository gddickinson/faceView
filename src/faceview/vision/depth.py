"""Monocular depth estimation via MiDaS-small.

Loaded lazily via ``torch.hub`` — the first call downloads ~80 MB of
weights into the user's torch hub cache. Subsequent calls reuse the
loaded model (held in a singleton).

Output is a coarse, qualitative read of the scene: which third of the
frame is "near" vs "far", plus an overall description ("the user is
clearly the closest thing to the camera, with the background ~2× as
far"). Returning raw depth maps to the LLM is overkill — coarse text
is more useful for chat answers.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from faceview.core.errors import MissingDependency
from faceview.core.logger import get_logger


log = get_logger("depth")


class DepthEstimator:
    _instance: "DepthEstimator | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "DepthEstimator":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = DepthEstimator()
        return cls._instance

    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise MissingDependency("torch", "vision-tools") from exc
        with self._lock:
            if self._model is not None:
                return
            log.info("depth.loading_midas")
            # MiDaS via torch.hub — small variant for speed.
            try:
                model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small",
                    trust_repo=True, source="github",
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"MiDaS download/load failed: {exc}"
                ) from exc
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True, source="github",
            )
            transform = transforms.small_transform
            model.to(self._device).eval()
            self._model = model
            self._transform = transform
            self._torch = torch
            log.info("depth.loaded")

    def depth_map(self, frame: np.ndarray) -> np.ndarray:
        """Return a (H, W) float32 array — larger values mean nearer."""
        self._ensure()
        torch = self._torch
        try:
            import cv2  # type: ignore
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:  # noqa: BLE001
            rgb = frame
        input_tensor = self._transform(rgb)
        if hasattr(input_tensor, "to"):
            input_tensor = input_tensor.to(self._device)
        with torch.no_grad():
            pred = self._model(input_tensor)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return pred.cpu().numpy().astype(np.float32)


def estimate_depth(frame: np.ndarray, region: str = "full") -> str:
    """One-paragraph depth read of the (region of the) frame."""
    from faceview.llm.vision_tool import _crop_to_region  # lazy: cycle

    if frame is None:
        return "No camera frame is available right now."
    crop = _crop_to_region(frame, region)
    try:
        d = DepthEstimator.shared().depth_map(crop)
    except MissingDependency:
        return ("Depth estimation needs torch — install with "
                "`pip install torch` to enable.")
    except Exception as exc:  # noqa: BLE001
        log.warning("depth.tool_error", error=str(exc))
        return f"Depth estimation failed: {exc}"
    if d is None or d.size == 0:
        return "Depth estimation produced no output."
    # Normalise to 0..1 (larger = nearer per MiDaS convention).
    dn = (d - d.min()) / max(1e-6, (d.max() - d.min()))
    h, w = dn.shape
    near = dn > 0.66
    far = dn < 0.33
    near_share = float(near.mean())
    far_share = float(far.mean())
    # Where is "near" concentrated? Compare top vs bottom / left vs right.
    top_near = float(near[: h // 2].mean())
    bot_near = float(near[h // 2:].mean())
    left_near = float(near[:, : w // 2].mean())
    right_near = float(near[:, w // 2:].mean())
    bits = [
        f"near plane covers {near_share:.0%} of the {region} region",
        f"far plane covers {far_share:.0%}",
    ]
    if abs(top_near - bot_near) > 0.1:
        bits.append("nearest things at the "
                    + ("top" if top_near > bot_near else "bottom"))
    if abs(left_near - right_near) > 0.1:
        bits.append("nearest things on the "
                    + ("left" if left_near > right_near else "right"))
    log.info("depth.described", region=region, near=near_share, far=far_share)
    return "Depth read: " + "; ".join(bits) + "."
