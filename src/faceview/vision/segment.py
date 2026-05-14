"""Object segmentation via cv2.GrabCut seeded by EfficientDet bbox.

Full SAM-style models are 400 MB+ and overkill for the use case here
("show me a clean mask of the cup the user is holding"). Instead we
seed OpenCV's classic GrabCut algorithm with the bounding box from
the existing object detector — gives a usable foreground mask in
~100 ms with no new dependencies and no model load.

The result is summarised as text for the LLM (mask coverage,
centroid zone) plus the same status surfacing via PerceptionStore if
we want to track it over time.
"""

from __future__ import annotations

import numpy as np

from faceview.core.logger import get_logger


log = get_logger("segment")


def segment_object(frame: np.ndarray, label: str) -> str:
    """Find the named object in current OBJECTS, run GrabCut, summarise."""
    if frame is None:
        return "No camera frame is available right now."
    try:
        import cv2  # type: ignore
    except ImportError:
        return "OpenCV isn't installed — can't run segmentation."
    if not label or not label.strip():
        return "I need an object label to segment."

    # Look up the most-recent OBJECTS detection matching the label.
    try:
        from faceview.vision.perception import PerceptionStore
        snap = PerceptionStore.shared().snapshot_dict()
    except Exception as exc:  # noqa: BLE001
        return f"Segmentation prep failed: {exc}"
    objs = (snap.get("objects") or {})
    dets = objs.get("detections") or []
    target = next(
        (d for d in dets
         if str(d.get("label", "")).lower() == label.lower().strip()),
        None,
    )
    if target is None:
        return (f"I don't currently see a '{label}' in the OBJECTS list "
                "to segment.")
    x, y, w, h = target.get("bbox", [0, 0, 0, 0])
    if w <= 5 or h <= 5:
        return f"The '{label}' bounding box is too small to segment."

    H, W = frame.shape[:2]
    # Clamp bbox to frame.
    x = max(0, min(W - 2, int(x)))
    y = max(0, min(H - 2, int(y)))
    w = max(1, min(W - x, int(w)))
    h = max(1, min(H - y, int(h)))

    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (x, y, w, h)
    try:
        cv2.grabCut(frame, mask, rect, bgd, fgd,
                    iterCount=3, mode=cv2.GC_INIT_WITH_RECT)
    except Exception as exc:  # noqa: BLE001
        log.warning("segment.grabcut_error", error=str(exc))
        return f"GrabCut failed: {exc}"
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    coverage = float(fg.sum()) / float(W * H)
    if fg.sum() == 0:
        return (f"Segmenting '{label}' produced an empty mask — the "
                "object boundary wasn't separable from the background.")
    # Centroid of the mask in frame-relative coords.
    ys, xs = np.nonzero(fg)
    cx = xs.mean() / max(1, W)
    cy = ys.mean() / max(1, H)
    col = "left" if cx < 0.33 else ("right" if cx > 0.66 else "centre")
    row = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
    log.info("segment.done", label=label, coverage=round(coverage, 3))
    return (f"Segmented '{label}': mask covers {coverage:.1%} of the "
            f"frame, centred in the {row}-{col}.")
