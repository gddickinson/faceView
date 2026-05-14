"""Face attributes (age + gender) via InsightFace.

InsightFace's ``buffalo_l`` bundle already includes a ``genderage``
ONNX model that's loaded by :class:`IdentityRecognizer` for owner
matching. This module reuses that same loaded model — zero extra
memory cost — to expose age + gender estimates as an on-demand tool.

Falls back gracefully if no IdentityRecognizer is running (no face
embedder available)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from faceview.core.logger import get_logger


log = get_logger("face_attr")


def _find_running_app():
    """Locate the FaceAnalysis instance owned by IdentityRecognizer,
    if any. Returns ``None`` when identity isn't running."""
    try:
        from faceview.vision.people import PeopleStore  # noqa: F401
        # PeopleStore holds an embed_fn — but we need the full FaceAnalysis
        # instance for the gender/age outputs, not just embed_fn. So walk
        # via IdentityRecognizer instances via gc isn't ideal — instead
        # we expose the app through a module-level handle.
    except Exception:  # noqa: BLE001
        pass
    return _APP_HANDLE.get("app")


_APP_HANDLE: dict = {"app": None}


def register_app(app) -> None:
    """Called by IdentityRecognizer.start() so the tool can reach the
    already-loaded FaceAnalysis without owning a duplicate."""
    _APP_HANDLE["app"] = app
    log.info("face_attr.app_registered")


def face_attributes(frame: np.ndarray) -> str:
    """Return a short sentence with age + gender for the largest face."""
    if frame is None:
        return "No camera frame is available right now."
    app = _find_running_app()
    if app is None:
        return ("Face attribute analysis needs the identity recognizer "
                "to be running first — turn the camera on.")
    try:
        faces = app.get(frame)
    except Exception as exc:  # noqa: BLE001
        log.warning("face_attr.error", error=str(exc))
        return f"Face analysis failed: {exc}"
    if not faces:
        return ("I don't see a clear face in the frame to estimate "
                "age or gender from.")
    faces.sort(
        key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    f = faces[0]
    age = int(getattr(f, "age", 0) or 0)
    sex = getattr(f, "sex", None)
    gender_str = (
        "female" if sex == "F" else ("male" if sex == "M" else "unknown")
    )
    n = len(faces)
    log.info("face_attr.done", age=age, sex=gender_str, faces=n)
    suffix = "" if n == 1 else f" (one of {n} faces visible)"
    return (f"Estimated about {age} years old, apparently {gender_str}"
            f"{suffix}. (Age estimates from InsightFace are rough — "
            "treat as ±5 years.)")
