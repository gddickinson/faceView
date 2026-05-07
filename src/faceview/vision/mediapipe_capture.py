"""MediaPipe FaceLandmarker live capture — drive the avatar from a webcam.

Uses Google's [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
to extract 478 facial landmarks + 52 ARKit-aligned blendshape
coefficients per frame. Coefficients flow through our existing
ARKit blendshape compatibility layer to drive whichever render
mode is active.

Usage::

    from faceview.vision.mediapipe_capture import MediaPipeCapture
    cap = MediaPipeCapture()                   # opens webcam 0
    coefs = cap.next_frame_blendshapes()       # dict[str, float]
    # then feed into our pipeline:
    from faceview.vision.arkit_blendshapes import arkit_to_au_values
    au_values = arkit_to_au_values(coefs)

This is the input bridge to faceview's avatar pipeline — Mediapipe
output drives the same FACS pipeline that previously came from our
SpeechEngine + AutoBlink + AutoSaccade systems.

Lazy imports gated on ``mediapipe`` and ``opencv-python`` (both in
the ``[vision]`` extra). Without those installed the module stays
importable but instantiation raises a clear MissingDependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from faceview.core.errors import MissingDependency


# Public path for users to override — the FaceLandmarker model file.
MODEL_FILENAME = "face_landmarker_v2_with_blendshapes.task"


def _resolve_model_path(override: Optional[Path]) -> Path:
    if override is not None:
        return Path(override)
    # Default: look in MediaPipe's standard download location, then
    # in our project assets/data/mediapipe/.
    home = Path.home() / ".cache" / "mediapipe" / MODEL_FILENAME
    if home.exists():
        return home
    from faceview.assets import assets_dir
    project = assets_dir() / "data" / "mediapipe" / MODEL_FILENAME
    return project


@dataclass
class MediaPipeCapture:
    """Webcam → MediaPipe → ARKit blendshape coefficients per frame."""
    camera_index: int = 0
    model_path: Optional[Path] = None
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        try:
            import cv2  # noqa: F401
            import mediapipe  # noqa: F401
            from mediapipe.tasks import python  # noqa: F401
        except ImportError as exc:
            raise MissingDependency(
                "mediapipe", "vision",
                hint=(
                    "Install with `pip install -e \".[vision]\"` and "
                    "download the FaceLandmarker model: "
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker_v2_with_blendshapes/"
                    "float16/1/face_landmarker_v2_with_blendshapes.task"
                ),
            ) from exc

        path = _resolve_model_path(self.model_path)
        if not path.exists():
            raise MissingDependency(
                "FaceLandmarker model task file", "vision",
                hint=f"Place model at {path} (see download URL in module docstring).",
            )

        import cv2
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._mp_image_cls = None  # set lazily

    def next_frame_blendshapes(self) -> dict[str, float]:
        """Read one webcam frame, return {arkit_name: coefficient}.

        Returns an empty dict when no face is detected. Caller can
        feed the dict directly into :func:`arkit_to_au_values`.
        """
        import cv2
        import mediapipe as mp

        ok, frame_bgr = self._cap.read()
        if not ok:
            return {}
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.face_blendshapes:
            return {}
        out: dict[str, float] = {}
        for category in result.face_blendshapes[0]:
            # MediaPipe's name field uses lowerCamelCase matching ARKit.
            out[category.category_name] = float(category.score)
        return out

    def close(self) -> None:
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()
            self._cap = None  # type: ignore[assignment]
