"""DECA / EMOCA — image-to-3D-face capture pipeline.

[DECA](https://github.com/yfeng95/DECA) takes a single image of a
face and reconstructs detailed FLAME parameters (shape, expression,
pose, jaw). [EMOCA](https://github.com/radekd91/emoca) extends DECA
with emotion fidelity. The newer
[INFERNO](https://github.com/radekd91/inferno) library supersedes
both.

USE CASE
--------
*Capture* path: webcam frame → DECA → FLAME shape + expression
parameters → drive our avatar. Alternative to the MediaPipe
FaceLandmarker bridge with higher fidelity but heavy ML deps.

DEPS + DATA
-----------
Heavy: torch, pytorch3d, face-alignment, plus DECA's pretrained
checkpoint (~1 GB) downloaded from the repo. Activates only when
``inferno`` or ``deca`` is on PYTHONPATH.

This module exposes a thin :class:`DECACapture` wrapper that lazy-
imports and gracefully raises :class:`MissingDependency` until
the user opts in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faceview.core.errors import MissingDependency


@dataclass
class DECACapture:
    """Wrap DECA's image → FLAME params pipeline."""

    checkpoint_dir: str = ""

    def __post_init__(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise MissingDependency(
                "torch", "vision",
                hint="Install with `pip install torch`.",
            ) from exc
        try:
            from decalib.deca import DECA  # type: ignore[import-not-found]
            from decalib.utils.config import cfg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependency(
                "DECA / decalib", "vision",
                hint=(
                    "Clone https://github.com/yfeng95/DECA, "
                    "download the pretrained checkpoint, and put "
                    "the repo on PYTHONPATH. The newer "
                    "https://github.com/radekd91/inferno library "
                    "is the recommended replacement."
                ),
            ) from exc
        if self.checkpoint_dir:
            cfg.pretrained_modelpath = self.checkpoint_dir
        self._deca = DECA(config=cfg)

    def fit_to_image(self, image_bgr: np.ndarray) -> dict:
        """Run DECA on one frame, return FLAME-style parameter dict."""
        # Import torch here so the lazy-import path stays clean.
        import torch
        rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            codedict = self._deca.encode(tensor)
        # codedict has shape, exp, pose, cam, light, tex tensors.
        # Strip to numpy for downstream consumption.
        return {k: v.cpu().numpy() if hasattr(v, "cpu") else v
                for k, v in codedict.items()}

    def to_au_values(self, codedict: dict) -> dict[str, float]:
        """Best-effort mapping from FLAME exp coefficients to our AU set.

        FLAME's exp coefficients are not 1:1 with FACS; this is a
        coarse heuristic. For high fidelity, prefer the dedicated
        FLAME → ARKit blendshape solver in EMOCA / INFERNO.
        """
        exp = codedict.get("exp")
        if exp is None:
            return {}
        e = np.asarray(exp).flatten()
        # First several exp components correlate roughly with AU-like
        # actions in trained models — empirical mapping.
        return {
            "AU1": float(np.clip(e[0] if len(e) > 0 else 0, -1, 1)),
            "AU4": float(np.clip(-e[1] if len(e) > 1 else 0, 0, 1)),
            "AU12": float(np.clip(e[2] if len(e) > 2 else 0, 0, 1)),
            "AU26": float(np.clip(e[5] if len(e) > 5 else 0, 0, 1)),
        }
