"""QR-code scanner via the built-in OpenCV detector.

No extra dependency, no extra model — :class:`cv2.QRCodeDetector` is
part of base opencv-python and handles standard QR codes well enough
for *"scan this for me"* use cases.

Returns the decoded text payload(s) plus a friendly relay sentence
for the LLM."""

from __future__ import annotations

import numpy as np

from faceview.core.logger import get_logger


log = get_logger("qr")


def scan_qr(frame: np.ndarray) -> str:
    """Decode any QR codes in the frame; return a one-line summary."""
    if frame is None:
        return "No camera frame is available right now."
    try:
        import cv2  # type: ignore
    except ImportError:
        return ("OpenCV isn't installed — install with "
                "`pip install opencv-python` to enable QR scanning.")
    det = cv2.QRCodeDetector()
    payloads: list[str] = []
    try:
        # detectAndDecodeMulti returns (retval, decoded_info, points, _).
        ok, decoded_info, points, _straight = det.detectAndDecodeMulti(frame)
        if ok and decoded_info is not None:
            payloads = [str(s) for s in decoded_info if s]
        if not payloads:
            # Single-QR fallback.
            data, _pts, _rect = det.detectAndDecode(frame)
            if data:
                payloads = [str(data)]
    except Exception as exc:  # noqa: BLE001
        log.warning("qr.error", error=str(exc))
        return f"QR scan failed: {exc}"
    if not payloads:
        return "I don't see a readable QR code in the frame."
    log.info("qr.read", count=len(payloads),
             previews=[p[:40] for p in payloads])
    if len(payloads) == 1:
        return f'QR code says: "{payloads[0]}"'
    bullets = "; ".join(f'"{p}"' for p in payloads[:5])
    return f"I read {len(payloads)} QR codes: {bullets}"
