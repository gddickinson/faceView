"""Owner-face enrollment flow.

Captures N frames of the user via the bus, embeds each one off-
thread (InsightFace inference is slow), averages the successful
embeddings into a single template, and saves it to
``owner_data/owner.npy`` (which :class:`PeopleStore` then picks up
as a synthetic ``"owner"`` entry).

The handler subscribed to FRAME runs on the GUI thread and only
makes cheap numpy copies; embedding runs in a worker thread so it
doesn't contend with the identity recogniser already running on the
bus thread.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType
from faceview.gui.controllers.base import BaseController


class EnrollmentController(BaseController):
    log_name = "enrollment_ctrl"

    def enroll_owner(self, n_samples: int = 10) -> None:
        identity = self.window.camera_ctrl.identity_recognizer
        if identity is None or not self.window.camera_ctrl.is_running():
            self.status(
                "Enroll: start the camera + identity first "
                "(Tools → Toggle camera)"
            )
            return

        frames: list[np.ndarray] = []
        # Over-capture; some embeds will fail.
        target_frames = max(n_samples * 3, 30)
        last_capture = [0.0]

        def _on_frame(frame) -> None:
            if frame is None or len(frames) >= target_frames:
                return
            # ~5 fps so we get varied poses, not 30 near-duplicates.
            now = time.time()
            if now - last_capture[0] < 0.2:
                return
            last_capture[0] = now
            frames.append(frame.copy())

        bus = get_bus()
        bus.subscribe(EventType.FRAME, _on_frame)
        self.status(
            f"Enrolling… hold still while I grab {target_frames} frames"
        )

        def _finish() -> None:
            t0 = time.time()
            while len(frames) < target_frames and time.time() - t0 < 10.0:
                time.sleep(0.1)
            try:
                bus.unsubscribe(EventType.FRAME, _on_frame)
            except Exception:  # noqa: BLE001
                pass
            if not frames:
                self.status("Enroll failed: no frames captured")
                return
            self.log.info("enroll.frames_captured", count=len(frames))
            samples: list[np.ndarray] = []
            for i, frame in enumerate(frames):
                try:
                    emb = identity.embed(frame)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("enroll.embed_error",
                                     error=str(exc), idx=i)
                    continue
                if emb is not None:
                    samples.append(emb)
                if len(samples) >= n_samples:
                    break
            if not samples:
                self.status(
                    f"Enroll failed: no face detected in any of "
                    f"{len(frames)} frames"
                )
                return
            mean = np.stack(samples).mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            path = identity.save_owner_template(mean)
            self.status(
                f"Enrolled owner from {len(samples)}/{len(frames)} "
                f"frames → {path.name}"
            )

        threading.Thread(target=_finish, name="enroll-owner",
                         daemon=True).start()
