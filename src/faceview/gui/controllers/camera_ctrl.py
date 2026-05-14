"""Camera + vision-analyser lifecycle.

Owns the webcam capture worker plus every downstream analyser that
subscribes to ``EventType.FRAME`` — presence, mouth, emotion,
identity, scene, gestures, objects, ambient scene captioner. Each
analyser is started lazily and degrades gracefully if its ML deps
aren't installed (the relevant status pill simply stays idle).

When the camera is turned off, the captioner is torn down too — it
has nothing to caption without frames and would otherwise keep
poking Ollama on its 15 s timer.
"""

from __future__ import annotations

from faceview.gui.controllers.base import BaseController


class CameraController(BaseController):
    log_name = "camera_ctrl"

    def __init__(self, window) -> None:
        super().__init__(window)
        self._camera_worker = None
        # Vision-pipeline analysers (lazy ML deps)
        self._presence = None
        self._mouth = None
        self._emotion = None
        self._identity = None
        self._scene = None
        self._scene_captioner = None
        self._gestures = None
        self._objects = None

    # ── public API ────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._camera_worker is not None

    def set_enabled(self, on: bool) -> None:
        if on and self._camera_worker is None:
            try:
                from faceview.vision.camera import CameraWorker
                self._camera_worker = CameraWorker()
                self._camera_worker.start()
                self._start_vision_analysers()
                self.status("Camera started + vision analysers up")
            except Exception as exc:  # noqa: BLE001
                self.log.warning("camera.start_failed", error=str(exc))
                self.status(f"Camera unavailable: {exc}")
                self._camera_worker = None
        elif not on and self._camera_worker is not None:
            try:
                self._camera_worker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._camera_worker = None
            # Tear down the ambient captioner too — it has nothing to
            # caption without the camera, and a still-running thread
            # keeps poking Ollama unnecessarily.
            if self._scene_captioner is not None:
                try:
                    self._scene_captioner.stop()
                except Exception:  # noqa: BLE001
                    pass
                self._scene_captioner = None
            self.status("Camera stopped")

    @property
    def identity_recognizer(self):
        """Exposed for enrollment + remember_person tool — they need
        access to the InsightFace handle that lives inside us."""
        return self._identity

    @property
    def camera_worker(self):
        return self._camera_worker

    # ── internals ─────────────────────────────────────────────────

    def _start_vision_analysers(self) -> None:
        """Bring up presence/mouth/emotion/identity/scene/gestures/
        objects/captioner if their deps are present."""
        if self._presence is None:
            try:
                from faceview.vision.presence import PresenceDetector
                self._presence = PresenceDetector()
                self._presence.start()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("presence.start_failed", error=str(exc))
                self._presence = None
        if self._mouth is None:
            try:
                from faceview.vision.mouth import MouthAnalyzer
                self._mouth = MouthAnalyzer()
                self._mouth.start()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("mouth.start_failed", error=str(exc))
                self._mouth = None
        if self._emotion is None:
            try:
                from faceview.vision.emotion import EmotionAnalyzer
                self._emotion = EmotionAnalyzer()
                self._emotion.start()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("emotion.start_failed", error=str(exc))
                self._emotion = None
        if self._identity is None:
            try:
                from faceview.vision.identity import IdentityRecognizer
                self._identity = IdentityRecognizer()
                self._identity.start()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("identity.start_failed", error=str(exc))
                self._identity = None
        # Cheap whole-frame scene descriptors.
        if self._scene is None:
            try:
                from faceview.vision.scene import SceneAnalyzer
                self._scene = SceneAnalyzer()
                self._scene.start()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("scene.start_failed", error=str(exc))
                self._scene = None
        # MediaPipe Gesture Recognizer (downloads model on first run).
        if self._gestures is None:
            try:
                from faceview.vision.gestures import GestureRecognizer
                rec = GestureRecognizer()
                if rec.start():
                    self._gestures = rec
            except Exception as exc:  # noqa: BLE001
                self.log.warning("gestures.start_failed", error=str(exc))
                self._gestures = None
        # MediaPipe Object Detector.
        if self._objects is None:
            try:
                from faceview.vision.objects import ObjectDetector
                det = ObjectDetector()
                if det.start():
                    self._objects = det
            except Exception as exc:  # noqa: BLE001
                self.log.warning("objects.start_failed", error=str(exc))
                self._objects = None
        # Ambient VLM captioner (moondream, ~15 s cadence).
        if self._scene_captioner is None:
            try:
                from faceview.vision.scene_caption import SceneCaptioner
                cap = SceneCaptioner()
                if cap.start():
                    self._scene_captioner = cap
            except Exception as exc:  # noqa: BLE001
                self.log.warning("scene_caption.start_failed",
                                 error=str(exc))
                self._scene_captioner = None
