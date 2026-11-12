"""Webcam capture and per-frame analysis: presence / identity / emotion / mouth.

Heavy ML libs (``cv2``, ``mediapipe``, ``insightface``, ``deepface``) are
imported inside class methods so the GUI shell can boot without them.

Exports are resolved lazily (PEP 562) so the Qt-free modules in this package
(e.g. the animation and character code used by the web interface) import without PySide6.
"""

from importlib import import_module

_EXPORTS = {
    "CameraWorker": "faceview.vision.camera",
    "PresenceDetector": "faceview.vision.presence",
    "IdentityRecognizer": "faceview.vision.identity",
    "EmotionAnalyzer": "faceview.vision.emotion",
    "MouthAnalyzer": "faceview.vision.mouth",
    "FaceParams": "faceview.vision.face_params",
    "render_face": "faceview.vision.sim_face",
    "SimCameraWorker": "faceview.vision.sim_camera",
    "FaceState": "faceview.vision.face_state",
    "AU_IDS": "faceview.vision.face_state",
    "face_state_to_params": "faceview.vision.face_state",
    "TalkingAvatar": "faceview.vision.avatar",
    "SpeechEngine": "faceview.vision.speech",
    "apply_expression": "faceview.vision.expressions",
    "expression_names": "faceview.vision.expressions",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
