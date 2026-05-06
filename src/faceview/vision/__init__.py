"""Webcam capture and per-frame analysis: presence / identity / emotion / mouth.

Heavy ML libs (``cv2``, ``mediapipe``, ``insightface``, ``deepface``) are
imported inside class methods so the GUI shell can boot without them.
"""

from faceview.vision.camera import CameraWorker
from faceview.vision.presence import PresenceDetector
from faceview.vision.identity import IdentityRecognizer
from faceview.vision.emotion import EmotionAnalyzer
from faceview.vision.mouth import MouthAnalyzer
from faceview.vision.sim_face import FaceParams, render_face
from faceview.vision.sim_camera import SimCameraWorker

__all__ = [
    "CameraWorker",
    "PresenceDetector",
    "IdentityRecognizer",
    "EmotionAnalyzer",
    "MouthAnalyzer",
    "FaceParams",
    "render_face",
    "SimCameraWorker",
]
