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
from faceview.vision.face_state import AU_IDS, FaceState, face_state_to_params
from faceview.vision.avatar import TalkingAvatar
from faceview.vision.speech import SpeechEngine
from faceview.vision.expressions import apply_expression, expression_names

__all__ = [
    "CameraWorker",
    "PresenceDetector",
    "IdentityRecognizer",
    "EmotionAnalyzer",
    "MouthAnalyzer",
    "FaceParams",
    "render_face",
    "SimCameraWorker",
    "FaceState",
    "AU_IDS",
    "face_state_to_params",
    "TalkingAvatar",
    "SpeechEngine",
    "apply_expression",
    "expression_names",
]
