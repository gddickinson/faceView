"""Audio capture, VAD, STT, TTS workers.

All heavy ML deps are imported lazily inside class methods. Constructing the
worker is cheap and never raises; ``start()`` is what actually loads models
and may raise :class:`MissingDependency` if the right extras aren't installed.
"""

from faceview.speech.audio_capture import AudioCapture
from faceview.speech.tts import TtsWorker
from faceview.speech.stt import SttWorker
from faceview.speech.vad import VadGate

__all__ = ["AudioCapture", "TtsWorker", "SttWorker", "VadGate"]
