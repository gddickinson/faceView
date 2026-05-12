"""Monitor windows — live readouts of audio/emotion/mouth/speech.

Each monitor is a small top-level :class:`QDialog` that subscribes to
specific bus events and renders a focused visualisation. They are
opened from the **Monitor** menu in the main window and let the user
inspect what's happening inside the pipeline at any time.

Module layout:

- ``audio.py``       — waveform-strip + VAD speech/silence pill
- ``emotion.py``     — per-class score bars over time
- ``mouth.py``       — viseme history + jaw_open trace
- ``transcript.py``  — full STT transcript history with timestamps
"""

from __future__ import annotations

from faceview.gui.monitors.audio import AudioMonitor
from faceview.gui.monitors.emotion import EmotionMonitor
from faceview.gui.monitors.mouth import MouthMonitor
from faceview.gui.monitors.transcript import TranscriptMonitor


MONITORS = {
    "audio": AudioMonitor,
    "emotion": EmotionMonitor,
    "mouth": MouthMonitor,
    "transcript": TranscriptMonitor,
}


__all__ = ["MONITORS", "AudioMonitor", "EmotionMonitor", "MouthMonitor", "TranscriptMonitor"]
