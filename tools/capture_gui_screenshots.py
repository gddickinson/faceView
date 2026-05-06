"""Drive the GUI through several states and capture screenshots for the README.

Each scene seeds the panels with different demo content and starts the
simulated camera with a different scenario, then saves a PNG into
``docs/images/``. Designed to run fully offscreen — safe for CI.

Usage::

    python -m tools.capture_gui_screenshots
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Must be set before any PySide6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FACEVIEW_HEADLESS", "1")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from faceview.core.event_bus import get_bus  # noqa: E402
from faceview.core.events import (  # noqa: E402
    ChatMessage,
    Emotion,
    EventType,
    Identity,
    MouthActivity,
    Presence,
    Transcript,
)
from faceview.core.logger import configure as configure_logging, get_logger  # noqa: E402
from faceview.gui.main_window import MainWindow  # noqa: E402
from faceview.utils.paths import docs_image_dir  # noqa: E402
from faceview.vision.sim_camera import SimCameraWorker  # noqa: E402
from faceview.vision.sim_face import FaceParams, render_face  # noqa: E402


log = get_logger("capture")


def _wait(ms: int) -> None:
    """Spin the Qt event loop for ``ms`` so animations and signals settle."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _push_one_frame(params: FaceParams) -> None:
    """Push one frame + matching events directly (no thread)."""
    frame = render_face(params, (640, 480))
    bus = get_bus()
    bus.publish(EventType.FRAME, frame)
    bus.publish(EventType.PRESENCE, Presence(face_count=1))
    speaking = params.jaw_open > 0.07
    bus.publish(
        EventType.MOUTH_ACTIVITY,
        MouthActivity(
            speaking=speaking,
            jaw_open=params.jaw_open,
            mouth_funnel=max(0.0, params.jaw_open - 0.05),
            mouth_pucker=max(0.0, -params.smile) * 0.6,
            viseme="AA" if params.jaw_open > 0.25 else ("EE" if params.smile > 0.5 else None),
        ),
    )
    if params.brow_raise > 0.6 and params.jaw_open > 0.3:
        bus.publish(EventType.EMOTION, Emotion(label="surprise", confidence=0.84))
    elif params.smile > 0.5:
        bus.publish(EventType.EMOTION, Emotion(label="happy", confidence=0.81))
    elif params.smile < -0.3:
        bus.publish(EventType.EMOTION, Emotion(label="sad", confidence=0.66))
    else:
        bus.publish(EventType.EMOTION, Emotion(label="neutral", confidence=0.74))
    bus.publish(EventType.IDENTITY, Identity(is_owner=True, similarity=0.71, label="sim-owner"))


def scene_main(window: MainWindow) -> None:
    """Default landing screenshot — owner present, neutral, idle."""
    window.chat.history.clear()
    window.transcript.view.clear()
    window.chat.seed_demo_conversation()
    window.transcript.seed_demo()
    _push_one_frame(FaceParams(smile=0.18, brow_raise=0.0, eye_open=0.95))
    _wait(150)
    window.statusBar().showMessage(
        "owner present · neutral · mic idle · http://127.0.0.1:8765 · MCP ready"
    )


def scene_happy(window: MainWindow) -> None:
    window.chat.history.clear()
    bus = get_bus()
    bus.publish(EventType.CHAT_USER_MESSAGE, ChatMessage("user", "Tell me a joke!"))
    bus.publish(EventType.LLM_REPLY, ChatMessage(
        "assistant",
        "Why did the camera bring a notebook?\n— Because it kept losing focus.",
    ))
    _push_one_frame(FaceParams.happy())
    _wait(150)
    window.statusBar().showMessage("happy · smiling · mouth slightly open")


def scene_speaking(window: MainWindow) -> None:
    window.transcript.view.clear()
    bus = get_bus()
    bus.publish(EventType.VAD_SPEECH_START, None)
    bus.publish(EventType.TRANSCRIPT_PARTIAL, Transcript("Hey Claude can you", is_final=False))
    bus.publish(EventType.TRANSCRIPT_FINAL, Transcript("Hey Claude can you take a screenshot?", is_final=True))
    p = FaceParams.speaking(0.6)
    p.jaw_open = 0.34
    p.smile = 0.2
    _push_one_frame(p)
    _wait(150)
    window.statusBar().showMessage("speaking detected · viseme=AA · STT engaged")


def scene_surprised(window: MainWindow) -> None:
    _push_one_frame(FaceParams.surprised())
    _wait(150)
    window.statusBar().showMessage("emotion: surprise · brow raised · jaw open")


def scene_no_face(window: MainWindow) -> None:
    bus = get_bus()
    bus.publish(EventType.PRESENCE, Presence(face_count=0))
    bus.publish(EventType.IDENTITY, Identity(is_owner=False, similarity=0.0, label="absent"))
    bus.publish(EventType.EMOTION, Emotion(label="neutral", confidence=0.0))
    bus.publish(EventType.MOUTH_ACTIVITY, MouthActivity(speaking=False, jaw_open=0, mouth_funnel=0, mouth_pucker=0))
    # Black-ish frame to indicate no one there.
    import numpy as np
    bus.publish(EventType.FRAME, np.zeros((480, 640, 3), dtype=np.uint8))
    _wait(150)
    window.statusBar().showMessage("no face detected · pausing camera analysis")


SCENES: list[tuple[str, callable]] = [
    ("main", scene_main),
    ("happy", scene_happy),
    ("speaking", scene_speaking),
    ("surprised", scene_surprised),
    ("absent", scene_no_face),
]


def main() -> int:
    configure_logging()

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    _wait(120)

    out_dir = docs_image_dir()
    saved: list[Path] = []

    for name, fn in SCENES:
        log.info("scene.start", name=name)
        fn(window)
        path = window.shotter.capture_window(window, name)
        saved.append(path)

    # Standalone face crops (helpful in the README's "what is sim-face" section).
    for label, params in [
        ("face_neutral", FaceParams.neutral()),
        ("face_happy", FaceParams.happy()),
        ("face_surprised", FaceParams.surprised()),
        ("face_sad", FaceParams.sad()),
    ]:
        from PIL import Image
        arr = render_face(params, (320, 320))
        p = out_dir / f"{label}.png"
        Image.fromarray(arr[:, :, ::-1]).save(p)
        saved.append(p)

    print("captured:")
    for p in saved:
        print(" ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
