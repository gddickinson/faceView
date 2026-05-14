"""Capture fresh screenshots + a couple of small GIFs for the new
features shipped since the last README refresh.

Run offscreen — safe for CI::

    python -m tools.capture_recent_features

Drops PNGs (and the odd GIF) into ``docs/images/`` with the
``recent_*`` prefix so the README can reference them stably.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FACEVIEW_HEADLESS", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from faceview.core.event_bus import get_bus  # noqa: E402
from faceview.core.events import (  # noqa: E402
    Blink,
    ChatMessage,
    DetectedObject,
    Emotion,
    EventType,
    FaceDistance,
    Gaze,
    Gesture,
    HeadPose,
    Identity,
    MouthActivity,
    ObjectsSeen,
    PixelTransmission,
    Presence,
    RoomMap,
    RoomMapItem,
    SceneCaption,
    SceneInfo,
    TurnRecord,
)
from faceview.core.logger import configure as configure_logging, get_logger  # noqa: E402
from faceview.gui.main_window import MainWindow  # noqa: E402
from faceview.utils.paths import docs_image_dir  # noqa: E402


log = get_logger("capture_recent")


def _wait(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _seed_perception(strong: bool = True) -> None:
    """Push synthetic events so PerceptionStore + StatusPanel show
    realistic data without a real camera/mic."""
    bus = get_bus()
    bus.publish(EventType.PRESENCE, Presence(face_count=1))
    bus.publish(EventType.IDENTITY,
                Identity(is_owner=False, similarity=0.81, label="George"))
    bus.publish(EventType.EMOTION,
                Emotion(label="happy" if strong else "neutral",
                        confidence=0.78))
    bus.publish(EventType.MOUTH_ACTIVITY,
                MouthActivity(speaking=False, jaw_open=0.04,
                              mouth_funnel=0.02, mouth_pucker=0.01))
    bus.publish(EventType.HEAD_POSE,
                HeadPose(yaw=0.12, pitch=-0.04, roll=0.02))
    bus.publish(EventType.GAZE,
                Gaze(direction="camera", yaw=0.05, pitch=0.0,
                     attention=0.91))
    bus.publish(EventType.FACE_DISTANCE,
                FaceDistance(label="normal", bbox_ratio=0.18))
    bus.publish(EventType.BLINK,
                Blink(eye_open=0.30, state="open", rate_per_min=14.0))
    bus.publish(EventType.GESTURE,
                Gesture(label="thumbs_up", hand="right",
                        confidence=0.88))
    bus.publish(EventType.SCENE,
                SceneInfo(brightness=0.62, brightness_label="lit",
                          motion=0.04, motion_label="still"))
    bus.publish(EventType.OBJECTS, ObjectsSeen(detections=[
        DetectedObject(label="person", score=0.96,
                       bbox=(120, 60, 380, 380)),
        DetectedObject(label="cup", score=0.78,
                       bbox=(520, 290, 80, 90)),
        DetectedObject(label="laptop", score=0.84,
                       bbox=(80, 380, 320, 80)),
    ]))
    bus.publish(EventType.SCENE_CAPTION, SceneCaption(
        text="A person sitting at a desk with a coffee cup beside an open laptop.",
        model="moondream",
        latency_s=1.6,
    ))


def _seed_room_map() -> None:
    bus = get_bus()
    bus.publish(EventType.ROOM_MAP, RoomMap(
        items=[
            RoomMapItem(label="person",  x=0.05,  z=1.20),
            RoomMapItem(label="cup",     x=0.55,  z=0.95),
            RoomMapItem(label="laptop",  x=-0.65, z=1.45),
            RoomMapItem(label="book",    x=-0.20, z=2.10),
        ],
        frame_w=1280, frame_h=720, hfov_deg=65.0, units="metres",
    ))


def scene_perception(window: MainWindow) -> Path:
    """Capture the perception panel with all signals populated."""
    _seed_perception(strong=True)
    _seed_room_map()  # so the perception panel also shows tracker noise
    _wait(300)
    # Capture just the perception panel widget.
    return window.shotter.capture(
        window.perception_panel, "recent_perception_panel.png",
    )


def scene_status_telemetry(window: MainWindow) -> Path:
    """Status panel after firing a TURN_RECORDED + a PIXELS_LEAVING."""
    _seed_perception(strong=True)
    bus = get_bus()
    bus.publish(EventType.TURN_RECORDED, TurnRecord(
        engine="anthropic", model="claude-sonnet-4-6",
        duration_s=1.42, prompt_tokens=312, completion_tokens=187,
        usd_cost=0.00375,
    ))
    bus.publish(EventType.PIXELS_LEAVING, PixelTransmission(
        active=True, destination="anthropic", tool="look_at_camera",
    ))
    _wait(200)
    return window.shotter.capture(
        window.status_panel, "recent_status_telemetry.png",
    )


def scene_chat_markdown(window: MainWindow) -> Path:
    """Chat panel with the seeded demo content (code fence + emphasis)."""
    window.chat._blocks = []
    window.chat._live_block = None
    window.chat.seed_demo_conversation()
    _wait(200)
    return window.shotter.capture(window.chat, "recent_chat_markdown.png")


def scene_chat_find(window: MainWindow) -> Path:
    """Chat panel with the Ctrl+F find bar showing."""
    window.chat._blocks = []
    window.chat._live_block = None
    window.chat.seed_demo_conversation()
    window.chat.find_bar.set_query("screenshot")
    window.chat.find_bar.show()
    window.chat._do_find("screenshot", backward=False)
    _wait(200)
    return window.shotter.capture(window.chat, "recent_chat_find.png")


def scene_room_map(window: MainWindow) -> Path:
    """Open the Room Map window + seed it with synthetic items."""
    window.open_room_map()
    _seed_room_map()
    _wait(250)
    bus = get_bus()
    bus.publish(EventType.HEAD_POSE,
                HeadPose(yaw=0.18, pitch=0.0, roll=0.0))
    _wait(200)
    return window.shotter.capture_window(
        window.room_map_window, "recent_room_map.png",
    )


def scene_calibration_dialog(window: MainWindow) -> Path:
    """Capture the room-map calibration dialog."""
    _seed_room_map()
    _wait(200)
    if window.room_map_window is None:
        window.open_room_map()
    _wait(200)
    from faceview.gui.room_map_panel import _CalibrationDialog
    dlg = _CalibrationDialog(
        window.room_map_window.canvas._items, window.room_map_window,
    )
    dlg.show()
    _wait(200)
    return window.shotter.capture_window(
        dlg, "recent_calibration_dialog.png",
    )


def scene_main_with_perception(window: MainWindow) -> Path:
    """The whole main window after seeding everything — replaces the
    aging readme_main.png hero shot."""
    window.chat._blocks = []
    window.chat._live_block = None
    window.chat.seed_demo_conversation()
    window.transcript.seed_demo()
    _seed_perception(strong=True)
    _seed_room_map()
    _wait(300)
    return window.shotter.capture_window(window, "recent_main_window.png")


def scene_dark_theme(window: MainWindow) -> Path:
    """Same hero shot but with dark theme active (U6)."""
    from faceview.gui.theme import apply_theme
    apply_theme("dark")
    window.chat._blocks = []
    window.chat._live_block = None
    window.chat.seed_demo_conversation()
    window.transcript.seed_demo()
    _seed_perception(strong=True)
    _seed_room_map()
    _wait(300)
    path = window.shotter.capture_window(window, "recent_dark_theme.png")
    # Restore system theme so the next scenes look normal.
    apply_theme("system")
    return path


def scene_telemetry_with_pixels(window: MainWindow) -> Path:
    """Status panel showing telemetry + the red recording indicator
    flashing during a real Anthropic round-trip (simulated)."""
    _seed_perception(strong=True)
    bus = get_bus()
    bus.publish(EventType.TURN_RECORDED, TurnRecord(
        engine="anthropic", model="claude-sonnet-4-6",
        duration_s=1.42, prompt_tokens=312, completion_tokens=187,
        usd_cost=0.00375,
    ))
    bus.publish(EventType.PIXELS_LEAVING, PixelTransmission(
        active=True, destination="anthropic", tool="look_at_camera",
    ))
    _wait(200)
    return window.shotter.capture(
        window.status_panel, "recent_status_telemetry.png",
    )


# ── streaming chat GIF ──────────────────────────────────────────


def _capture_widget_png(widget) -> Image.Image:
    """Grab the widget as a PIL Image via QBuffer (no raw bits())."""
    from io import BytesIO
    from PySide6.QtCore import QBuffer, QIODevice
    pix = widget.grab()
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pix.save(buf, "PNG")
    return Image.open(BytesIO(buf.data())).copy().convert("RGB")


def gif_chat_streaming(window: MainWindow) -> Path:
    """Animate Claude streaming a markdown reply token-by-token."""
    bus = get_bus()
    window.chat._blocks = []
    window.chat._live_block = None
    # User opener.
    bus.publish(EventType.CHAT_USER_MESSAGE,
                ChatMessage("user",
                            "Show me a quick Python snippet to "
                            "count words in a string."))
    _wait(150)

    frames: list[Image.Image] = []
    # Capture an initial frame (just the user line + cursor blink).
    frames.append(_capture_widget_png(window.chat))

    # Stream tokens.
    streamed = (
        "Sure — here you go:\n\n"
        "```python\n"
        "def word_count(s: str) -> int:\n"
        "    return len(s.split())\n"
        "```\n\n"
        "It splits on whitespace; **easy to extend** for case-folding."
    )
    chunks = [streamed[i : i + 8] for i in range(0, len(streamed), 8)]
    for tok in chunks:
        bus.publish(EventType.LLM_TOKEN, tok)
        _wait(40)
    frames.append(_capture_widget_png(window.chat))
    # Final reply triggers markdown re-render.
    bus.publish(EventType.LLM_REPLY,
                ChatMessage("assistant", streamed))
    _wait(120)
    frames.append(_capture_widget_png(window.chat))

    # Build the GIF — a few intermediate frames plus the final
    # markdown-rendered one held for ~1 s.
    out = docs_image_dir() / "recent_chat_streaming.gif"
    # Re-capture two more frames at the final state for the hold.
    frames.append(frames[-1])
    frames.append(frames[-1])
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=400, loop=0, optimize=False,
    )
    log.info("gif.saved", path=str(out), frames=len(frames))
    return out


# ── runner ──────────────────────────────────────────────────────


def main() -> int:
    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 860)
    window.show()
    _wait(160)

    out_dir = docs_image_dir()
    saved: list[Path] = []

    for fn in (
        scene_main_with_perception,
        scene_dark_theme,
        scene_perception,
        scene_status_telemetry,
        scene_chat_markdown,
        scene_chat_find,
        scene_room_map,
        scene_calibration_dialog,
    ):
        try:
            p = fn(window)
            saved.append(p)
            log.info("scene.captured", path=str(p))
        except Exception as exc:  # noqa: BLE001
            log.warning("scene.failed", scene=fn.__name__, error=str(exc))

    # GIFs — best-effort; if PIL chokes on a frame just skip.
    try:
        saved.append(gif_chat_streaming(window))
    except Exception as exc:  # noqa: BLE001
        log.warning("gif.streaming.failed", error=str(exc))

    print("captured:")
    for p in saved:
        print(" ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
