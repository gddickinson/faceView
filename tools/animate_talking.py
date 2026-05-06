"""Record a TalkingAvatar speaking a phrase to GIF + frame strip + monitor PNG.

Drives :class:`faceview.vision.avatar.TalkingAvatar` at fixed dt for the full
duration of an utterance plus a short tail, captures every frame, and saves:

- ``docs/images/avatar_talking.gif`` — animated GIF (loops once per second
  delay; matches the avatar's actual timing).
- ``docs/images/avatar_strip.png``  — 6-frame contact sheet for the README.
- ``docs/images/avatar_monitor.png`` — the GUI mid-speech with the talking
  face shown in the camera panel and a chat exchange in the centre column.

Usage::

    python -m tools.animate_talking
    python -m tools.animate_talking --text "Hi! I'm faceView." --emotion happy
    python -m tools.animate_talking --speed 0.9 --fps 24
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before any PySide6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FACEVIEW_HEADLESS", "1")

from PIL import Image  # noqa: E402
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
)
from faceview.core.logger import configure as configure_logging, get_logger  # noqa: E402
from faceview.gui.main_window import MainWindow  # noqa: E402
from faceview.utils.paths import docs_image_dir  # noqa: E402
from faceview.vision.avatar import TalkingAvatar  # noqa: E402
from faceview.vision.sim_face import render_face  # noqa: E402


log = get_logger("animate")


def _spin(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def render_gif(
    text: str,
    *,
    emotion: str = "happy",
    fps: int = 24,
    speed: float = 1.0,
    size: tuple[int, int] = (480, 480),
    out_gif: Path,
    out_strip: Path,
    strip_frames: int = 6,
) -> dict:
    """Run the avatar through ``text`` and save a GIF and contact-sheet PNG."""
    # The avatar's clock is in real-time. Use a fake-time tick so we can dump
    # frames as fast as the renderer allows, and the GIF still shows correct
    # timing because we set the per-frame delay to 1000/fps ms.
    av = TalkingAvatar(emotion=emotion, seed=7)
    utt = av.say(text, speed=speed)
    total = max(1.0, utt.duration + 0.7)
    n = int(total * fps)
    frames: list[Image.Image] = []

    log.info("animate.start", text=text, fps=fps, n=n, dur=total)
    for i in range(n):
        t = i / fps
        params = av.tick(t)
        bgr = render_face(params, size)
        # PIL wants RGB.
        frames.append(Image.fromarray(bgr[:, :, ::-1]))

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    delay = max(40, int(1000 / fps))
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=delay,
        loop=0,
        optimize=False,
        disposal=2,
    )
    log.info("animate.gif_saved", path=str(out_gif), frames=len(frames), delay_ms=delay)

    # Contact sheet — sample N evenly-spaced frames.
    picks = [int(i * (len(frames) - 1) / max(1, strip_frames - 1)) for i in range(strip_frames)]
    cell_w, cell_h = size
    sheet = Image.new("RGB", (cell_w * strip_frames, cell_h), (12, 15, 20))
    for col, idx in enumerate(picks):
        sheet.paste(frames[idx], (col * cell_w, 0))
    sheet.save(out_strip)
    log.info("animate.strip_saved", path=str(out_strip), cols=strip_frames)

    return {"frames": len(frames), "duration": total, "gif": str(out_gif), "strip": str(out_strip)}


def render_monitor_png(text: str, emotion: str, out: Path) -> Path:
    """Take a screenshot of the GUI mid-utterance with the avatar in the camera panel."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    _spin(80)

    # Drive a still frame of the avatar near the middle of the utterance so
    # the captured shot shows an obvious open mouth.
    av = TalkingAvatar(emotion=emotion, seed=7)
    utt = av.say(text)
    mid = utt.duration * 0.45
    # Run several ticks up to mid so the smoothing has converged.
    for i in range(int(mid * 30) + 1):
        params = av.tick(i / 30.0)
    frame = render_face(params, (640, 480))

    bus = get_bus()
    bus.publish(EventType.FRAME, frame)
    bus.publish(EventType.PRESENCE, Presence(face_count=1))
    bus.publish(EventType.IDENTITY, Identity(is_owner=True, similarity=0.71, label="claude-avatar"))
    bus.publish(EventType.EMOTION, Emotion(label=emotion, confidence=0.83))
    bus.publish(EventType.MOUTH_ACTIVITY, MouthActivity(
        speaking=True, jaw_open=params.jaw_open,
        mouth_funnel=max(0.0, params.jaw_open - 0.05),
        mouth_pucker=max(0.0, -params.smile) * 0.6,
        viseme="AA",
    ))

    bus.publish(EventType.CHAT_USER_MESSAGE, ChatMessage("user", "Say something animated."))
    bus.publish(EventType.LLM_REPLY, ChatMessage("assistant", text))
    win.statusBar().showMessage(
        f"avatar speaking: {emotion} · viseme stream live · jaw_open={params.jaw_open:.2f}"
    )
    _spin(120)
    saved = win.shotter.capture(win, out)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="Hi! I'm faceView. I can see, hear, and talk.",
        help="Phrase to animate.",
    )
    parser.add_argument("--emotion", default="happy")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--size", type=int, nargs=2, default=[480, 480])
    args = parser.parse_args()

    configure_logging()

    # PIL needs a QApplication to use the QPainter inside render_face.
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    images = docs_image_dir()
    info = render_gif(
        args.text,
        emotion=args.emotion,
        fps=args.fps,
        speed=args.speed,
        size=tuple(args.size),
        out_gif=images / "avatar_talking.gif",
        out_strip=images / "avatar_strip.png",
    )
    monitor = render_monitor_png(args.text, args.emotion, images / "avatar_monitor.png")
    info["monitor"] = str(monitor)

    print("animation done:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
