"""Offscreen smoke run.

Boots the GUI under ``QT_QPA_PLATFORM=offscreen``, seeds it with demo state
(so panels render with believable content), and saves a screenshot into
``docs/images/headless_smoke.png``. Exits cleanly with code 0.

Usage::

    python -m tools.run_headless
    python -m tools.run_headless --shot docs/images/foo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Must enable offscreen BEFORE PySide6 is imported anywhere.
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FACEVIEW_HEADLESS", "1")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from faceview.core.logger import configure as configure_logging, get_logger
from faceview.gui.main_window import MainWindow


log = get_logger("headless")


def main() -> int:
    parser = argparse.ArgumentParser(description="faceView offscreen smoke")
    parser.add_argument("--shot", default="docs/images/headless_smoke.png")
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Skip demo content seeding; show empty panels.",
    )
    args = parser.parse_args()

    configure_logging()
    log.info("headless.boot")

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    if not args.no_demo:
        win.seed_demo_state()
    win.resize(1280, 800)
    # Force layout so widget sizes are populated before grab().
    win.show()
    QApplication.processEvents()

    out = Path(args.shot).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = win.shotter.capture(win, out)
    log.info("headless.shot_saved", path=str(saved))
    print(f"saved: {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
