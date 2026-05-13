#!/usr/bin/env python3
"""Top-level entry point for faceView.

Convenience launcher so you can run ``python main.py`` from the
project root instead of ``PYTHONPATH=src python -m faceview``.

By default, launches the GUI in avatar mode (talking head visible
in the camera panel). Pass ``--headless`` for the offscreen smoke
test, or ``--no-avatar`` to start without the avatar worker.

The actual application logic lives in ``src/faceview/__main__.py``;
this file just sets up the import path and dispatches.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Make ``src/`` importable so ``import faceview`` works without
    installing the package."""
    src = Path(__file__).resolve().parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="faceview",
        description="Multimodal desktop GUI for chatting with Claude "
                      "via a talking-head avatar.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run the offscreen smoke test (no GUI window).",
    )
    parser.add_argument(
        "--no-avatar", action="store_true",
        help="Disable the avatar talking-head worker.",
    )
    parser.add_argument(
        "--persona", default=None,
        help="Override the default avatar persona name.",
    )
    args = parser.parse_args()

    _ensure_src_on_path()

    if args.headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if not args.no_avatar:
        os.environ.setdefault("FACEVIEW_AVATAR", "1")
    if args.persona:
        os.environ["FACEVIEW_AVATAR_PERSONA"] = args.persona

    from faceview.__main__ import main as _main
    return int(_main() or 0)


if __name__ == "__main__":
    sys.exit(main())
