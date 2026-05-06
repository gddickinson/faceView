"""End-to-end: GUI boots offscreen, takes a screenshot, exits cleanly."""

from __future__ import annotations

from pathlib import Path

from faceview.gui.main_window import MainWindow


def test_main_window_smoke(qtbot, tmp_path: Path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.seed_demo_state()
    win.resize(1280, 800)
    qtbot.waitExposed(win)

    out = tmp_path / "smoke.png"
    saved = win.shotter.capture(win, out)
    assert saved.exists()
    # 1280×800 with 4-byte ARGB at 1 dpr should be at least ~80 KB
    assert saved.stat().st_size > 20_000
