"""The screenshot helper saves a non-empty PNG offscreen."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

from faceview.gui.screenshotter import Screenshotter


def test_capture_label_writes_png(tmp_path: Path, qtbot):
    label = QLabel("hello world")
    label.resize(320, 80)
    qtbot.addWidget(label)
    qtbot.waitExposed(label)

    out = tmp_path / "label.png"
    saved = Screenshotter().capture(label, out)
    assert saved == out
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_capture_window_uses_docs_images(tmp_path, monkeypatch, qtbot):
    """capture_window writes into docs/images/<name>.png by default."""
    label = QLabel("docs")
    label.resize(120, 60)
    qtbot.addWidget(label)
    qtbot.waitExposed(label)

    monkeypatch.setattr(
        "faceview.gui.screenshotter.docs_image_dir",
        lambda: tmp_path,
    )
    saved = Screenshotter().capture_window(label, "test")
    assert saved.name == "test.png"
    assert saved.parent == tmp_path
    assert saved.exists()
