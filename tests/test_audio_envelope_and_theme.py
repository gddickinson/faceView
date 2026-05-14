"""A47 + U6 — TTS amplitude envelope + dark/light theme switch."""

from __future__ import annotations


# ── A47: amplitude envelope from PCM samples ────────────────────


def test_compute_envelope_normalises_to_unit():
    from faceview.speech.tts_kokoro import _compute_envelope
    import numpy as np
    # Single-tone, constant amplitude → every window should map to 1.0
    # (the max).
    sr = 16_000
    samples = np.ones(sr // 2, dtype=np.float32) * 0.4
    env = _compute_envelope(samples, sr, window_ms=30)
    assert len(env) > 0
    # All windows have the same RMS → all normalised to 1.0.
    for _t, a in env:
        assert abs(a - 1.0) < 1e-3


def test_compute_envelope_silence_is_zero():
    from faceview.speech.tts_kokoro import _compute_envelope
    import numpy as np
    sr = 16_000
    env = _compute_envelope(np.zeros(sr, dtype=np.float32), sr)
    # All silent → normalisation by the trivial max keeps everything 0.
    assert all(a < 0.01 for _t, a in env)


def test_compute_envelope_returns_increasing_timestamps():
    from faceview.speech.tts_kokoro import _compute_envelope
    import numpy as np
    sr = 16_000
    samples = np.random.randn(sr).astype("float32") * 0.5
    env = _compute_envelope(samples, sr, window_ms=30)
    timestamps = [t for t, _a in env]
    assert timestamps == sorted(timestamps)
    # Total duration roughly matches the 1 s clip.
    assert timestamps[-1] >= 0.9


def test_compute_envelope_empty_input_returns_empty():
    from faceview.speech.tts_kokoro import _compute_envelope
    import numpy as np
    assert _compute_envelope(np.array([], dtype="float32"), 16000) == []


def test_emit_envelope_publishes_amplitude_events(fresh_bus, qtbot):
    """The emitter thread should publish AUDIO_AMPLITUDE events in
    order until stopped. We use qtbot to pump the Qt event loop so
    cross-thread bus signals actually deliver."""
    import threading
    from faceview.core.events import EventType
    from faceview.speech.tts_kokoro import _emit_envelope

    received: list = []
    fresh_bus.subscribe(EventType.AUDIO_AMPLITUDE, received.append)

    envelope = [(0.00, 0.1), (0.05, 0.5), (0.10, 0.9)]
    stop = threading.Event()
    t = threading.Thread(
        target=_emit_envelope, args=(envelope, stop), daemon=True,
    )
    t.start()
    # Spin the Qt event loop until all three queued signals deliver.
    qtbot.waitUntil(lambda: len(received) >= 3, timeout=2000)
    t.join(timeout=1.0)
    amps = [r.amplitude for r in received[:3]]
    assert amps == [0.1, 0.5, 0.9]


def test_emit_envelope_respects_stop_flag(fresh_bus):
    """Setting the stop event bails out without emitting the rest."""
    import threading
    from faceview.core.events import EventType
    from faceview.speech.tts_kokoro import _emit_envelope

    received: list = []
    fresh_bus.subscribe(EventType.AUDIO_AMPLITUDE, received.append)

    # Long envelope; stop almost immediately.
    envelope = [(i * 0.5, 0.5) for i in range(10)]
    stop = threading.Event()
    stop.set()  # set up-front
    t = threading.Thread(
        target=_emit_envelope, args=(envelope, stop), daemon=True,
    )
    t.start()
    t.join(timeout=1.0)
    # With stop pre-set, no events fire (the loop checks before each
    # publish).
    assert received == []


# ── U6: theme switcher ─────────────────────────────────────────


def test_load_persisted_default_is_system(qtbot):
    from PySide6.QtCore import QSettings
    from faceview.gui.theme import load_persisted
    # Clear any prior setting.
    QSettings("faceview", "main").remove("theme/mode")
    assert load_persisted() == "system"


def test_apply_theme_persists_choice(qtbot):
    from PySide6.QtCore import QSettings
    from faceview.gui.theme import apply_theme, load_persisted
    apply_theme("dark")
    assert load_persisted() == "dark"
    apply_theme("light")
    assert load_persisted() == "light"
    apply_theme("system")
    assert load_persisted() == "system"


def test_apply_theme_changes_palette(qtbot):
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication
    from faceview.gui.theme import apply_theme
    apply_theme("dark")
    app = QApplication.instance()
    win_color = app.palette().color(QPalette.ColorRole.Window)
    assert win_color.red() < 60 and win_color.blue() < 60  # dark bg
    apply_theme("light")
    win_color = app.palette().color(QPalette.ColorRole.Window)
    assert win_color.red() > 200  # light bg
    # Reset for other tests.
    apply_theme("system")


def test_main_window_set_theme_facade(qtbot):
    from faceview.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    # Should not raise; status bar gets a message.
    w.set_theme("dark")
    w.set_theme("light")
    w.set_theme("system")
    w.set_theme("invalid_value")  # falls back to system, no crash
