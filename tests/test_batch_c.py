"""C7 + P7 + A47 wire-up — shared cross-persona facts, frame buffer,
audio-driven jaw bias."""

from __future__ import annotations


# ── C7 — shared cross-persona facts ────────────────────────────


def _store(tmp_path, monkeypatch, persona="p1"):
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    cog.CognitionStore.set_incognito(False)
    return cog.CognitionStore(persona)


def test_share_fact_writes_to_shared_file(tmp_path, monkeypatch):
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    store = cog.CognitionStore("first")
    store.share_fact("player", "name", "George", confidence=1.0)
    shared_path = tmp_path / "memory" / "_shared.json"
    assert shared_path.exists()
    import json
    data = json.loads(shared_path.read_text())
    assert data["player"]["name"]["value"] == "George"


def test_name_propagates_across_personas(tmp_path, monkeypatch):
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    cog.CognitionStore.set_incognito(False)
    # First persona learns the name.
    a = cog.CognitionStore("persona_a")
    a.record_chat_turn("My name is George", "Hi George.")
    assert a.get_fact("player", "name") == "George"
    # Second persona, brand-new store, should see the name through
    # narrate_for_prompt (shared facts get folded into semantic).
    b = cog.CognitionStore("persona_b")
    assert b.get_fact("player", "name") is None  # not on disk yet
    out = b.narrate_for_prompt()
    # After narrate the local cache should hold the shared fact.
    assert b.get_fact("player", "name") == "George"
    assert "George" in out


def test_share_fact_also_sets_local_fact(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.share_fact("player", "loves", "dark roast", confidence=0.9)
    assert store.get_fact("player", "loves") == "dark roast"


# ── P7 — rolling frame buffer ──────────────────────────────────


def test_frame_buffer_default_singleton(fresh_bus, monkeypatch):
    import faceview.vision.frame_buffer as fb
    monkeypatch.setattr(fb.FrameBuffer, "_instance", None)
    a = fb.FrameBuffer.shared()
    b = fb.FrameBuffer.shared()
    assert a is b


def test_frame_buffer_caches_recent_frames(fresh_bus, monkeypatch):
    import numpy as np
    import faceview.vision.frame_buffer as fb
    monkeypatch.setattr(fb.FrameBuffer, "_instance", None)
    buf = fb.FrameBuffer.shared()
    from faceview.core.events import EventType
    for i in range(5):
        fresh_bus.publish(
            EventType.FRAME,
            np.full((10, 10, 3), i, dtype="uint8"),
        )
    assert buf.count() == 5


def test_frame_buffer_drops_old_frames_by_time(fresh_bus, monkeypatch):
    """A frame timestamped well in the past is evicted on the next
    push (the time-based cutoff kicks in)."""
    import numpy as np
    import time as _time
    import faceview.vision.frame_buffer as fb
    monkeypatch.setattr(fb.FrameBuffer, "_instance", None)
    buf = fb.FrameBuffer(seconds=1.0)
    fb.FrameBuffer._instance = buf
    f0 = np.zeros((5, 5, 3), dtype="uint8")
    # Stale push.
    buf.push(f0, ts=_time.time() - 10)
    # Fresh push.
    buf.push(np.ones((5, 5, 3), dtype="uint8"))
    # The 10-s-old frame should have been evicted.
    assert buf.count() == 1


def test_frame_buffer_byte_ceiling(monkeypatch):
    """A burst of large frames is bounded by max_bytes."""
    import numpy as np
    import faceview.vision.frame_buffer as fb
    monkeypatch.setattr(fb.FrameBuffer, "_instance", None)
    # 1 MB per frame; cap at 3 MB.
    buf = fb.FrameBuffer(
        seconds=60.0, max_frames=100, max_bytes=3_000_000,
    )
    fb.FrameBuffer._instance = buf
    for _ in range(10):
        buf.push((np.zeros((512, 512, 3), dtype="uint8") + 1))
    # 768 KB per frame; ~4 frames worth fits in 3 MB.
    assert buf.byte_size() <= 3_000_000
    assert buf.count() <= 5


def test_clip_last_returns_recent_frames(monkeypatch):
    import numpy as np
    import time as _time
    import faceview.vision.frame_buffer as fb
    monkeypatch.setattr(fb.FrameBuffer, "_instance", None)
    buf = fb.FrameBuffer(seconds=10.0)
    fb.FrameBuffer._instance = buf
    now = _time.time()
    buf.push(np.zeros((4, 4, 3), dtype="uint8"), ts=now - 8.0)
    buf.push(np.ones((4, 4, 3), dtype="uint8"), ts=now - 1.0)
    recent = buf.clip_last(seconds=2.0)
    assert len(recent) == 1


# ── A47 — audio-driven jaw bias ────────────────────────────────


def test_audio_jaw_bias_starts_at_zero(qtbot):
    from faceview.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.avatar_ctrl.audio_jaw_bias() == 0.0


def test_audio_amplitude_event_updates_bias(qtbot, fresh_bus):
    from faceview.core.events import AudioAmplitude, EventType
    from faceview.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    fresh_bus.publish(EventType.AUDIO_AMPLITUDE,
                      AudioAmplitude(amplitude=0.6))
    # Bias is smoothed (0.6 × 0.4 weight on first sample).
    assert w.avatar_ctrl.audio_jaw_bias() > 0.0
    # A few more samples → converges toward the input.
    for _ in range(10):
        fresh_bus.publish(EventType.AUDIO_AMPLITUDE,
                          AudioAmplitude(amplitude=0.8))
    assert w.avatar_ctrl.audio_jaw_bias() > 0.7


def test_audio_jaw_bias_decays_to_zero_when_stale(qtbot, fresh_bus):
    import time as _time
    from faceview.core.events import AudioAmplitude, EventType
    from faceview.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    fresh_bus.publish(EventType.AUDIO_AMPLITUDE,
                      AudioAmplitude(amplitude=0.9))
    # Back-date the receipt to make the signal stale.
    w.avatar_ctrl._audio_amp_ts = _time.time() - 5.0
    assert w.avatar_ctrl.audio_jaw_bias() == 0.0
