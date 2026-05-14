"""Kokoro neural-TTS engine.

Local ONNX model that runs in real time on Apple Silicon CPU.
Selected by :class:`TtsWorker` when the kokoro-onnx package is
installed and the model files exist on disk.

Model location: ``data_dir() / "tts" / kokoro-v1.0.onnx`` plus
``voices-v1.0.bin``. We don't auto-download — the files are large and
download semantics belong somewhere the user can monitor. Provide a
helper :func:`download_kokoro_assets` for opt-in pulling.

Playback goes through ``afplay`` rather than ``sounddevice.play``
because the latter conflicts with the mic-capture ``InputStream``
that's typically also open — the result is loud digital noise.
``afplay`` is a stock macOS tool, opens its own audio session, and
just works.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import AudioAmplitude, EventType
from faceview.core.logger import get_logger
from faceview.utils.paths import data_dir


log = get_logger("tts.kokoro")


MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)

DEFAULT_VOICE = "af_sarah"   # most natural-feeling all-rounder
DEFAULT_SPEED = 1.0
DEFAULT_LANG = "en-us"


def assets_dir() -> Path:
    d = data_dir() / "tts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path() -> Path:
    return assets_dir() / "kokoro-v1.0.onnx"


def voices_path() -> Path:
    return assets_dir() / "voices-v1.0.bin"


def assets_present() -> bool:
    return model_path().exists() and voices_path().exists()


def download_kokoro_assets() -> tuple[bool, str]:
    """Pull model + voices into ``assets_dir()``. Returns (ok, message).

    No progress callback; the model is ~310 MB and the voices ~27 MB,
    typical download time on a home connection is well under a minute.
    """
    d = assets_dir()
    try:
        if not model_path().exists():
            log.info("kokoro.download", target="model", url=MODEL_URL)
            urllib.request.urlretrieve(MODEL_URL, model_path())
        if not voices_path().exists():
            log.info("kokoro.download", target="voices", url=VOICES_URL)
            urllib.request.urlretrieve(VOICES_URL, voices_path())
    except (urllib.error.URLError, OSError) as exc:
        return False, str(exc)
    return True, f"assets ready at {d}"


class KokoroEngine:
    """Lazy-loaded kokoro-onnx synthesizer + sounddevice playback.

    Thread-safe at the worker-thread level: ``TtsWorker`` serialises
    calls via its own queue, so we don't need internal locking around
    ``create()``. The engine is initialised on first call, not on
    construction, so missing dependencies fail cleanly only when TTS
    is actually requested.
    """

    def __init__(self, *,
                 voice: str = DEFAULT_VOICE,
                 speed: float = DEFAULT_SPEED,
                 lang: str = DEFAULT_LANG) -> None:
        self.voice = voice
        self.speed = float(speed)
        self.lang = lang
        self._engine = None
        self._sd = None
        self._lock = threading.Lock()
        # Handle to the active afplay subprocess so an interrupt
        # request from the GUI can kill the current utterance mid-play.
        self._proc: Optional[subprocess.Popen] = None

    # ── lifecycle ─────────────────────────────────────────────

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from kokoro_onnx import Kokoro  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "kokoro-onnx not installed (pip install kokoro-onnx soundfile)"
                ) from exc
            if not assets_present():
                raise RuntimeError(
                    "kokoro model/voices missing — run "
                    "`python -m faceview.speech.tts_kokoro --download` "
                    f"or place files in {assets_dir()}"
                )
            t0 = time.time()
            self._engine = Kokoro(str(model_path()), str(voices_path()))
            log.info("kokoro.ready", init_s=round(time.time() - t0, 2))
        return self._engine

    def _ensure_writer(self):
        if self._sd is not None:
            return self._sd
        try:
            import soundfile as sf  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "soundfile not installed (pip install soundfile)"
            ) from exc
        self._sd = sf  # name kept for legacy compatibility
        return sf

    # ── public ────────────────────────────────────────────────

    def voices(self) -> list[str]:
        try:
            return sorted(self._ensure_engine().get_voices())
        except Exception:  # noqa: BLE001
            return []

    def speak(self, text: str) -> Optional[float]:
        """Synth + play synchronously. Returns audio duration in seconds."""
        text = (text or "").strip()
        if not text:
            return None
        engine = self._ensure_engine()
        sf = self._ensure_writer()
        t0 = time.time()
        samples, sr = engine.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang,
        )
        synth_s = time.time() - t0
        duration = len(samples) / float(sr) if sr else 0.0
        log.info("kokoro.speak", chars=len(text),
                 synth_s=round(synth_s, 2),
                 duration_s=round(duration, 2),
                 voice=self.voice)
        # Write a temp WAV and play via macOS `afplay`. Avoids
        # sounddevice conflicting with the mic-capture InputStream.
        # Track the subprocess so push-to-talk can interrupt mid-play.
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="faceview_tts_")
        os.close(fd)
        try:
            sf.write(path, samples, int(sr))
            # A47 — compute amplitude envelope ahead of time so we can
            # emit AUDIO_AMPLITUDE events in lock-step with playback.
            envelope = _compute_envelope(samples, int(sr))
            proc = subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._proc = proc
            stop_evt = threading.Event()
            env_thread = threading.Thread(
                target=_emit_envelope,
                args=(envelope, stop_evt),
                name="kokoro-envelope",
                daemon=True,
            )
            env_thread.start()
            try:
                proc.wait()
            finally:
                self._proc = None
                stop_evt.set()
                # Final zero sample so avatar closes the mouth.
                try:
                    get_bus().publish(
                        EventType.AUDIO_AMPLITUDE,
                        AudioAmplitude(amplitude=0.0),
                    )
                except Exception:  # noqa: BLE001
                    pass
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return duration

    def stop(self) -> None:
        """Interrupt the current utterance, if any."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    def set_voice(self, voice: str) -> None:
        self.voice = voice

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.5, min(2.0, float(speed)))


# ── envelope helpers (A47) ────────────────────────────────────────


def _compute_envelope(
    samples, sample_rate: int, window_ms: float = 30.0,
) -> list[tuple[float, float]]:
    """Return ``[(t_seconds, amplitude_0_to_1), ...]`` chunks of the
    audio. Cheap RMS in fixed-size windows; no external deps."""
    try:
        import numpy as np
    except ImportError:
        return []
    if sample_rate <= 0:
        return []
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return []
    win = max(1, int(sample_rate * window_ms / 1000.0))
    out: list[tuple[float, float]] = []
    # RMS per window.
    n = arr.size
    rms_max = 1e-6
    rmss: list[float] = []
    for i in range(0, n, win):
        chunk = arr[i:i + win]
        rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        rms_max = max(rms_max, rms)
        rmss.append(rms)
    # Normalise to [0, 1] using the loudest window so quieter
    # utterances still open the mouth properly.
    for i, r in enumerate(rmss):
        t = (i * win) / float(sample_rate)
        out.append((t, min(1.0, r / rms_max)))
    return out


def _emit_envelope(
    envelope: list[tuple[float, float]], stop_evt: threading.Event,
) -> None:
    """Walk the envelope and publish AudioAmplitude events at the
    right wall-clock times. Stops cleanly when ``stop_evt`` fires
    (i.e. the user interrupted TTS) or the list is exhausted."""
    if not envelope:
        return
    t0 = time.time()
    for t_target, amp in envelope:
        if stop_evt.is_set():
            return
        wait = (t0 + t_target) - time.time()
        if wait > 0:
            if stop_evt.wait(timeout=min(wait, 0.5)):
                return
        try:
            get_bus().publish(
                EventType.AUDIO_AMPLITUDE,
                AudioAmplitude(amplitude=float(amp)),
            )
        except Exception:  # noqa: BLE001
            return


# ── CLI entry: `python -m faceview.speech.tts_kokoro --download` ─────


def _main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Kokoro TTS asset utility")
    p.add_argument("--download", action="store_true",
                   help="Fetch model + voices into the data dir")
    p.add_argument("--say", default=None,
                   help="Speak the given text (for quick verification)")
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    args = p.parse_args()
    if args.download:
        ok, msg = download_kokoro_assets()
        print(("ok: " if ok else "fail: ") + msg)
        return 0 if ok else 1
    if args.say:
        eng = KokoroEngine(voice=args.voice, speed=args.speed)
        eng.speak(args.say)
        return 0
    print(f"assets dir: {assets_dir()}")
    print(f"model present: {model_path().exists()}")
    print(f"voices present: {voices_path().exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
