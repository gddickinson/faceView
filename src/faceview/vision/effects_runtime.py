"""Effects runtime — schedule + apply active effects per frame.

Tracks transient effect instances (each with start time, duration,
intensity) and persistent slider state (skin hue, eye-glow color,
head morph blends, …). Per frame:

1. Sliders pre-modify FaceParams (always-on adjustments).
2. Active PreFX run on FaceParams.
3. Renderer renders.
4. Active PostFX run on the rendered BGR.
5. Expired effects are pruned.

The runtime is thread-safe for the public mutator methods so the
GUI / HTTP API can trigger from one thread while the camera worker
ticks from another.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from faceview.vision.effects import (
    POST_HANDLERS, PRE_HANDLERS, REGISTRY, Stage, get_spec,
)


@dataclass
class _ActiveEffect:
    name: str
    start_t: float
    duration: float
    intensity: float


@dataclass
class SliderState:
    """Persistent (always-on) tweaks applied every frame."""
    # Skin / eye tone overrides — None means "use persona default".
    skin_hue: Optional[float] = None         # 0..360
    skin_saturation: Optional[float] = None  # 0..1
    skin_value: Optional[float] = None       # 0..1
    eye_color_hex: Optional[str] = None      # "#rrggbb"
    bloom_amp: Optional[float] = None        # 0..1
    emit_pulse_scale: Optional[float] = None # 0..3
    # Shape morphs (multiplied with the per-frame AU).
    eye_open_bias: float = 0.0      # -0.5..0.5
    smile_bias: float = 0.0          # -0.5..0.5
    brow_raise_bias: float = 0.0     # -0.5..0.5
    head_pitch_bias: float = 0.0     # -0.3..0.3
    head_yaw_bias: float = 0.0       # -0.3..0.3
    # Identity-blend slider (mixed into params.identity_weights).
    head_age: float = 0.0    # -1 (young) ..0..+1 (elder)
    head_gender: float = 0.0 # -1 (female) ..0..+1 (male)
    # Direct blendshape sliders — drive ICT shapes that don't fit the
    # AU vocabulary. All are 0..1 unless noted.
    pupil_dilate: float = 0.0     # 0 = constricted, 1 = dilated
    jaw_forward: float = 0.0      # protrude jaw forward
    jaw_left_right: float = 0.0   # -1 = jaw left, +1 = jaw right
    mouth_left_right: float = 0.0 # -1 = mouth left, +1 = mouth right
    mouth_funnel: float = 0.0     # forward funnel (lips horn-shape)
    mouth_close: float = 0.0      # active lip-close (separate from pucker)
    cheek_puff: float = 0.0       # both cheeks puffed (holding breath)
    # Hair overlay — procedural style + colour.
    hair_style: str = "none"      # see hair_overlay.STYLES
    hair_color: str = "#3a2418"   # hex string


class EffectsRuntime:
    """Tracks active effects + slider state. Apply hooks per frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: list[_ActiveEffect] = []
        self.sliders = SliderState()

    # ── triggering ─────────────────────────────────────────────

    def trigger(self, name: str, *,
                 intensity: float = 1.0,
                 duration: Optional[float] = None) -> bool:
        """Schedule an effect. Returns True if known."""
        spec = get_spec(name)
        if spec is None:
            return False
        with self._lock:
            self._active.append(_ActiveEffect(
                name=name,
                start_t=time.monotonic(),
                duration=float(duration if duration is not None
                                  else spec.default_duration),
                intensity=float(max(0.0, min(1.0, intensity))),
            ))
        return True

    def stop(self, name: str) -> int:
        """Stop all instances of `name`. Returns how many were active."""
        with self._lock:
            before = len(self._active)
            self._active = [e for e in self._active if e.name != name]
            return before - len(self._active)

    def stop_all(self) -> int:
        with self._lock:
            n = len(self._active)
            self._active.clear()
            return n

    def list_active(self) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "name": e.name,
                    "intensity": e.intensity,
                    "elapsed": max(0.0, now - e.start_t),
                    "remaining": max(0.0, e.duration - (now - e.start_t)),
                }
                for e in self._active
            ]

    def list_specs(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "category": s.category,
                "stage": s.stage.value,
                "label": s.label,
                "description": s.description,
                "default_duration": s.default_duration,
            } for s in REGISTRY
        ]

    # ── slider mutators ────────────────────────────────────────

    def set_slider(self, key: str, value) -> bool:
        if not hasattr(self.sliders, key):
            return False
        with self._lock:
            try:
                cur = getattr(self.sliders, key)
                if isinstance(cur, str) or cur is None and isinstance(value, str):
                    setattr(self.sliders, key, str(value))
                else:
                    setattr(self.sliders, key, float(value))
            except (TypeError, ValueError):
                return False
        return True

    def get_sliders(self) -> dict:
        with self._lock:
            return {k: getattr(self.sliders, k)
                    for k in self.sliders.__dataclass_fields__}

    def reset_sliders(self) -> None:
        with self._lock:
            self.sliders = SliderState()

    # ── per-frame hooks ────────────────────────────────────────

    def apply_pre(self, params) -> None:
        """Mutate FaceParams: sliders first, then active PreFX."""
        s = self.sliders
        # Persistent slider biases — set _slider_* attributes that the
        # renderer reads directly (these override the persona's natural
        # palette / pulse / bloom even in xray and other sci-fi modes).
        if s.skin_hue is not None:
            params._slider_skin_hue = float(s.skin_hue)
            params.skin_hue = float(s.skin_hue)
        if s.skin_saturation is not None:
            params._slider_skin_sat = float(s.skin_saturation)
            params._persona_skin_sat = float(s.skin_saturation)
        if s.skin_value is not None:
            params._slider_skin_val = float(s.skin_value)
            params._persona_skin_val = float(s.skin_value)
        if s.eye_color_hex:
            params._persona_eye_color = s.eye_color_hex
        if s.bloom_amp is not None:
            params._slider_bloom_amp = float(s.bloom_amp)
        if s.emit_pulse_scale is not None:
            params._slider_emit_pulse_scale = float(s.emit_pulse_scale)
        # Eye-open bias — extend clip range so the slider can fully
        # close eyes (down to 0) or wide-open them past the natural
        # 1.05 cap from face_state_to_params.
        params.eye_open = max(0.0, min(1.4, getattr(params, "eye_open", 1.0)
                                            + s.eye_open_bias))
        params.smile = max(-1.0, min(1.0, getattr(params, "smile", 0.0)
                                            + s.smile_bias))
        params.brow_raise = max(-1.0, min(1.0,
                                            getattr(params, "brow_raise", 0.0)
                                            + s.brow_raise_bias))
        params.pitch = float(getattr(params, "pitch", 0.0)) + s.head_pitch_bias
        params.yaw = float(getattr(params, "yaw", 0.0)) + s.head_yaw_bias

        # Identity-weight nudges from age / gender sliders. Blends
        # additively into params.identity_weights without erasing
        # the persona's base values.
        iw = dict(getattr(params, "identity_weights", None) or {})
        if abs(s.head_age) > 1e-3:
            # Identity004 tracks "elder" in the ICT PCA basis — push +
            # for elder, - for young.
            iw["identity004"] = float(iw.get("identity004", 0.0)
                                          + s.head_age * 2.0)
            iw["identity009"] = float(iw.get("identity009", 0.0)
                                          - s.head_age * 1.5)
        if abs(s.head_gender) > 1e-3:
            # Identity000 tracks gender (positive = male).
            iw["identity000"] = float(iw.get("identity000", 0.0)
                                          + s.head_gender * 2.0)
        if iw:
            params.identity_weights = iw

        # Direct blendshape sliders — populate params.direct_blendshapes
        # so the renderer feeds them into the ICT coefficient stream.
        direct = dict(getattr(params, "direct_blendshapes", None) or {})
        if abs(s.pupil_dilate) > 1e-3:
            v = float(np.clip(s.pupil_dilate, 0.0, 1.0))
            direct["PupilDilate_L"] = v
            direct["PupilDilate_R"] = v
        if abs(s.jaw_forward) > 1e-3:
            direct["jawForward"] = float(np.clip(s.jaw_forward, 0.0, 1.0))
        if abs(s.jaw_left_right) > 1e-3:
            v = float(np.clip(s.jaw_left_right, -1.0, 1.0))
            if v > 0:
                direct["jawRight"] = v
            else:
                direct["jawLeft"] = -v
        if abs(s.mouth_left_right) > 1e-3:
            v = float(np.clip(s.mouth_left_right, -1.0, 1.0))
            if v > 0:
                direct["mouthRight"] = v
            else:
                direct["mouthLeft"] = -v
        if abs(s.mouth_funnel) > 1e-3:
            direct["mouthFunnel"] = float(np.clip(s.mouth_funnel, 0.0, 1.0))
        if abs(s.mouth_close) > 1e-3:
            direct["mouthClose"] = float(np.clip(s.mouth_close, 0.0, 1.0))
        if abs(s.cheek_puff) > 1e-3:
            v = float(np.clip(s.cheek_puff, 0.0, 1.0))
            direct["cheekPuff_L"] = v
            direct["cheekPuff_R"] = v
        if direct:
            params.direct_blendshapes = direct

        # Active PreFX.
        with self._lock:
            actives = list(self._active)
        now = time.monotonic()
        for e in actives:
            handler = PRE_HANDLERS.get(e.name)
            if handler is None:
                continue
            elapsed = now - e.start_t
            if elapsed > e.duration:
                continue
            u = elapsed / max(e.duration, 1e-6)
            try:
                handler(params, u, e.intensity)
            except Exception:
                pass

    def _apply_hair(self, bgr: np.ndarray,
                      feature_pixels: dict | None) -> np.ndarray:
        """Composite the chosen hair style on top of the rendered face.

        Detects the head bbox from the foreground and uses the
        forehead feature pixel for hairline placement. No-op when
        ``hair_style == "none"``.
        """
        if self.sliders.hair_style == "none":
            return bgr
        try:
            from faceview.vision.hair_overlay import apply_hair
        except Exception:
            return bgr
        # Head bbox via foreground pixels (cheap luma threshold).
        luma = bgr.max(axis=2)
        ys, xs = np.where(luma > 30)
        if not len(xs):
            return bgr
        head_bbox = (int(xs.min()), int(ys.min()),
                       int(xs.max()), int(ys.max()))
        forehead_y = None
        if feature_pixels and "forehead" in feature_pixels:
            forehead_y = int(feature_pixels["forehead"][1])
        return apply_hair(bgr, self.sliders.hair_style,
                            self.sliders.hair_color, head_bbox, forehead_y)


    def apply_post(self, bgr: np.ndarray, *,
                    feature_pixels: Optional[dict] = None) -> np.ndarray:
        """Apply all active PostFX to the rendered BGR.

        ``feature_pixels`` is a dict of named (x, y) anchors on the
        face (eye_L, eye_R, mouth, cheek_L/R, forehead, chin, …)
        produced by the renderer. Effects that need to overlay on a
        specific feature read it from there; effects that don't care
        ignore the kwarg.
        """
        feat = feature_pixels or {}
        # Apply persistent hair overlay first so PostFX (smoke,
        # sparkles, glitch, scanlines) all draw over the hair too.
        bgr = self._apply_hair(bgr, feat)
        with self._lock:
            actives = list(self._active)
            now = time.monotonic()
            self._active = [e for e in self._active
                              if now - e.start_t <= e.duration]
        for e in actives:
            handler = POST_HANDLERS.get(e.name)
            if handler is None:
                continue
            elapsed = time.monotonic() - e.start_t
            if elapsed > e.duration:
                continue
            u = elapsed / max(e.duration, 1e-6)
            try:
                # Try kwargs-aware handler first, fall back to legacy
                # 3-arg signature.
                try:
                    bgr = handler(bgr, u, e.intensity, features=feat)
                except TypeError:
                    bgr = handler(bgr, u, e.intensity)
            except Exception:
                pass
        return bgr


_GLOBAL: Optional[EffectsRuntime] = None


def get_runtime() -> EffectsRuntime:
    """Module-level singleton — shared across threads."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = EffectsRuntime()
    return _GLOBAL
