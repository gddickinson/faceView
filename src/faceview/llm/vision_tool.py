"""``look_at_camera`` tool — lets the chat bots peek at the webcam.

The chat bots don't ordinarily see camera pixels; the vision stack only
feeds them structured signals (presence / emotion / gaze / gestures /
…) through the system prompt. This module exposes a single function-tool
that either engine can call when it genuinely needs visual context the
ongoing chat doesn't already provide.

Two engines, two paths to the same answer:

* **Anthropic** — the most-recent FRAME (or AVATAR_FRAME) is grabbed off
  the bus, encoded as base-64 JPEG, and attached as an ``image`` content
  block inside the ``tool_result`` message. Claude does its own
  captioning natively, so no extra model is needed.

* **Ollama** — we run a separate local VLM (``moondream`` / ``llava`` /
  ``llama3.2-vision``, whichever is installed) against the JPEG via
  ``/api/generate`` and return its text description. The chat model
  only needs tool-calling support (llama3.1+, qwen2.5, mistral-nemo) —
  it never sees pixels itself.

The grabber subscribes to both :data:`EventType.FRAME` *and*
:data:`EventType.AVATAR_FRAME` so test mode (two bots watching each
other) keeps the cache fresh from the avatar side when the webcam is
off. Heavy deps (``cv2``, ``urllib`` JSON for Ollama) are imported
lazily so this module can be imported without the vision extra.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType
from faceview.core.logger import get_logger


log = get_logger("vision.tool")


# ── frame cache ──────────────────────────────────────────────────────────


class FrameGrabber:
    """Singleton bus subscriber that caches the latest FRAME / AVATAR_FRAME."""

    _instance: "FrameGrabber | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "FrameGrabber":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = FrameGrabber()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = None         # np.ndarray BGR
        self._latest_ts = 0.0
        self._latest_source = ""    # "camera" | "avatar"
        bus = get_bus()
        bus.subscribe(EventType.FRAME, self._on_camera)
        bus.subscribe(EventType.AVATAR_FRAME, self._on_avatar)

    def _on_camera(self, frame) -> None:
        self._update(frame, "camera")

    def _on_avatar(self, frame) -> None:
        # Only fall back to avatar frames if no real camera has shown up
        # in the last 2 s — otherwise the user always wins.
        if frame is None:
            return
        with self._lock:
            recent_camera = (
                self._latest_source == "camera"
                and time.time() - self._latest_ts < 2.0
            )
        if not recent_camera:
            self._update(frame, "avatar")

    def _update(self, frame, source: str) -> None:
        if frame is None:
            return
        with self._lock:
            self._latest = frame
            self._latest_ts = time.time()
            self._latest_source = source

    def have_frame(self) -> bool:
        with self._lock:
            return self._latest is not None

    def latest_jpeg_b64(
        self,
        max_dim: int = 768,
        quality: int = 80,
    ) -> Optional[tuple[str, str]]:
        """Encode the most-recent frame as base64 JPEG.

        Returns ``(b64_str, source)`` or ``None`` if no frame yet or cv2
        is missing.
        """
        with self._lock:
            frame = self._latest
            source = self._latest_source
        if frame is None:
            return None
        try:
            import cv2  # type: ignore
        except ImportError:
            log.warning("vision.tool.no_cv2")
            return None

        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii"), source


# ── tool schemas ─────────────────────────────────────────────────────────


LOOK_TOOL_DESCRIPTION = (
    "Take a single still snapshot from the webcam and look at it. The "
    "GUI already pipes presence, identity, emotion, gaze, gestures, "
    "scene brightness, visible objects, AND a periodic ambient VLM "
    "caption into your system prompt on every turn — call this tool "
    "ONLY when you need a richer or more targeted look than the "
    "ambient caption already gives you (e.g. 'what does the text on "
    "that page say?', 'is the user holding scissors or a pencil?'). "
    "Optional 'question' steers the deeper VLM with a specific query. "
    "Optional 'region' crops to a part of the frame before captioning."
)

_REGION_DESCRIPTION = (
    "Optional crop region — one of: 'full' (default), 'center', "
    "'top', 'bottom', 'left', 'right', 'top_left', 'top_right', "
    "'bottom_left', 'bottom_right'."
)
_QUESTION_DESCRIPTION = (
    "Optional natural-language question to focus the vision model on "
    "a specific aspect (e.g. 'is the person smiling?', 'what colour "
    "is the cup?'). Omit for a general description."
)

LOOK_TOOL_ANTHROPIC: dict = {
    "name": "look_at_camera",
    "description": LOOK_TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string",
                         "description": _QUESTION_DESCRIPTION},
            "region": {"type": "string",
                       "description": _REGION_DESCRIPTION},
        },
    },
}

LOOK_TOOL_OLLAMA: dict = {
    "type": "function",
    "function": {
        "name": "look_at_camera",
        "description": LOOK_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string",
                             "description": _QUESTION_DESCRIPTION},
                "region": {"type": "string",
                           "description": _REGION_DESCRIPTION},
            },
        },
    },
}


# ── region helpers ───────────────────────────────────────────────────────


_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # name → (x_frac, y_frac, w_frac, h_frac)
    "full":          (0.00, 0.00, 1.00, 1.00),
    "center":        (0.25, 0.25, 0.50, 0.50),
    "top":           (0.00, 0.00, 1.00, 0.50),
    "bottom":        (0.00, 0.50, 1.00, 0.50),
    "left":          (0.00, 0.00, 0.50, 1.00),
    "right":         (0.50, 0.00, 0.50, 1.00),
    "top_left":      (0.00, 0.00, 0.50, 0.50),
    "top_right":     (0.50, 0.00, 0.50, 0.50),
    "bottom_left":   (0.00, 0.50, 0.50, 0.50),
    "bottom_right":  (0.50, 0.50, 0.50, 0.50),
}


def _crop_to_region(frame, region: str):
    """Crop a frame in-place to one of the named regions. Falls back
    to ``full`` for unknown labels."""
    spec = _REGIONS.get(
        (region or "full").strip().lower(), _REGIONS["full"]
    )
    if spec == _REGIONS["full"]:
        return frame
    h, w = frame.shape[:2]
    x = int(spec[0] * w)
    y = int(spec[1] * h)
    cw = int(spec[2] * w)
    ch = int(spec[3] * h)
    return frame[y:y + ch, x:x + cw]


def _encode_jpeg(frame, max_dim: int = 768, quality: int = 80):
    """Resize + JPEG-encode → base64. Returns (b64, w, h) or None."""
    import base64 as _b64
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        return None
    return _b64.b64encode(buf.tobytes()).decode("ascii")


# ── additional on-demand tools (OCR / tracking / open-vocab check) ──────


READ_TEXT_TOOL_ANTHROPIC: dict = {
    "name": "read_text",
    "description": (
        "Run OCR on the current webcam frame and read any visible text "
        "(signs, pages, labels, screens). Use when the user asks you "
        "to 'read' something or wants the exact text. Optional "
        "'region' crops to part of the frame first."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "region": {"type": "string",
                       "description": _REGION_DESCRIPTION},
        },
    },
}
READ_TEXT_TOOL_OLLAMA: dict = {
    "type": "function",
    "function": {
        "name": "read_text",
        "description": READ_TEXT_TOOL_ANTHROPIC["description"],
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string",
                           "description": _REGION_DESCRIPTION},
            },
        },
    },
}


TRACK_OBJECT_TOOL_ANTHROPIC: dict = {
    "name": "track_object",
    "description": (
        "Start tracking a named object across the webcam frames for a "
        "short duration so you can later ask 'is it still there?', "
        "'has it moved?', or 'where is it now?'. The object must be "
        "in the currently-visible OBJECTS list. Status is reported "
        "in the perception block of your next turns until the timer "
        "expires."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string",
                      "description": "Object class to track (e.g. 'cup', "
                                     "'person', 'cell phone')."},
            "duration_s": {"type": "number",
                           "description": "How many seconds to track "
                                          "(2-60, default 10)."},
        },
        "required": ["label"],
    },
}
TRACK_OBJECT_TOOL_OLLAMA: dict = {
    "type": "function",
    "function": {
        "name": "track_object",
        "description": TRACK_OBJECT_TOOL_ANTHROPIC["description"],
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "duration_s": {"type": "number"},
            },
            "required": ["label"],
        },
    },
}


CHECK_VISIBLE_TOOL_ANTHROPIC: dict = {
    "name": "check_visible",
    "description": (
        "Ask an open-vocabulary visibility question (CLIP) about the "
        "current frame — e.g. 'a person wearing glasses', 'a coffee "
        "mug', 'a laptop showing code'. Returns yes/no plus a "
        "confidence score. Use this when the structured OBJECTS list "
        "doesn't cover the concept you want to check."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Free-form description of what to "
                                     "look for in the frame."},
            "region": {"type": "string",
                       "description": _REGION_DESCRIPTION},
        },
        "required": ["query"],
    },
}
CHECK_VISIBLE_TOOL_OLLAMA: dict = {
    "type": "function",
    "function": {
        "name": "check_visible",
        "description": CHECK_VISIBLE_TOOL_ANTHROPIC["description"],
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "region": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}


def run_read_text(grabber: FrameGrabber, region: str = "full") -> str:
    """Tool executor for read_text."""
    from faceview.vision.ocr import read_text
    with grabber._lock:
        frame = grabber._latest
    if frame is None:
        return "No camera frame is available right now."
    return read_text(frame, region=region)


def run_track_object(label: str, duration_s: float = 10.0) -> str:
    """Tool executor for track_object."""
    from faceview.vision.tracker import ObjectTracker
    ok, msg = ObjectTracker.shared().start_tracking(
        label, duration_s=duration_s,
    )
    log.info("vision.tool.track_object",
             label=label, duration_s=duration_s, ok=ok)
    return msg


def run_check_visible(
    grabber: FrameGrabber, query: str, region: str = "full",
) -> str:
    """Tool executor for check_visible."""
    from faceview.vision.clip_query import check_visible
    with grabber._lock:
        frame = grabber._latest
    if frame is None:
        return "No camera frame is available right now."
    return check_visible(frame, query, region=region)


# ── Tier 2 + Tier 3 tool schemas ─────────────────────────────────────────


def _basic_region_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "region": {"type": "string",
                       "description": _REGION_DESCRIPTION},
        },
    }


DESCRIBE_COLOR_TOOL_ANTHROPIC: dict = {
    "name": "describe_color",
    "description": ("Identify the dominant colours of a region of the "
                    "frame via k-means. Use for 'what colour is X?' "
                    "questions when you don't need a full VLM call."),
    "input_schema": _basic_region_schema(),
}
DESCRIBE_POSE_TOOL_ANTHROPIC: dict = {
    "name": "describe_pose",
    "description": ("Run MediaPipe Pose on the current frame and "
                    "return a short posture read (sitting / standing, "
                    "leaning, arms crossed, hand raised, …)."),
    "input_schema": {"type": "object", "properties": {}},
}
FACE_ATTRS_TOOL_ANTHROPIC: dict = {
    "name": "face_attributes",
    "description": ("Estimate age + gender of the most prominent face "
                    "via InsightFace. Estimates are rough (±5 years). "
                    "Don't volunteer this unless asked."),
    "input_schema": {"type": "object", "properties": {}},
}
SCAN_QR_TOOL_ANTHROPIC: dict = {
    "name": "scan_qr",
    "description": ("Decode any QR codes in the current frame using "
                    "OpenCV's built-in detector."),
    "input_schema": {"type": "object", "properties": {}},
}
ESTIMATE_DEPTH_TOOL_ANTHROPIC: dict = {
    "name": "estimate_depth",
    "description": ("Run MiDaS-small for monocular depth estimation. "
                    "Returns a coarse near/far summary of the (region "
                    "of the) frame. Slow on first call (~80 MB "
                    "download)."),
    "input_schema": _basic_region_schema(),
}
GAZE_TARGET_TOOL_ANTHROPIC: dict = {
    "name": "gaze_target",
    "description": ("Combine the user's iris direction and head pose "
                    "to label what they're looking at (the camera, "
                    "off-screen, the screen, down, etc). Cheap — no "
                    "new model."),
    "input_schema": {"type": "object", "properties": {}},
}
SEGMENT_OBJECT_TOOL_ANTHROPIC: dict = {
    "name": "segment_object",
    "description": ("Run GrabCut seeded by the EfficientDet bbox of "
                    "the named object to get a foreground mask. "
                    "Returns coverage + centroid zone. Object must be "
                    "in the current OBJECTS list."),
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string",
                      "description": "Object class to segment."},
        },
        "required": ["label"],
    },
}


def _to_ollama(schema: dict) -> dict:
    """Convert an Anthropic-style tool schema to Ollama's tool format."""
    params = schema.get("input_schema") or {"type": "object",
                                            "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": params,
        },
    }


DESCRIBE_COLOR_TOOL_OLLAMA = _to_ollama(DESCRIBE_COLOR_TOOL_ANTHROPIC)
DESCRIBE_POSE_TOOL_OLLAMA = _to_ollama(DESCRIBE_POSE_TOOL_ANTHROPIC)
FACE_ATTRS_TOOL_OLLAMA = _to_ollama(FACE_ATTRS_TOOL_ANTHROPIC)
SCAN_QR_TOOL_OLLAMA = _to_ollama(SCAN_QR_TOOL_ANTHROPIC)
ESTIMATE_DEPTH_TOOL_OLLAMA = _to_ollama(ESTIMATE_DEPTH_TOOL_ANTHROPIC)
GAZE_TARGET_TOOL_OLLAMA = _to_ollama(GAZE_TARGET_TOOL_ANTHROPIC)
SEGMENT_OBJECT_TOOL_OLLAMA = _to_ollama(SEGMENT_OBJECT_TOOL_ANTHROPIC)


# ── Tier 2 + 3 executors ─────────────────────────────────────────────────


def _frame(grabber: FrameGrabber):
    with grabber._lock:
        return grabber._latest


def run_describe_color(grabber: FrameGrabber, region: str = "full") -> str:
    from faceview.vision.color import describe_color
    return describe_color(_frame(grabber), region=region)


def run_describe_pose(grabber: FrameGrabber) -> str:
    from faceview.vision.pose import describe_pose
    return describe_pose(_frame(grabber))


def run_face_attributes(grabber: FrameGrabber) -> str:
    from faceview.vision.face_attr import face_attributes
    return face_attributes(_frame(grabber))


def run_scan_qr(grabber: FrameGrabber) -> str:
    from faceview.vision.qr import scan_qr
    return scan_qr(_frame(grabber))


def run_estimate_depth(grabber: FrameGrabber, region: str = "full") -> str:
    from faceview.vision.depth import estimate_depth
    return estimate_depth(_frame(grabber), region=region)


def run_gaze_target() -> str:
    from faceview.vision.gaze_target import gaze_target
    return gaze_target()


def run_segment_object(grabber: FrameGrabber, label: str) -> str:
    from faceview.vision.segment import segment_object
    return segment_object(_frame(grabber), label)


# Cache of model name → health status so we only probe each candidate
# once per process. Models that return HTTP 500 (typically "no longer
# compatible with your version of Ollama") are demoted permanently.
_VLM_HEALTH: dict[str, bool] = {}


def _vlm_is_healthy(model: str, host: str, timeout: float = 6.0) -> bool:
    """One-shot probe against ``/api/generate`` with a tiny dummy image.

    Caches the result so subsequent calls in the same process are
    free. A model that produces *any* successful response (200) is
    considered healthy; anything else (500, timeout, refused) gets
    demoted so the picker moves to the next candidate.
    """
    if model in _VLM_HEALTH:
        return _VLM_HEALTH[model]
    # 1×1 black PNG, base64-encoded.
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
        "YGD4DwABAQEAtAlnkAAAAABJRU5ErkJggg=="
    )
    body = json.dumps({
        "model": model,
        "prompt": ".",
        "images": [tiny_png],
        "stream": False,
        "options": {"num_predict": 1},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
    except (urllib.error.URLError, ConnectionError, TimeoutError,
            OSError) as exc:
        log.warning("vision.tool.vlm_unhealthy", model=model, error=str(exc))
        ok = False
    _VLM_HEALTH[model] = ok
    if ok:
        log.info("vision.tool.vlm_healthy", model=model)
    return ok


def pick_deep_vision_model(host: str = "http://127.0.0.1:11434") -> Optional[str]:
    """Choose a vision model for *on-demand* look_at_camera calls.

    Preference order favours capability over speed (the inverse of the
    ambient captioner): llama3.2-vision → llava:13b → llava → moondream.
    Override with ``FACEVIEW_OLLAMA_DEEP_VISION_MODEL``. Health-checks
    each candidate once per process so users don't hit stale-model
    HTTP 500s (Ollama versions occasionally invalidate older weights)."""
    env = os.environ.get("FACEVIEW_OLLAMA_DEEP_VISION_MODEL")
    if env:
        # Honour explicit pin even if unhealthy — surface the failure
        # to the user via the tool's own error message rather than
        # silently falling back.
        return env
    try:
        from faceview.llm.ollama_client import list_ollama_models
        models = list_ollama_models(host)
    except Exception:  # noqa: BLE001
        models = []
    if not models:
        return None
    for needle in (
        "llama3.2-vision", "llava:13b", "llava-llama3", "llava:7b",
        "minicpm-v", "llava", "moondream",
    ):
        for m in models:
            if needle not in m.lower():
                continue
            if _vlm_is_healthy(m, host):
                return m
            # Otherwise fall through to next candidate.
    return None


def reset_vlm_health_cache_for_tests() -> None:
    """Drop the health cache — for tests only."""
    _VLM_HEALTH.clear()


# ── executors ────────────────────────────────────────────────────────────


def _signal_pixels(active: bool, destination: str, tool: str = "") -> None:
    """Publish a PIXELS_LEAVING event so the status panel can flash."""
    from faceview.core.event_bus import get_bus
    from faceview.core.events import PixelTransmission
    try:
        get_bus().publish(
            EventType.PIXELS_LEAVING,
            PixelTransmission(active=active, destination=destination,
                              tool=tool),
        )
    except Exception:  # noqa: BLE001 — bus issue must never break a tool
        pass


def run_look_anthropic(
    grabber: FrameGrabber,
    question: str = "",
    region: str = "full",
) -> list[dict]:
    """Build the content-block list for the ``tool_result`` message.

    Anthropic's own vision is used (Claude reads the attached image),
    so ``question`` is only echoed back as text context — Claude
    answers it natively from the pixels. ``region`` crops the frame
    before encoding."""
    with grabber._lock:
        raw = grabber._latest
        source = grabber._latest_source
    if raw is None:
        log.warning("vision.tool.look.no_frame")
        return [{"type": "text",
                 "text": "No camera frame is available right now."}]
    cropped = _crop_to_region(raw, region)
    b64 = _encode_jpeg(cropped)
    if b64 is None:
        return [{"type": "text",
                 "text": "Couldn't encode the current frame."}]
    # Pixels will leave the host on the next Anthropic API call (which
    # the engine performs right after we return). Flash the indicator
    # now — there's no clean "after" hook, so the status panel auto-
    # clears after a few seconds.
    _signal_pixels(True, destination="anthropic", tool="look_at_camera")
    log.info("vision.tool.look.anthropic",
             source=source, region=region, jpeg_bytes=len(b64),
             question=question[:60])
    bits = ["camera snapshot"]
    if source == "avatar":
        bits.append("source: avatar render")
    if region and region != "full":
        bits.append(f"region: {region}")
    if question:
        bits.append(f"focus: {question}")
    note = "(" + "; ".join(bits) + ")"
    return [
        {"type": "image",
         "source": {"type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64}},
        {"type": "text", "text": note},
    ]


def run_look_ollama(
    grabber: FrameGrabber,
    vlm_model: str,
    host: str = "http://127.0.0.1:11434",
    timeout: float = 60.0,
    question: str = "",
    region: str = "full",
) -> str:
    """Run a local VLM over the latest frame; return its text description.

    Honours optional ``question`` (becomes the prompt) and ``region``
    (crops the frame before encoding). Defaults to a general
    description of the full frame."""
    with grabber._lock:
        raw = grabber._latest
    if raw is None:
        return "No camera frame is available right now."
    cropped = _crop_to_region(raw, region)
    b64 = _encode_jpeg(cropped)
    if b64 is None:
        return "Couldn't encode the current frame."
    if question.strip():
        prompt = (
            f"Look at this webcam image and answer this question "
            f"directly and concisely: {question.strip()}"
        )
    else:
        prompt = ("Describe the contents of this webcam image in one "
                  "short paragraph. Focus on people, what they're "
                  "doing, and any visible objects. Be specific but "
                  "concise.")
    log.info("vision.tool.look.ollama_start",
             model=vlm_model, jpeg_bytes=len(b64),
             region=region, question=question[:60])
    _signal_pixels(True, destination=f"ollama:{vlm_model}",
                   tool="look_at_camera")
    body = json.dumps({
        "model": vlm_model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {"num_predict": 220},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        text = (data.get("response") or "").strip()
        if not text:
            log.warning("vision.tool.look.ollama_empty")
            _signal_pixels(False, destination=f"ollama:{vlm_model}",
                           tool="look_at_camera")
            return "(The vision model returned no description.)"
        log.info("vision.tool.look.ollama_done", chars=len(text))
        _signal_pixels(False, destination=f"ollama:{vlm_model}",
                       tool="look_at_camera")
        return text
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError,
            ValueError) as exc:
        log.warning("vision.tool.ollama_failed", error=str(exc))
        _signal_pixels(False, destination=f"ollama:{vlm_model}",
                       tool="look_at_camera")
        return f"(Vision lookup failed: {exc})"


# ── remember_person tool ─────────────────────────────────────────────────


REMEMBER_TOOL_DESCRIPTION = (
    "Save the name of the person currently visible in the webcam so "
    "you (and future sessions) recognise them by face. Call this AFTER "
    "they have told you their name — do NOT guess. The person should "
    "be facing the camera when you call this. Use the person's natural "
    "spelling (e.g. 'Alice', 'Dr. Smith'). Only call when the "
    "Perception block reports an unfamiliar person; do NOT re-save "
    "someone the system already recognises."
)

REMEMBER_TOOL_ANTHROPIC: dict = {
    "name": "remember_person",
    "description": REMEMBER_TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's natural-spelling name.",
            },
        },
        "required": ["name"],
    },
}

REMEMBER_TOOL_OLLAMA: dict = {
    "type": "function",
    "function": {
        "name": "remember_person",
        "description": REMEMBER_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The person's natural-spelling name.",
                },
            },
            "required": ["name"],
        },
    },
}


def run_remember_person(grabber: FrameGrabber, name: str) -> str:
    """Embed the latest frame and save it under ``name``.

    Shared by both Anthropic + Ollama tool loops. Returns a single
    human-readable string for the model to relay back to the user.
    """
    from faceview.vision.people import PeopleStore  # lazy: avoid cycles

    if not name or not name.strip():
        return "I need an actual name to remember someone by."
    pair = grabber.latest_jpeg_b64()
    if pair is None:
        return ("No camera frame is available right now — turn the "
                "camera on first.")
    # We have a recent frame in the cache as raw BGR — fetch it again
    # without re-encoding. The grabber's own `_latest` is the numpy
    # array we need.
    with grabber._lock:
        frame = grabber._latest
    if frame is None:
        return "Camera frame disappeared between snapshot and embed."
    store = PeopleStore.shared()
    ok, msg = store.remember(name.strip(), frame)
    log.info("vision.tool.remember", name=name.strip(), ok=ok,
             roster=store.count())
    return msg


# ── convenience bundles ──────────────────────────────────────────────────
#
# Defined after every tool schema so the lists hold the actual objects
# rather than forward references. Engines just splice these together.

TIER1_TOOLS_ANTHROPIC = [
    LOOK_TOOL_ANTHROPIC, REMEMBER_TOOL_ANTHROPIC,
    READ_TEXT_TOOL_ANTHROPIC, TRACK_OBJECT_TOOL_ANTHROPIC,
    CHECK_VISIBLE_TOOL_ANTHROPIC,
]
TIER23_TOOLS_ANTHROPIC = [
    DESCRIBE_COLOR_TOOL_ANTHROPIC, DESCRIBE_POSE_TOOL_ANTHROPIC,
    FACE_ATTRS_TOOL_ANTHROPIC, SCAN_QR_TOOL_ANTHROPIC,
    ESTIMATE_DEPTH_TOOL_ANTHROPIC, GAZE_TARGET_TOOL_ANTHROPIC,
    SEGMENT_OBJECT_TOOL_ANTHROPIC,
]

TIER1_TOOLS_OLLAMA = [
    LOOK_TOOL_OLLAMA, REMEMBER_TOOL_OLLAMA,
    READ_TEXT_TOOL_OLLAMA, TRACK_OBJECT_TOOL_OLLAMA,
    CHECK_VISIBLE_TOOL_OLLAMA,
]
TIER23_TOOLS_OLLAMA = [
    DESCRIBE_COLOR_TOOL_OLLAMA, DESCRIBE_POSE_TOOL_OLLAMA,
    FACE_ATTRS_TOOL_OLLAMA, SCAN_QR_TOOL_OLLAMA,
    ESTIMATE_DEPTH_TOOL_OLLAMA, GAZE_TARGET_TOOL_OLLAMA,
    SEGMENT_OBJECT_TOOL_OLLAMA,
]


# ── toggle ───────────────────────────────────────────────────────────────


def vision_tool_enabled() -> bool:
    """Master switch: ``FACEVIEW_VISION_TOOL=0`` disables tool injection."""
    raw = os.environ.get("FACEVIEW_VISION_TOOL")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}
