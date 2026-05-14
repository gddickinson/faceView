"""Live perception aggregator + LLM-context narrator.

Subscribes to every structured vision signal on the bus and caches the
most-recent value for each. Two outputs:

* :meth:`narrate_now` — a one-paragraph plain-text status block,
  designed to be prepended to the LLM system prompt every turn via
  :meth:`faceview.llm.conversation.Conversation.add_system_extras_provider`.
* :meth:`snapshot_dict` — a structured dict the debug panel and the
  HTTP ``/perception`` endpoint render directly.

The aggregator is **always safe to construct** — if no vision workers
are running it simply returns the empty string from :meth:`narrate_now`
and the LLM behaves exactly as before this module existed.

Singleton: there's one process-wide :class:`PerceptionStore` so
multiple subscribers (LLM + GUI panel) share the same cache.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import (
    Blink, Emotion, EventType, FaceDistance, Gaze, Gesture, HeadPose,
    Identity, MouthActivity, ObjectsSeen, Presence, SceneCaption,
    SceneInfo,
)


# Stale-after window for each signal. If the cached value is older than
# this many seconds, narrate_now treats it as absent.
_STALE_AFTER = 4.0


class PerceptionStore:
    """Caches the latest of each vision/perception signal."""

    _instance: "PerceptionStore | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "PerceptionStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = PerceptionStore()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.presence: Optional[Presence] = None
        self.identity: Optional[Identity] = None
        self.emotion: Optional[Emotion] = None
        self.mouth: Optional[MouthActivity] = None
        self.head_pose: Optional[HeadPose] = None
        self.gaze: Optional[Gaze] = None
        self.distance: Optional[FaceDistance] = None
        self.blink: Optional[Blink] = None
        self.gesture: Optional[Gesture] = None
        self.scene: Optional[SceneInfo] = None
        self.scene_caption: Optional[SceneCaption] = None
        self.objects: Optional[ObjectsSeen] = None
        # Timestamp the current "stranger" run started — used so the
        # LLM is only nudged to ask for a name once a stranger has
        # been steadily visible for a few seconds (avoids reacting to
        # a single bad frame). Reset whenever identity flips to a
        # known person or the face disappears.
        self._stranger_since: Optional[float] = None

        bus = get_bus()
        bus.subscribe(EventType.PRESENCE,
                      lambda p: self._set("presence", p))
        bus.subscribe(EventType.IDENTITY,
                      lambda p: self._on_identity(p))
        bus.subscribe(EventType.EMOTION,
                      lambda p: self._set("emotion", p))
        bus.subscribe(EventType.MOUTH_ACTIVITY,
                      lambda p: self._set("mouth", p))
        bus.subscribe(EventType.HEAD_POSE,
                      lambda p: self._set("head_pose", p))
        bus.subscribe(EventType.GAZE,
                      lambda p: self._set("gaze", p))
        bus.subscribe(EventType.FACE_DISTANCE,
                      lambda p: self._set("distance", p))
        bus.subscribe(EventType.BLINK,
                      lambda p: self._set("blink", p))
        bus.subscribe(EventType.GESTURE,
                      lambda p: self._set("gesture", p))
        bus.subscribe(EventType.SCENE,
                      lambda p: self._set("scene", p))
        bus.subscribe(EventType.SCENE_CAPTION,
                      lambda p: self._set("scene_caption", p))
        bus.subscribe(EventType.OBJECTS,
                      lambda p: self._set("objects", p))

    def _set(self, name: str, payload: Any) -> None:
        if payload is None:
            return
        with self._lock:
            setattr(self, name, payload)

    def _on_identity(self, payload: Identity) -> None:
        """Identity-specific path: track the stranger-since timer too."""
        if payload is None:
            return
        with self._lock:
            self.identity = payload
            if payload.label == "stranger":
                if self._stranger_since is None:
                    self._stranger_since = time.time()
            else:
                self._stranger_since = None

    # ── readers ─────────────────────────────────────────────────────────

    def snapshot_dict(self) -> dict:
        """Structured snapshot, freshness-marked, JSON-safe."""
        with self._lock:
            vals = {
                "presence": self.presence,
                "identity": self.identity,
                "emotion": self.emotion,
                "mouth": self.mouth,
                "head_pose": self.head_pose,
                "gaze": self.gaze,
                "distance": self.distance,
                "blink": self.blink,
                "gesture": self.gesture,
                "scene": self.scene,
                "scene_caption": self.scene_caption,
                "objects": self.objects,
            }
        now = time.time()
        out: dict[str, Any] = {}
        for k, v in vals.items():
            if v is None:
                out[k] = None
                continue
            d = asdict(v) if is_dataclass(v) else dict(v)
            ts = float(d.get("ts", now))
            # Scene captions can be a minute or two old and still be
            # genuinely useful context — give them a longer fresh window.
            stale_after = 90.0 if k == "scene_caption" else _STALE_AFTER
            d["fresh"] = (now - ts) <= stale_after
            out[k] = d
        return out

    def narrate_now(self) -> str:
        """One-paragraph live status for the LLM system prompt.

        Returns ``""`` when no signals are present so adding the
        provider is always safe (the engine just sees the original
        system prompt)."""
        snap = self.snapshot_dict()
        bits: list[str] = []

        presence = _fresh(snap, "presence")
        identity = _fresh(snap, "identity")
        if presence:
            n = int(presence.get("face_count") or 0)
            if n == 0:
                bits.append("no face is visible in the camera")
            elif n == 1:
                who = ""
                if identity:
                    label = identity.get("label", "stranger")
                    sim = identity.get("similarity", 0.0)
                    if identity.get("is_owner"):
                        who = f" (recognised as the owner, sim {sim:.2f})"
                    elif label and label != "stranger":
                        who = (f" (recognised as {label}, sim {sim:.2f})")
                    else:
                        who = " (face not recognised)"
                bits.append(f"one face is visible{who}")
            else:
                bits.append(f"{n} faces are visible")

        # Stranger-since nudge — only fires after a steady few seconds
        # so we don't pester the user from a single misdetection. This
        # is the cue the LLM watches for: ask the name, then call
        # ``remember_person(name=…)``.
        since = self._stranger_since
        if (since is not None
                and identity
                and identity.get("label") == "stranger"
                and presence
                and int(presence.get("face_count") or 0) >= 1):
            elapsed = time.time() - since
            if elapsed >= 2.0:
                bits.append(
                    f"an unfamiliar person has been visible for "
                    f"{elapsed:.0f} s — you don't know their name yet; "
                    "please ask politely, then call remember_person "
                    "with what they tell you"
                )

        emo = _fresh(snap, "emotion")
        if emo:
            bits.append(f"expression: {emo.get('label','neutral')} "
                        f"({emo.get('confidence', 0.0):.0%})")

        mouth = _fresh(snap, "mouth")
        if mouth:
            if mouth.get("speaking"):
                v = mouth.get("viseme") or "speaking"
                bits.append(f"mouth: {v}")
            else:
                bits.append("mouth: closed")

        gaze = _fresh(snap, "gaze")
        if gaze:
            d = gaze.get("direction", "away")
            if d == "camera":
                bits.append("gaze: looking at the camera")
            elif d == "away":
                bits.append("gaze: looking away")
            else:
                bits.append(f"gaze: looking {d}")

        head = _fresh(snap, "head_pose")
        if head:
            label = _head_pose_label(head)
            if label:
                bits.append(f"head: {label}")

        dist = _fresh(snap, "distance")
        if dist:
            bits.append(f"distance: {dist.get('label','normal')}")

        blink = _fresh(snap, "blink")
        if blink:
            state = blink.get("state", "open")
            rate = blink.get("rate_per_min", 0.0)
            if state == "drowsy":
                bits.append(f"eyes: drowsy ({rate:.0f}/min)")
            elif state == "closed":
                bits.append("eyes: closed")

        gesture = _fresh(snap, "gesture")
        if gesture and gesture.get("label") not in (None, "none", ""):
            hand = gesture.get("hand", "")
            label = gesture.get("label")
            if hand and hand != "none":
                bits.append(f"gesture: {hand} hand {label}")
            else:
                bits.append(f"gesture: {label}")

        scene = _fresh(snap, "scene")
        if scene:
            bits.append(
                f"scene: {scene.get('brightness_label','lit')}, "
                f"{scene.get('motion_label','still')}"
            )

        objs = _fresh(snap, "objects")
        if objs:
            items = objs.get("detections") or []
            if items:
                names: list[str] = []
                for d in items[:5]:
                    n = d.get("label")
                    if n and n not in names:
                        names.append(n)
                if names:
                    bits.append(f"objects visible: {', '.join(names)}")

        # Ambient VLM caption — a short natural-language description
        # of the scene refreshed every ~15 s by SceneCaptioner. Quoted
        # so the LLM sees clearly that it's a separate model's read
        # of the image (not the structured signals above).
        cap = _fresh(snap, "scene_caption")
        if cap:
            text = (cap.get("text") or "").strip()
            if text:
                age = int(max(0, time.time() - float(cap.get("ts", 0))))
                bits.append(f'scene caption ({age} s ago): "{text}"')

        # Active object-tracking sessions started via the track_object
        # tool — surface their current zones so the LLM can answer
        # follow-up questions without recalling the tool every turn.
        try:
            from faceview.vision.tracker import ObjectTracker
            line = ObjectTracker.shared().narrate()
        except Exception:  # noqa: BLE001
            line = ""
        if line:
            bits.append(line)

        # Roster: who the system has on file — so the LLM doesn't
        # ask the same person their name twice and doesn't confuse
        # introductions with strangers it should already know.
        try:
            from faceview.vision.people import PeopleStore
            roster = PeopleStore.shared().list_people()
        except Exception:  # noqa: BLE001
            roster = []
        if roster:
            bits.append("people on file: " + ", ".join(roster))

        if not bits:
            return ""
        return ("Live perception (from the user's webcam, refreshed each "
                "turn): " + "; ".join(bits) + ".")


# ── helpers ──────────────────────────────────────────────────────────────


def _fresh(snap: dict, key: str) -> Optional[dict]:
    d = snap.get(key)
    if d is None or not d.get("fresh"):
        return None
    return d


def _head_pose_label(head: dict) -> str:
    yaw = float(head.get("yaw", 0.0))
    pitch = float(head.get("pitch", 0.0))
    parts: list[str] = []
    if yaw > 0.35:
        parts.append("turned right")
    elif yaw < -0.35:
        parts.append("turned left")
    if pitch > 0.35:
        parts.append("tilted up")
    elif pitch < -0.35:
        parts.append("tilted down")
    if not parts:
        return "facing forward"
    return " and ".join(parts)
