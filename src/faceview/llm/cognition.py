"""Three-layer cognition + character system for the faceView avatar.

Architecture ported from the table_games "Living AI Personalities"
design, tuned for real-time chat instead of game rounds. Each persona
gets a :class:`CognitionStore` persisted to
``data_dir()/memory/<persona>.json``. The store has:

- **Character** — stable identity: name, backstory, Big Five traits,
  conversation style, personal goals, relationship-level thresholds.
  Loaded from ``assets/config/characters.json`` with a sensible default
  fallback.
- **Episodic** — time-stamped events with significance + felt emotion +
  rehearsal count. Recall is scored by recency × significance ×
  emotional intensity × context relevance × rehearsal. Consolidation
  prunes to ~400 entries when over 500.
- **Semantic** — facts/beliefs keyed by subject (``player``, ``self``,
  any other name). Each fact carries a confidence in [0, 1]. No decay.
- **Emotional** — current emotions with exponential decay. Half-life
  of ~6 hours; emotions below 0.05 intensity drop off.
- **Relationship score** — accumulates from each significant turn;
  brackets into named levels that gate conversational latitude.

The narrative for the LLM is rebuilt every turn from all four layers
plus the character sheet, so every engine (Anthropic / Ollama / Demo)
gets the same picture of "who am I and what do I know about us".
"""

from __future__ import annotations

import json
import math
import re
import threading
import time as _time
from pathlib import Path
from typing import Any, Optional

from faceview.llm.character import Character, character_for
from faceview.utils.paths import data_dir


# ── decay + retention parameters (real-time pace) ───────────────────

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_HOUR = 3_600.0

# Recall recency: exponential half-life in days. Older memories score
# lower in recall but aren't deleted purely by age.
RECENCY_HALF_LIFE_DAYS = 30.0

# Hard cap on episodic memories. When exceeded, the lowest-retention
# 20% are dropped. Retention = significance + rehearsal − age_months.
EPISODIC_HARD_CAP = 500
EPISODIC_TRIM_TO = 400

# Emotional decay: half-life in hours.
EMOTION_HALF_LIFE_HOURS = 6.0
EMOTION_FLOOR = 0.05

# Boost for emotionally-loaded recall.
EMOTIONAL_BOOST_FOR = {"joy", "anger", "surprise", "frustration", "tenderness", "fear"}


# ── importance / emotion scoring helpers ────────────────────────────


_PREFERENCE_RE = re.compile(
    r"\bi (?:like|love|hate|prefer|enjoy|can't stand|dislike|adore|miss|hope)\b",
    re.IGNORECASE,
)
_PERSONAL_RE = re.compile(
    r"\b(?:my name is|i'm called|call me|i live in|i work as|i'm from|i'm a|"
    r"my (?:wife|husband|partner|daughter|son|mother|father|kid|child|dog|cat|job|birthday))\b",
    re.IGNORECASE,
)
_REMEMBER_RE = re.compile(r"\bremember (?:this|that|me|my|when)\b", re.IGNORECASE)
_QUESTION_RE = re.compile(r"\?")

# Tiny keyword → emotion lookup for tagging the felt emotion of a turn.
_EMOTION_WORDS = {
    "joy":          ("love", "happy", "great", "wonderful", "awesome", "thank", "haha", "lol"),
    "frustration":  ("annoy", "stupid", "broken", "ugh", "again", "useless"),
    "anger":        ("angry", "hate", "furious", "rage", "pissed"),
    "sadness":      ("sad", "miss", "lonely", "lost", "gone", "tired"),
    "surprise":     ("wow", "really", "what?!", "no way"),
    "fear":         ("scared", "afraid", "worried", "anxious"),
    "tenderness":   ("dear", "darling", "thank you so much", "appreciate"),
    "curiosity":    ("why", "how", "what is", "explain"),
}


def _score_significance(user_text: str, assistant_text: str) -> int:
    """0–10 significance for one chat turn (used by recall + consolidation)."""
    s = 3  # default mid-low
    if _PERSONAL_RE.search(user_text):
        s = 9
    if _REMEMBER_RE.search(user_text):
        s = max(s, 8)
    if _PREFERENCE_RE.search(user_text):
        s = max(s, 6)
    if _QUESTION_RE.search(user_text) and len(user_text) > 40:
        s = max(s, 5)
    if len(user_text) > 200 or len(assistant_text) > 240:
        s = max(s, 4)
    return min(10, s)


def _felt_emotion(text: str) -> str:
    """Cheap keyword-based emotion tag for the chat turn."""
    t = text.lower()
    best, best_hits = "neutral", 0
    for emotion, words in _EMOTION_WORDS.items():
        hits = sum(1 for w in words if w in t)
        if hits > best_hits:
            best, best_hits = emotion, hits
    return best


# ── CognitionStore ──────────────────────────────────────────────────


class CognitionStore:
    """Per-persona persistent cognition: episodic + semantic + emotional
    + character + relationship progression."""

    # Shared cross-persona facts about the user (C7). Lives at
    # ~/.faceview/memory/_shared.json so every persona reads it when
    # narrating identity context — your name should follow you when
    # you swap from Claude to Iris to Theo.
    SHARED_FILE = "_shared.json"

    SCHEMA_VERSION = 3

    # Class-level "global" incognito flag — applies to every loaded
    # store. Toggle via :meth:`set_incognito` so callers don't need a
    # store handle. While True, ``record_chat_turn`` is a no-op:
    # nothing is added to episodic / per_person / semantic and the
    # relationship score doesn't tick up. Reads (``recall``,
    # ``narrate_for_prompt``) keep working from existing memory.
    _incognito_lock = threading.Lock()
    _incognito = False

    @classmethod
    def set_incognito(cls, on: bool) -> None:
        with cls._incognito_lock:
            cls._incognito = bool(on)

    @classmethod
    def is_incognito(cls) -> bool:
        with cls._incognito_lock:
            return cls._incognito

    def __init__(self, persona: str) -> None:
        self.persona = persona
        self.character: Character = character_for(persona)
        self.first_seen: Optional[str] = None
        self.session_count: int = 0
        self.relationship_score: int = 0
        self.episodic: list[dict[str, Any]] = []
        # Per-person episodic branches — see C3 on the roadmap. Keyed
        # by the display name from PeopleStore. Writes are routed to
        # the matching bucket when the live identity from
        # PerceptionStore is non-"stranger"; otherwise writes go to
        # the shared `episodic` list as before.
        self.per_person: dict[str, list[dict[str, Any]]] = {}
        self.semantic: dict[str, dict[str, dict]] = {}
        self.emotional: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._last_consolidate = 0.0
        # Set by ClaudeClient before each engine.stream_reply call so
        # narrate_for_prompt can do embedding-based retrieval against
        # the current user message. None → fall back to recency +
        # significance ranking.
        self._query_context: Optional[str] = None
        # Optional override: if explicitly set via set_current_speaker
        # we use it; otherwise we look up the live identity from
        # PerceptionStore on demand.
        self._speaker_override: Optional[str] = None

    # ── persistence ─────────────────────────────────────────────

    @classmethod
    def path_for(cls, persona: str) -> Path:
        d = data_dir() / "memory"
        d.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", persona) or "default"
        return d / f"{safe}.json"

    # ── shared cross-persona facts (C7) ────────────────────────

    @classmethod
    def _shared_path(cls) -> Path:
        d = data_dir() / "memory"
        d.mkdir(parents=True, exist_ok=True)
        return d / cls.SHARED_FILE

    @classmethod
    def shared_facts(cls) -> dict[str, Any]:
        """Read the shared cross-persona fact bag. Keys are the same
        as ``semantic`` subjects (``player``, ``history``) but only
        the ones flagged ``shared=True`` show up here."""
        p = cls._shared_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    @classmethod
    def _write_shared_facts(cls, facts: dict[str, Any]) -> None:
        p = cls._shared_path()
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(facts, indent=2))
        tmp.replace(p)

    def share_fact(self, subject: str, key: str, value: Any,
                    *, confidence: float = 0.9) -> None:
        """Set a fact AND replicate it to the shared bag so every
        persona sees it. Used by record_chat_turn for player.name
        (your name) so swapping persona doesn't lose it."""
        self.set_fact(subject, key, value, confidence=confidence)
        try:
            facts = type(self).shared_facts()
            bucket = facts.setdefault(subject, {})
            bucket[key] = {
                "value": value,
                "confidence": float(confidence),
                "updated": _time.time(),
            }
            type(self)._write_shared_facts(facts)
        except OSError:
            pass

    @classmethod
    def load(cls, persona: str) -> "CognitionStore":
        store = cls(persona)
        p = cls.path_for(persona)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                schema = int(data.get("schema") or 1)
                if schema >= 2:
                    store._load_v2(data)
                else:
                    store._migrate_v1(data)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        # First contact: stamp the date + bump session count.
        if store.first_seen is None:
            store.first_seen = _time.strftime("%Y-%m-%d")
            store._dirty = True
        store.session_count += 1
        store._dirty = True
        return store

    def _load_v2(self, data: dict) -> None:
        """v2 + v3 share most fields. v3 adds ``per_person``; on a v2
        file that key is missing and defaults to {}."""
        self.first_seen = data.get("first_seen")
        self.session_count = int(data.get("session_count") or 0)
        self.relationship_score = int(data.get("relationship_score") or 0)
        self.episodic = data.get("episodic") or []
        self.per_person = data.get("per_person") or {}
        self.semantic = data.get("semantic") or {}
        self.emotional = data.get("emotional") or {}

    def _migrate_v1(self, data: dict) -> None:
        """Bring forward old MemoryStore JSONs from earlier this session."""
        ledger = data.get("ledger") or {}
        self.first_seen = ledger.get("first_seen") or data.get("first_seen")
        # Old memories → episodic with default emotion + significance from importance.
        for mem in data.get("memories") or []:
            self.episodic.append({
                "ts":           mem.get("ts", _time.time()),
                "type":         mem.get("type", "chat"),
                "text":         mem.get("text", ""),
                "significance": int(mem.get("importance", 3)) * 2,  # 1-5 → 2-10
                "emotion":      "neutral",
                "session_id":   0,
                "recalled":     0,
            })
        # Old ledger → semantic facts.
        if ledger.get("user_name"):
            self.set_fact("player", "name", ledger["user_name"], confidence=1.0)
        for p in ledger.get("preferences") or []:
            self.set_fact("player", f"pref_{int(p.get('ts', 0))}",
                          p.get("text", ""), confidence=0.85)
        for m in ledger.get("milestones") or []:
            self.set_fact("history", f"event_{int(m.get('ts', 0))}",
                          m.get("text", ""), confidence=1.0)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        p = self.path_for(self.persona)
        payload = {
            "schema": self.SCHEMA_VERSION,
            "persona": self.persona,
            "first_seen": self.first_seen,
            "session_count": self.session_count,
            "relationship_score": self.relationship_score,
            "episodic": self.episodic,
            "per_person": self.per_person,
            "semantic": self.semantic,
            "emotional": self.emotional,
            "saved_at": _time.time(),
        }
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(p)
        self._dirty = False

    # ── episodic ────────────────────────────────────────────────

    def record_episode(self, type_: str, text: str, *,
                       significance: int = 3,
                       emotion: str = "neutral",
                       embedding: Optional[list[float]] = None,
                       speaker: Optional[str] = None) -> None:
        """Append an episode. When ``speaker`` (or the live identity
        from PerceptionStore) names a known person, the entry is
        filed under :attr:`per_person`; otherwise it goes to the
        shared :attr:`episodic` list."""
        entry: dict[str, Any] = {
            "ts": _time.time(),
            "type": type_,
            "text": text,
            "significance": max(1, min(10, int(significance))),
            "emotion": emotion,
            "session_id": self.session_count,
            "recalled": 0,
        }
        if embedding is not None:
            entry["embedding"] = embedding
        who = speaker or self.current_speaker()
        if who and who.lower() != "stranger":
            entry["speaker"] = who
            self.per_person.setdefault(who, []).append(entry)
        else:
            self.episodic.append(entry)
        self._dirty = True
        self._maybe_consolidate()

    def _maybe_consolidate(self) -> None:
        now = _time.time()
        if now - self._last_consolidate < 60.0:
            return
        # Retention = significance × 2 + rehearsal × 0.5 − age_in_months
        def _score(m: dict) -> float:
            age_m = (now - m.get("ts", now)) / (SECONDS_PER_DAY * 30)
            return (m.get("significance", 3) * 2
                    + m.get("recalled", 0) * 0.5
                    - age_m)
        if len(self.episodic) > EPISODIC_HARD_CAP:
            self.episodic.sort(key=_score, reverse=True)
            self.episodic = self.episodic[:EPISODIC_TRIM_TO]
            self._last_consolidate = now
        # Also trim each per-person bucket — fewer entries each but
        # the same per-bucket caps so a single chatty person can't
        # blow up the file.
        for name, bucket in list(self.per_person.items()):
            if len(bucket) > EPISODIC_HARD_CAP:
                bucket.sort(key=_score, reverse=True)
                self.per_person[name] = bucket[:EPISODIC_TRIM_TO]
                self._last_consolidate = now

    # ── memory consolidation / forgetting curve (C9) ─────────────

    def run_forgetting_pass(self) -> int:
        """Background sweep that drops low-retention episodic
        entries even when we're well under the hard cap.

        Heuristic:
          * Old (>180 days), unrehearsed (recalled=0), and low-
            significance (≤2) entries are dropped.
          * Frequently-rehearsed memories (recalled ≥ 5) are
            promoted into semantic facts under ``self/recalled_*``
            so they survive consolidation.

        Returns the number of entries dropped. Designed to be
        called on a slow timer (every few minutes) from a
        supervisor / app-level scheduler — NOT from inside
        record_chat_turn (which should stay fast).
        """
        now = _time.time()
        before = len(self.episodic)
        kept: list[dict] = []
        for mem in self.episodic:
            age_d = (now - mem.get("ts", now)) / SECONDS_PER_DAY
            sig = int(mem.get("significance", 3))
            recalled = int(mem.get("recalled", 0))
            # Promote well-rehearsed memories.
            if recalled >= 5:
                key = f"recalled_{int(mem.get('ts', 0))}"
                self.set_fact(
                    "self", key,
                    str(mem.get("text", ""))[:180],
                    confidence=min(0.95, 0.6 + 0.05 * recalled),
                )
            # Drop forgettable ones.
            if (age_d > 180 and recalled == 0 and sig <= 2):
                continue
            kept.append(mem)
        # Same sweep across per-person buckets.
        for name, bucket in list(self.per_person.items()):
            new_bucket: list[dict] = []
            for mem in bucket:
                age_d = (now - mem.get("ts", now)) / SECONDS_PER_DAY
                sig = int(mem.get("significance", 3))
                recalled = int(mem.get("recalled", 0))
                if recalled >= 5:
                    key = f"recalled_{name}_{int(mem.get('ts', 0))}"
                    self.set_fact(
                        "self", key,
                        str(mem.get("text", ""))[:180],
                        confidence=min(0.95, 0.6 + 0.05 * recalled),
                    )
                if (age_d > 180 and recalled == 0 and sig <= 2):
                    continue
                new_bucket.append(mem)
            self.per_person[name] = new_bucket
        self.episodic = kept
        dropped = before - len(kept)
        if dropped:
            self._dirty = True
        return dropped

    def recall(self, context: str, *, limit: int = 5) -> list[dict]:
        """Score memories by recency × significance × emotion × relevance × rehearsal."""
        if not self.episodic:
            return []
        now = _time.time()
        ctx_words = [w for w in context.lower().split() if len(w) > 2]
        scored: list[tuple[float, dict]] = []
        for mem in self.episodic:
            age_d = (now - mem.get("ts", now)) / SECONDS_PER_DAY
            recency = math.exp(-math.log(2) * age_d / RECENCY_HALF_LIFE_DAYS)
            sig = mem.get("significance", 3) / 10.0
            emo_boost = 0.3 if mem.get("emotion") in EMOTIONAL_BOOST_FOR else 0.0
            text = (mem.get("text") or "").lower()
            mtype = (mem.get("type") or "").lower()
            relevance = 0.0
            for w in ctx_words:
                if w in text: relevance += 0.2
                if w in mtype: relevance += 0.3
            rehearsal = min(0.3, mem.get("recalled", 0) * 0.05)
            score = (recency * 0.3 + sig * 0.25 + relevance * 0.25
                     + emo_boost + rehearsal)
            scored.append((score, mem))
        scored.sort(key=lambda x: -x[0])
        out = [m for _s, m in scored[:limit]]
        for mem in out:
            mem["recalled"] = mem.get("recalled", 0) + 1
        if out:
            self._dirty = True
        return out

    # ── semantic ────────────────────────────────────────────────

    def set_fact(self, subject: str, key: str, value: Any,
                 *, confidence: float = 0.85) -> None:
        bucket = self.semantic.setdefault(subject, {})
        bucket[key] = {
            "value": value,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "updated": _time.time(),
        }
        self._dirty = True

    def get_fact(self, subject: str, key: str) -> Any:
        return (self.semantic.get(subject) or {}).get(key, {}).get("value")

    def all_facts(self, subject: str) -> dict[str, Any]:
        return {k: v["value"] for k, v in (self.semantic.get(subject) or {}).items()}

    # ── emotional ───────────────────────────────────────────────

    def set_emotion(self, label: str, intensity: float,
                    trigger: str = "") -> None:
        self.emotional[label] = {
            "intensity": max(0.0, min(1.0, float(intensity))),
            "trigger": trigger,
            "ts": _time.time(),
        }
        self._dirty = True

    def current_emotions(self) -> dict[str, dict]:
        """All emotions after applying exponential decay."""
        now = _time.time()
        decayed: dict[str, dict] = {}
        for label, data in self.emotional.items():
            hours = (now - data.get("ts", now)) / SECONDS_PER_HOUR
            factor = math.exp(-math.log(2) * hours / EMOTION_HALF_LIFE_HOURS)
            intensity = data.get("intensity", 0.0) * factor
            if intensity >= EMOTION_FLOOR:
                decayed[label] = {
                    "intensity": intensity,
                    "trigger": data.get("trigger", ""),
                }
        return decayed

    def dominant_emotion(self) -> tuple[str, float]:
        current = self.current_emotions()
        if not current:
            return "neutral", 0.0
        label, data = max(current.items(), key=lambda kv: kv[1]["intensity"])
        return label, data["intensity"]

    # ── chat turn integration ───────────────────────────────────

    def record_chat_turn(self, user_text: str, assistant_text: str) -> None:
        if not user_text and not assistant_text:
            return
        # Incognito (C6) — drop the entire turn on the floor. No
        # episodic write, no fact extraction, no relationship bump.
        # Existing memory stays loaded so the LLM still sounds like
        # itself; only the current turn vanishes.
        if CognitionStore.is_incognito():
            return
        sig = _score_significance(user_text, assistant_text)
        felt = _felt_emotion(f"{user_text}\n{assistant_text}")
        text = f"User: {user_text.strip()[:240]}"
        if assistant_text:
            text += f" — You: {assistant_text.strip()[:240]}"
        # Embed the turn for later retrieval. We embed the user's side
        # (what they said) since retrieval queries are also user
        # messages — same distribution. None on missing dep / empty.
        embedding: Optional[list[float]] = None
        try:
            from faceview.llm.embeddings import EmbeddingService
            embedding = EmbeddingService.shared().embed(user_text)
        except Exception:  # noqa: BLE001 — never block memory write
            embedding = None
        self.record_episode("chat", text, significance=sig,
                            emotion=felt, embedding=embedding)
        # Extract facts.
        m = re.search(r"my name is ([A-Z][a-zA-Z'\-]{1,40})",
                      user_text, re.IGNORECASE)
        if m:
            # C7 — names propagate across personas via the shared bag.
            self.share_fact("player", "name", m.group(1).strip(),
                             confidence=1.0)
        if _PREFERENCE_RE.search(user_text):
            self.set_fact("player",
                          f"pref_{int(_time.time())}",
                          user_text.strip()[:200],
                          confidence=0.8)
        if _REMEMBER_RE.search(user_text):
            self.set_fact("history",
                          f"event_{int(_time.time())}",
                          user_text.strip()[:200],
                          confidence=1.0)
        # Felt emotion bumps the avatar's emotional state.
        if felt != "neutral":
            self.set_emotion(felt, intensity=0.4, trigger="chat")
        # Relationship score grows with significance.
        self.relationship_score += sig
        self._dirty = True

    def maybe_decay_and_compact(self) -> None:
        self._maybe_consolidate()

    # ── relationship ────────────────────────────────────────────

    def relationship(self) -> dict[str, Any]:
        lvl = self.character.level_for(self.relationship_score)
        return {
            "score": self.relationship_score,
            "level": lvl["level"],
            "name": lvl["name"],
            "unlocks": lvl.get("unlocks", ""),
        }

    # ── current-speaker awareness (C3) ───────────────────────────

    def set_current_speaker(self, name: Optional[str]) -> None:
        """Explicit override for who we're talking to. ``None`` falls
        back to whatever PerceptionStore reports."""
        self._speaker_override = name

    def current_speaker(self) -> Optional[str]:
        """Best guess of the currently-visible person's name.

        Order: explicit override → live PerceptionStore identity (if
        fresh and not ``stranger``) → ``None``."""
        if self._speaker_override is not None:
            return self._speaker_override
        try:
            from faceview.vision.perception import PerceptionStore
            snap = PerceptionStore.shared().snapshot_dict()
            ident = snap.get("identity")
            if ident and ident.get("fresh"):
                label = ident.get("label") or ""
                if label and label.lower() != "stranger":
                    return label
        except Exception:  # noqa: BLE001
            pass
        return None

    def _episodes_for(self, speaker: Optional[str]) -> list[dict]:
        """All episodes visible to a given speaker — their own bucket
        plus the shared global episodes. Order isn't sorted; callers
        score themselves."""
        if speaker is None:
            return list(self.episodic)
        return list(self.per_person.get(speaker, [])) + list(self.episodic)

    # ── retrieval-augmented recall ──────────────────────────────

    def set_query_context(self, text: Optional[str]) -> None:
        """Set by ClaudeClient before each engine call so
        :meth:`narrate_for_prompt` can do retrieval against the
        actual user message. Pass ``None`` to clear."""
        self._query_context = text

    def recall_by_embedding(
        self, query: str, *, limit: int = 3,
        min_similarity: float = 0.25,
    ) -> list[dict]:
        """Top-K episodic memories by cosine similarity to ``query``.

        Searches across the shared episodic list plus the current
        speaker's per-person bucket (if any) so the conversation has
        access to "things we've discussed before". Empty when no
        episodes have embeddings or the query can't be embedded."""
        speaker = self.current_speaker()
        pool = self._episodes_for(speaker)
        if not query or not pool:
            return []
        try:
            from faceview.llm.embeddings import EmbeddingService, cosine
            q_vec = EmbeddingService.shared().embed(query)
        except Exception:  # noqa: BLE001
            return []
        if q_vec is None:
            return []
        scored: list[tuple[float, dict]] = []
        for mem in pool:
            emb = mem.get("embedding")
            if not emb:
                continue
            sim = cosine(q_vec, emb)
            if sim >= min_similarity:
                scored.append((sim, mem))
        scored.sort(key=lambda x: -x[0])
        out = [m for _s, m in scored[:limit]]
        for mem in out:
            mem["recalled"] = mem.get("recalled", 0) + 1
        if out:
            self._dirty = True
        return out

    # ── narration for LLM system prompt ─────────────────────────

    def narrate_for_prompt(self, *, recall_context: Optional[str] = None,
                           recall_n: int = 5) -> str:
        sections: list[str] = []

        sections.append("[Identity]\n" + self.character.narrate_identity())

        rel = self.relationship()
        rel_line = (f"[Relationship] Level {rel['level']} — {rel['name']}"
                    f" (score {rel['score']}). {rel['unlocks']}.")
        if self.first_seen:
            rel_line += f" First met {self.first_seen}. Session {self.session_count}."
        sections.append(rel_line)

        # Mood
        label, intensity = self.dominant_emotion()
        if intensity > 0:
            sections.append(f"[Mood] {label} ({int(intensity*100)}%).")

        # C7 — fold any shared cross-persona facts into the local
        # view so a persona that doesn't have its own copy of the
        # user's name still sees it.
        try:
            shared = type(self).shared_facts() or {}
            for subject, bucket in shared.items():
                local = self.semantic.setdefault(subject, {})
                for k, v in bucket.items():
                    if k not in local:
                        local[k] = v
        except Exception:  # noqa: BLE001
            pass

        # Facts about player + history
        player_facts = self.all_facts("player")
        if player_facts:
            name = player_facts.get("name")
            other = {k: v for k, v in player_facts.items() if k != "name"}
            bits: list[str] = []
            if name:
                bits.append(f"Their name is {name}.")
            if other:
                bits.append("They've told you: "
                            + "; ".join(str(v)[:100] for v in list(other.values())[-4:])
                            + ".")
            sections.append("[Player] " + " ".join(bits))

        history = self.all_facts("history")
        if history:
            recent = list(history.values())[-3:]
            sections.append("[Shared history] "
                            + "; ".join(str(h)[:100] for h in recent) + ".")

        # Per-person ledger — when we recognise who we're talking to,
        # show the last few exchanges WITH THAT PERSON specifically.
        # This is the C3 unlock: George's "did you finish that bug
        # yet" comes back to George, not to Alice's session next time.
        speaker = self.current_speaker()
        if speaker:
            bucket = self.per_person.get(speaker) or []
            recent = bucket[-3:]
            if recent:
                bits = [f"- {m.get('text','')}" for m in recent]
                sections.append(
                    f"[Conversation history with {speaker}]\n"
                    + "\n".join(bits)
                )

        # Retrieval-augmented: if a query context is set (the live
        # user message) and we have embeddings, surface semantically
        # similar past episodes BEFORE the keyword-based recall block.
        # They're tagged separately so the LLM (and a reader of the
        # system prompt) can tell them apart.
        query = self._query_context
        rel_mems = (self.recall_by_embedding(query, limit=3)
                    if query else [])
        if rel_mems:
            seen_ts = {m.get("ts") for m in rel_mems}
            lines = []
            for m in rel_mems:
                emo = m.get("emotion", "neutral")
                tag = f" (felt: {emo})" if emo != "neutral" else ""
                lines.append(f"- {m.get('text','')}{tag}")
            sections.append(
                "[Relevant past memories — semantically similar to "
                "what they just said]\n" + "\n".join(lines)
            )
        else:
            seen_ts = set()

        # Episodic recall — recent + relevant (keyword-scored).
        effective_ctx = recall_context or query or "recent"
        mems = self.recall(effective_ctx, limit=recall_n)
        # De-duplicate against the embedding-retrieved block above.
        mems = [m for m in mems if m.get("ts") not in seen_ts]
        if mems:
            lines = []
            for m in mems:
                emo = m.get("emotion", "neutral")
                tag = f" (felt: {emo})" if emo != "neutral" else ""
                lines.append(f"- {m.get('text','')}{tag}")
            sections.append("[Memories]\n" + "\n".join(lines))

        return "\n\n".join(sections)

    # ── summary / clear ─────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "character": self.character.name,
            "first_seen": self.first_seen,
            "session_count": self.session_count,
            "user_name": self.get_fact("player", "name"),
            "relationship": self.relationship(),
            "episodic": len(self.episodic),
            "per_person": {n: len(b) for n, b in self.per_person.items()},
            "semantic_subjects": list(self.semantic.keys()),
            "current_emotion": self.dominant_emotion(),
            "path": str(self.path_for(self.persona)),
        }

    def clear(self) -> None:
        self.episodic = []
        self.per_person = {}
        self.semantic = {}
        self.emotional = {}
        self.relationship_score = 0
        self.first_seen = _time.strftime("%Y-%m-%d")
        self._dirty = True

    # ── memory editing (C5) ─────────────────────────────────────

    def forget_recent(self, n: int = 1) -> int:
        """Remove the N most-recent episodic entries.

        Searches the shared list + every per-person bucket and pops
        the latest by timestamp. Returns the number actually removed.
        Used by the ``forget_memory`` tool when no query is given."""
        candidates: list[tuple[float, list, int]] = []
        for i, m in enumerate(self.episodic):
            candidates.append((float(m.get("ts", 0)), self.episodic, i))
        for bucket in self.per_person.values():
            for i, m in enumerate(bucket):
                candidates.append((float(m.get("ts", 0)), bucket, i))
        if not candidates:
            return 0
        candidates.sort(key=lambda c: -c[0])
        removed = 0
        # Track indices we've popped per list so subsequent pops on
        # the same list adjust correctly.
        for _ts, bucket, idx in candidates[:max(0, int(n))]:
            try:
                bucket.pop(idx)
                removed += 1
            except IndexError:
                pass
        if removed:
            self._dirty = True
        return removed

    def forget_matching(self, query: str, *, limit: int = 1) -> int:
        """Remove up to ``limit`` episodes whose text contains
        ``query`` (case-insensitive). Returns the count removed."""
        q = (query or "").strip().lower()
        if not q:
            return 0
        targets: list[tuple[list, int]] = []
        for i, m in enumerate(self.episodic):
            if q in str(m.get("text", "")).lower():
                targets.append((self.episodic, i))
        for bucket in self.per_person.values():
            for i, m in enumerate(bucket):
                if q in str(m.get("text", "")).lower():
                    targets.append((bucket, i))
        if not targets:
            return 0
        # Highest indices first so subsequent pops don't shift earlier ones.
        targets.sort(key=lambda t: -t[1])
        removed = 0
        for bucket, idx in targets[:max(0, int(limit))]:
            try:
                bucket.pop(idx)
                removed += 1
            except IndexError:
                pass
        if removed:
            self._dirty = True
        return removed
