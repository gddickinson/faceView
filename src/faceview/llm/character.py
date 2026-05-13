"""Character dataclass + registry for the avatar's persona identity.

A Character is the stable "who am I" backbone used by the cognition
system. Loaded from ``assets/config/characters.json`` keyed by persona
name; missing personas fall back to a sensible default Character.

Big Five trait numbers are kept here as plain floats so other parts of
the GUI (idle animation timing, conversational tempo) can read them
later without going through the cognition layer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from faceview.utils.paths import project_root


_CHARACTERS_JSON = (
    project_root() / "src" / "faceview" / "assets" / "config" / "characters.json"
)


@dataclass
class Character:
    """The persona's stable identity. Loaded from characters.json."""

    name: str = "Claude"
    age: Optional[int] = None
    occupation: str = ""
    backstory: str = (
        "You're a helpful conversational avatar in a desktop GUI. "
        "You can see the user via webcam and they can see your face."
    )
    # Big Five — colours responses + (future) idle animation.
    traits: dict[str, float] = field(default_factory=lambda: {
        "openness":          0.7,
        "conscientiousness": 0.7,
        "extraversion":      0.5,
        "agreeableness":     0.7,
        "neuroticism":       0.3,
    })
    conversation: dict[str, Any] = field(default_factory=lambda: {
        "verbosity":          0.5,
        "humor":              0.4,
        "philosophicalDepth": 0.5,
        "topics":             [],
        "catchphrases":       [],
        "outlook":            "",
    })
    goals: list[str] = field(default_factory=list)
    relationship_levels: list[dict] = field(default_factory=lambda: [
        {"level": 1, "name": "Acquaintance", "threshold": 0,
         "unlocks": "Polite chat, surface topics"},
        {"level": 2, "name": "Familiar",     "threshold": 25,
         "unlocks": "Shares opinions, light humour"},
        {"level": 3, "name": "Friend",       "threshold": 75,
         "unlocks": "Personal stories, asks about you"},
        {"level": 4, "name": "Close Friend", "threshold": 200,
         "unlocks": "Vulnerability, references shared history"},
        {"level": 5, "name": "Companion",    "threshold": 500,
         "unlocks": "Full emotional honesty, inside jokes"},
    ])

    # ── queries ───────────────────────────────────────────────────────

    def level_for(self, score: int) -> dict:
        """Highest level whose threshold the score meets."""
        chosen = self.relationship_levels[0]
        for lvl in self.relationship_levels:
            if score >= lvl["threshold"]:
                chosen = lvl
        return chosen

    def narrate_identity(self) -> str:
        """One paragraph for the LLM: who am I?"""
        parts: list[str] = []
        # Skip the auto "You are X, age, occupation." line if the
        # backstory already starts with "You are…" — otherwise both
        # appear back-to-back.
        backstory = (self.backstory or "").strip()
        if not backstory.lower().startswith("you are"):
            intro_bits: list[str] = [self.name]
            if self.age:
                intro_bits.append(str(self.age))
            if self.occupation:
                intro_bits.append(self.occupation)
            parts.append("You are " + ", ".join(intro_bits) + ".")
        if backstory:
            parts.append(backstory)
        topics = (self.conversation or {}).get("topics") or []
        if topics:
            parts.append("Topics you naturally bring up: "
                        + ", ".join(topics[:5]) + ".")
        catch = (self.conversation or {}).get("catchphrases") or []
        if catch:
            parts.append("Catchphrases you sometimes use: "
                        + "; ".join(catch[:3]) + ".")
        if self.goals:
            parts.append("Personal aims: " + "; ".join(self.goals[:3]) + ".")
        outlook = (self.conversation or {}).get("outlook") or ""
        if outlook:
            parts.append("Outlook: " + outlook)
        return " ".join(parts)


_DEFAULT_CHARACTER = Character()


# ── registry ───────────────────────────────────────────────────────────


def load_character_registry() -> dict[str, dict]:
    """Read assets/config/characters.json. Missing = empty registry."""
    if not _CHARACTERS_JSON.exists():
        return {}
    try:
        return json.loads(_CHARACTERS_JSON.read_text())
    except (OSError, ValueError):
        return {}


def list_character_keys() -> list[str]:
    return sorted(load_character_registry().keys())


def character_for(persona: str) -> Character:
    """Pick the registered Character for ``persona`` (or default)."""
    reg = load_character_registry()
    data = reg.get(persona)
    if not data:
        return _DEFAULT_CHARACTER
    base = asdict(_DEFAULT_CHARACTER)
    merged = {**base, **data}
    # Deep-merge dict subfields so partial overrides keep our defaults.
    for key in ("traits", "conversation"):
        merged[key] = {**base.get(key, {}), **(data.get(key) or {})}
    return Character(**merged)
