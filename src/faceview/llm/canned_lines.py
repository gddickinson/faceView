"""Canned lines for test mode (and the web demo): the scripted two-bot ping-pong.

Qt-free so the web interface can show the same scripted lines without the event
bus. :mod:`faceview.llm.test_conversation` imports these names.
"""

from __future__ import annotations


SEED_PROMPTS = [
    "Hi! Can you see me on your side?",
    "Yes — and can you see me? What expression am I making?",
    "I think you look curious. Want to play a word game?",
    "Sure. Say a word and I'll respond with an emotion.",
    "Sunshine.",
    "Happy. Storm.",
    "Anxious. Library.",
    "Calm. Your turn — give me a feeling.",
    "Hope. What does hope look like for you?",
    "A small light at the end of a long hallway. Yours?",
    "The first sentence of a book you can't put down.",
]


_EMOTION_KEYWORDS = {
    "happy":    ["sunshine", "smile", "play", "joy", "hope", "calm", "fun", "great"],
    "sad":      ["alone", "lost", "sorry", "miss", "rain", "dark", "regret"],
    "surprise": ["wow", "really", "?", "what", "sudden", "storm", "amazing"],
    "neutral":  [],
}


def _guess_emotion(text: str) -> str:
    t = text.lower()
    for label, words in _EMOTION_KEYWORDS.items():
        for w in words:
            if w in t:
                return label
    return "neutral"
