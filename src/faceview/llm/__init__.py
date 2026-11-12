"""LLM integration: Anthropic Claude client + message history.

Exports are resolved lazily (PEP 562) so the Qt-free modules in this package
(e.g. the animation and character code used by the web interface) import without PySide6.
"""

from importlib import import_module

_EXPORTS = {
    "ClaudeClient": "faceview.llm.claude_client",
    "Conversation": "faceview.llm.conversation",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
