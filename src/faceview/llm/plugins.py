"""LLM-tool plugin system (I4).

Drop a Python file into ``~/.faceview/plugins/*.py`` (or the
project-local ``.faceview/plugins/*.py``); it can register one or
more LLM-callable tools that auto-appear in both Anthropic + Ollama
engine catalogues.

A minimal plugin looks like:

    # .faceview/plugins/coffee.py
    from faceview.llm.plugins import register_tool

    def _hello(args: dict) -> str:
        return f"Hello, {args.get('name', 'friend')}!"

    register_tool(
        name="greet",
        description="Say hello to the user by name.",
        schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
        executor=_hello,
    )

On boot, :func:`discover_and_load_plugins` is called from `app.py`
which walks the plugin directories and `runpy`-executes each file
(equivalent to ``python -m`` so module globals are isolated). Any
``register_tool`` calls inside populate :data:`_REGISTRY`, which the
engine bundles in `vision_tool.py` then merge into the tool list.

Plugins should keep their executors **pure** — return a string (or
list of Anthropic-shape content blocks) and don't poke at the bus,
panels, or the LLM client. Anything more invasive belongs in the
faceview core, not a plugin.
"""

from __future__ import annotations

import importlib.util
import runpy
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from faceview.config import settings
from faceview.core.logger import get_logger


log = get_logger("plugins")


@dataclass
class PluginTool:
    name: str
    description: str
    schema: dict
    executor: Callable[[dict], Any]


# Module-level registry, mutated by register_tool() at import/exec
# time of each plugin file. Reads (engines, tests) go through the
# helper functions below.
_REGISTRY: dict[str, PluginTool] = {}
_REGISTRY_LOCK = threading.Lock()


# ── public registration API ──────────────────────────────────────


def register_tool(
    name: str,
    description: str,
    schema: dict,
    executor: Callable[[dict], Any],
) -> None:
    """Called from inside a plugin file to add a tool to the catalogue.

    Re-registering the same name overwrites silently — supports a
    live-edit workflow where the user re-discovers plugins after
    tweaking one."""
    if not name or not isinstance(name, str):
        raise ValueError("plugin tool name must be a non-empty string")
    if not callable(executor):
        raise ValueError(f"plugin '{name}' executor is not callable")
    with _REGISTRY_LOCK:
        _REGISTRY[name] = PluginTool(
            name=name,
            description=description or "",
            schema=schema or {"type": "object", "properties": {}},
            executor=executor,
        )
    log.info("plugins.registered", name=name)


def unregister_tool(name: str) -> bool:
    with _REGISTRY_LOCK:
        return _REGISTRY.pop(name, None) is not None


def clear_registry() -> None:
    """Used by tests + the rediscover flow."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


def list_plugin_tools() -> list[PluginTool]:
    with _REGISTRY_LOCK:
        return list(_REGISTRY.values())


def get_plugin_tool(name: str) -> Optional[PluginTool]:
    with _REGISTRY_LOCK:
        return _REGISTRY.get(name)


# ── schema adaptors ──────────────────────────────────────────────


def anthropic_tool_dicts() -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.schema,
        }
        for t in list_plugin_tools()
    ]


def ollama_tool_dicts() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        }
        for t in list_plugin_tools()
    ]


def run_plugin_tool(name: str, args: dict) -> str:
    """Dispatch helper for engines. Always returns a string —
    plugins that return non-string values get coerced via ``str()``
    so the LLM tool-result path stays simple."""
    tool = get_plugin_tool(name)
    if tool is None:
        return f"Unknown plugin tool: {name}"
    try:
        result = tool.executor(args or {})
    except Exception as exc:  # noqa: BLE001
        log.warning("plugins.executor_failed",
                    name=name, error=str(exc))
        return f"Plugin '{name}' raised: {exc}"
    return str(result) if not isinstance(result, str) else result


# ── discovery ────────────────────────────────────────────────────


def _plugin_dirs() -> list[Path]:
    """Plugin search path: ~/.faceview/plugins + any project-local
    overrides. Returns existing dirs only."""
    out: list[Path] = []
    for d in (settings.data_dir / "plugins",):
        if d.exists() and d.is_dir():
            out.append(d)
    return out


def discover_and_load_plugins() -> int:
    """Re-load every ``*.py`` plugin under the search path.

    Returns the total number of tools registered after the sweep.
    Idempotent (clears the registry first) so it's safe to call
    repeatedly — e.g. after the user edits a plugin file."""
    clear_registry()
    loaded = 0
    for d in _plugin_dirs():
        for path in sorted(d.glob("*.py")):
            try:
                runpy.run_path(str(path), run_name="__faceview_plugin__")
                loaded += 1
                log.info("plugins.loaded", path=str(path))
            except Exception as exc:  # noqa: BLE001
                log.warning("plugins.load_failed",
                            path=str(path), error=str(exc))
    return len(list_plugin_tools())
