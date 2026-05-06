"""stdio MCP server adapter.

Exposes the same operations as the FastAPI server as native MCP tools so a
Claude Code session can invoke them directly. The server is launched as a
separate process via ``tools/run_mcp_server.py``; it does not run inside the
GUI process. It connects back to the GUI via the FastAPI control plane on
``127.0.0.1`` (which the GUI starts at boot).

Configure Claude Code with::

    claude mcp add faceview python -m tools.run_mcp_server

Or add to ``~/.claude.json``::

    "mcpServers": {
      "faceview": {
        "command": "python",
        "args": ["-m", "tools.run_mcp_server"]
      }
    }
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


def _api_base() -> str:
    from faceview.config import settings
    return settings.api_url


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=_api_base(), timeout=10.0)


async def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    async with _client() as cx:
        r = await cx.post(path, json=body)
        r.raise_for_status()
        return r.json()


async def _get(path: str, **params) -> Any:
    async with _client() as cx:
        r = await cx.get(path, params=params)
        r.raise_for_status()
        return r.json()


def build_server():
    """Construct the MCP server. Imported lazily so the package is optional."""
    try:
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        from mcp.types import TextContent, Tool  # type: ignore
    except ImportError as exc:
        from faceview.core.errors import MissingDependency
        raise MissingDependency("mcp", "mcp") from exc

    server = Server("faceview")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="send_chat",
                description="Send a user-message into the faceView chat panel.",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            Tool(
                name="speak",
                description="Have the GUI's TTS speak this text.",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            Tool(
                name="camera_state",
                description="Get current presence/identity/emotion/mouth state.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="list_events",
                description="Recent events from the bus (default last 50).",
                inputSchema={
                    "type": "object",
                    "properties": {"n": {"type": "integer", "default": 50}},
                },
            ),
            Tool(
                name="screenshot",
                description="Save a screenshot of the GUI to docs/images/<name>.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "default": "shot.png"},
                        "encode_b64": {"type": "boolean", "default": False},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "send_chat":
            res = await _post("/chat", {"text": arguments["text"]})
        elif name == "speak":
            res = await _post("/speak", {"text": arguments["text"]})
        elif name == "camera_state":
            res = await _get("/state")
        elif name == "list_events":
            res = await _get("/events", n=arguments.get("n", 50))
        elif name == "screenshot":
            res = await _post("/screenshot", {
                "name": arguments.get("name", "shot.png"),
                "encode_b64": arguments.get("encode_b64", False),
            })
        else:
            res = {"ok": False, "error": f"unknown tool: {name}"}

        return [TextContent(type="text", text=str(res))]

    return server, stdio_server


async def main() -> None:
    server, stdio_server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def run() -> None:
    asyncio.run(main())
