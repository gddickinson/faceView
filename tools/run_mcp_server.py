"""Entry point for the stdio MCP server.

Launched by Claude Code as a stdio child process. Connects back to the GUI's
FastAPI control plane (``http://127.0.0.1:8765`` by default) so the GUI must
already be running.

Usage::

    python -m tools.run_mcp_server
"""

from __future__ import annotations

from faceview.server.mcp_server import run


if __name__ == "__main__":
    run()
