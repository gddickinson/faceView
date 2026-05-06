"""Local control surface: FastAPI HTTP API + stdio MCP server.

Both adapters wrap the same :class:`~faceview.server.service.Service` so that
adding a new operation only requires one implementation.
"""

from faceview.server.service import Service

__all__ = ["Service"]
