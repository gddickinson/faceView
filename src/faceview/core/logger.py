"""Logging setup.

Use :func:`get_logger` everywhere. structlog gives us key=value formatting
which is much easier to scan in a multi-thread app than free-form ``%s``
strings.
"""

from __future__ import annotations

import logging
import sys

import structlog


_CONFIGURED = False


def configure(level: str = "INFO") -> None:
    """Configure structlog + the stdlib root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure()
    return structlog.get_logger(name)
