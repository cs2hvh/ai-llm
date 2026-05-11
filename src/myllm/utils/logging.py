"""Structured logging.

Default output is line-delimited JSON for production log aggregation.
Set ``MYLLM_LOG_FORMAT=console`` for human-readable colored output during
development. Log level via ``MYLLM_LOG_LEVEL`` (default ``INFO``).

Usage:
    from myllm.utils import get_logger
    log = get_logger(__name__)
    log.info("event_name", key=value, count=42)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(
    level: str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure structlog. Idempotent — safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = (level or os.environ.get("MYLLM_LOG_LEVEL", "INFO")).upper()
    fmt_name = (fmt or os.environ.get("MYLLM_LOG_FORMAT", "json")).lower()
    log_level = getattr(logging, level_name, logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any
    if fmt_name == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    else:
        renderer = structlog.processors.JSONRenderer(sort_keys=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Also tame the stdlib root logger so libraries that use it print sanely.
    logging.basicConfig(level=log_level, stream=sys.stderr, format="%(message)s")

    _CONFIGURED = True


def get_logger(name: str) -> Any:
    """Return a structlog BoundLogger for the given module."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
