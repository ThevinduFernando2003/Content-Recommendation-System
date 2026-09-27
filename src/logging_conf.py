"""Logging setup. Library code logs; only CLI scripts print."""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")  # type: ignore[attr-defined]
        return True


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once. Safe to call more than once."""
    log_level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    root.setLevel(log_level)
    if getattr(root, "_netflix_configured", False):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    handler.addFilter(RequestIdFilter())
    root.handlers.clear()
    root.addHandler(handler)
    root._netflix_configured = True  # type: ignore[attr-defined]
