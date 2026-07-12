# Logging configuration
# ============================================================================
# app/utils/logger.py
# ----------------------------------------------------------------------------
# Structured logging setup (console + rotating file) and latency-measurement
# helpers (a Stopwatch-style context manager + decorator).
#
# The spec explicitly asks us to log retrieval time, embedding time and LLM
# response time. `timer()` and `timed()` make that a one-liner anywhere.
#
# In ASP.NET terms: the ILoggerFactory configuration + a reusable Stopwatch.
# ============================================================================

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import lru_cache, wraps
from logging.handlers import RotatingFileHandler
from typing import Callable, Iterator, TypeVar

from app.config import settings

# A generic type var so the @timed decorator preserves the wrapped function's
# signature/return type for type-checkers.
F = TypeVar("F", bound=Callable[..., object])


# ----------------------------------------------------------------------------
# 1. LOGGING
# ----------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@lru_cache(maxsize=1)
def _configure_root_logging() -> None:
    """Configure the root logger exactly once per process.

    We attach two handlers:
      * StreamHandler       -> prints to the console (terminal).
      * RotatingFileHandler -> writes to logs/app.log, rotating at 5 MB and
        keeping 3 backups so logs never fill the disk.

    Guarded by lru_cache so repeated imports never add duplicate handlers
    (a classic bug that makes every log line print 2x, 3x, ...).
    """
    settings.ensure_directories()

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=str(settings.logs_dir / "app.log"),
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=3,              # keep app.log.1 .. app.log.3
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Clear any pre-existing handlers (e.g. added by uvicorn) to avoid dupes.
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, ensuring root logging is configured first.

    Usage:
        from app.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("hello")
    """
    _configure_root_logging()
    return logging.getLogger(name)


# ----------------------------------------------------------------------------
# 2. TIMING (latency measurement)
# ----------------------------------------------------------------------------
@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """Context manager that measures wall-clock time of a code block.

    Yields a mutable dict; after the block exits, dict['seconds'] and
    dict['ms'] hold the elapsed time. Also logs it at INFO level.

    Example:
        with timer("retrieval", log) as t:
            chunks = retriever.search(q)
        # t['ms'] now available
    """
    log = logger or get_logger("utils.timer")
    result: dict = {"label": label, "seconds": 0.0, "ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["seconds"] = elapsed
        result["ms"] = elapsed * 1000.0
        log.info("[timing] %s took %.1f ms", label, result["ms"])


def timed(label: str | None = None) -> Callable[[F], F]:
    """Decorator version of `timer` for whole functions.

    Example:
        @timed("embed_texts")
        def embed(...): ...
    """
    def decorator(func: F) -> F:
        tag = label or func.__name__

        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            log = get_logger(func.__module__)
            with timer(tag, log):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
