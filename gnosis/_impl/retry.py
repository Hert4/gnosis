"""
smartsearch/retry.py — Retry helper with exponential backoff.
"""

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRANSIENT = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    )

    _TRANSIENT = _TRANSIENT + (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    )
except ImportError:
    pass


def with_retry(
    func: Callable[[], T],
    retries: int = 3,
    backoff: float = 2.0,
    reraise_on: tuple = (),
) -> T:
    """
    Call func with exponential backoff on transient errors.

    Args:
        func:        zero-arg callable (use lambda to pass args)
        retries:     max attempts (default 3)
        backoff:     base wait seconds, scales with attempt (default 2.0)
        reraise_on:  exception types to raise immediately without retry
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, retries + 1):
        try:
            return func()
        except reraise_on:
            raise
        except Exception as e:
            last_exc = e
            if attempt == retries:
                logger.error(f"[retry] Failed after {retries} attempts: {e}")
                raise
            wait = backoff * attempt
            logger.warning(
                f"[retry] Attempt {attempt}/{retries} failed: {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)
    raise last_exc
