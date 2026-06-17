"""Minimal async retry helper: exponential backoff + jitter for transient faults.

Intentionally dependency-free (no tenacity) and small. The caller decides what
is transient by passing the exception types in ``retry_on`` — nothing else is
retried, so non-retriable errors (e.g. a 4xx / validation) surface immediately.
"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, Tuple, Type, TypeVar

from loguru import logger

T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    label: str = "request",
) -> T:
    """Call ``fn`` (an async, no-arg callable), retrying transient failures.

    Backoff is exponential (``base_delay * 2**n``, capped at ``max_delay``) with
    jitter (50–100% of the computed delay) to avoid synchronized retry storms.
    Only exceptions in ``retry_on`` are retried; anything else propagates at once.
    The final failure is re-raised after ``attempts`` tries.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random() / 2  # jitter: 50–100% of the delay
            logger.warning(
                f"{label}: attempt {attempt}/{attempts} failed "
                f"({type(exc).__name__}: {exc}); retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
