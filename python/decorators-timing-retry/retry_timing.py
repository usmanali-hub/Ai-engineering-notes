"""A timing decorator and a retry decorator with exponential backoff."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

log = logging.getLogger(__name__)


def _name(func: object) -> str:
    """Callable objects (class instances, partials) may not have ``__name__``."""
    return getattr(func, "__name__", repr(func))


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Log how long each call takes. Uses perf_counter, which is monotonic."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            log.info("%s took %.4fs", _name(func), time.perf_counter() - start)

    return wrapper


def retry(
    times: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a function when it raises one of ``exceptions``.

    ``times`` is the total number of attempts. The wait grows by ``backoff`` after each failure
    (0.5s, 1s, 2s, ...). ``sleep`` is injectable so tests do not actually wait.
    Only listed exceptions are retried; anything else propagates immediately.
    """
    if times < 1:
        raise ValueError("times must be at least 1")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            wait = delay
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == times:
                        raise
                    log.warning(
                        "%s failed (%s), retry %d/%d in %.2fs", _name(func), exc, attempt, times - 1, wait
                    )
                    sleep(wait)
                    wait *= backoff
            raise AssertionError("unreachable")  # pragma: no cover

        return wrapper

    return decorator
