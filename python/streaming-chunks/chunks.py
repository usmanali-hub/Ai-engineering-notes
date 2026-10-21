"""Process large inputs lazily with generators and context managers."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TextIO, TypeVar

T = TypeVar("T")


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield lists of up to ``size`` items without loading the whole iterable."""
    if size < 1:
        raise ValueError("size must be at least 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_chunks(f: TextIO, size: int = 1024) -> Iterator[str]:
    """Yield ``size`` characters at a time from an open text file."""
    if size < 1:
        raise ValueError("size must be at least 1")
    while chunk := f.read(size):
        yield chunk


def overlapping_chunks(text: str, size: int, overlap: int) -> Iterator[str]:
    """Split text into windows of ``size`` characters that share ``overlap`` characters.

    This is the same idea used to chunk documents for retrieval, where overlap keeps
    sentences that straddle a boundary retrievable from at least one chunk.
    """
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("need size >= 1 and 0 <= overlap < size")
    step = size - overlap
    for start in range(0, max(len(text) - overlap, 1), step):
        yield text[start : start + size]


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager that records elapsed seconds in ``result['seconds']``."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
