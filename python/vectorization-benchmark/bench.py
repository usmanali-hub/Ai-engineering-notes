"""Compare plain Python loops with vectorized NumPy for three common operations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np


def sum_of_squares_loop(xs: list[float]) -> float:
    total = 0.0
    for x in xs:
        total += x * x
    return total


def sum_of_squares_vec(a: np.ndarray) -> float:
    return float(np.dot(a, a))


def zscore_loop(xs: list[float]) -> list[float]:
    n = len(xs)
    mean = sum(xs) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / n)
    return [(x - mean) / std for x in xs]


def zscore_vec(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / a.std()


def rolling_mean_loop(xs: list[float], window: int) -> list[float]:
    return [sum(xs[i : i + window]) / window for i in range(len(xs) - window + 1)]


def rolling_mean_vec(a: np.ndarray, window: int) -> np.ndarray:
    csum = np.cumsum(np.insert(a, 0, 0.0))
    return (csum[window:] - csum[:-window]) / window


def best_time(func: Callable[[], object], repeat: int = 3) -> float:
    """Return the fastest of ``repeat`` runs in seconds (the minimum is the least noisy)."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


def run(n: int = 1_000_000, window: int = 20, seed: int = 0) -> list[tuple[str, float, float]]:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=n)
    lst = arr.tolist()
    cases = [
        ("sum of squares", lambda: sum_of_squares_loop(lst), lambda: sum_of_squares_vec(arr)),
        ("z-score", lambda: zscore_loop(lst), lambda: zscore_vec(arr)),
        ("rolling mean (w=%d)" % window, lambda: rolling_mean_loop(lst, window), lambda: rolling_mean_vec(arr, window)),
    ]
    return [(name, best_time(loop), best_time(vec)) for name, loop, vec in cases]


if __name__ == "__main__":
    n = 1_000_000
    print(f"n = {n:,}")
    print(f"{'operation':<22}{'loop (s)':>12}{'numpy (s)':>12}{'speedup':>10}")
    for name, loop_s, vec_s in run(n):
        print(f"{name:<22}{loop_s:>12.4f}{vec_s:>12.4f}{loop_s / vec_s:>9.0f}x")
