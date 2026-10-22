# Vectorization benchmark

**Goal:** measure, not assume, how much faster NumPy is than plain Python loops for three common
operations, and keep the vectorized versions provably equal to the loop versions.

## What's here

- `bench.py`: loop and vectorized versions of sum of squares, z-score and rolling mean, plus a small timing helper
- `test_bench.py`: tests that each vectorized version matches its loop version and is faster on large input

## Result (n = 1,000,000 random floats, best of 3 runs)

| Operation | Python loop | NumPy | Speedup |
|---|---|---|---|
| Sum of squares | 0.0217 s | 0.0001 s | ~145x |
| Z-score | 0.1416 s | 0.0044 s | ~32x |
| Rolling mean (window 20) | 0.3855 s | 0.0106 s | ~36x |

Numbers come from one run on a cloud sandbox, so treat them as order of magnitude. Run
`python bench.py` to get your own.

## Key ideas

- **Why it is faster:** NumPy runs the loop in compiled C over a contiguous block of memory, avoiding
  per-element Python object overhead and interpreter dispatch.
- **Rolling mean without a loop:** a cumulative sum turns each window sum into one subtraction
  (`csum[i+w] - csum[i]`), so the cost no longer grows with the window size.
- **Always test equivalence.** A fast function that returns different numbers is useless, so every
  vectorized version is checked against the simple loop with `np.allclose`.
- **Use the minimum of several runs.** Slower runs mostly measure noise from other processes.
- **Tradeoff:** the vectorized code uses more memory (it needs the whole array at once). For data that does
  not fit in memory, process it in chunks instead (see the streaming chunks note).

## Run

```bash
pip install numpy pytest
python python/vectorization-benchmark/bench.py
pytest python/vectorization-benchmark
```
