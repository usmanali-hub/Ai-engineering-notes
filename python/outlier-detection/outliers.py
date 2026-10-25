"""Two ways to flag outliers: z-score and IQR fences, and why they disagree."""

import numpy as np


def zscore_outliers(x: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """True where |x - mean| / std exceeds ``threshold``. A constant column has no outliers."""
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std == 0:
        return np.zeros(x.shape, dtype=bool)
    return np.abs(x - x.mean()) / std > threshold


def iqr_outliers(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    """True outside [Q1 - k*IQR, Q3 + k*IQR]. Quartiles ignore the extremes, so they resist them."""
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def modified_zscore_outliers(x: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Robust z-score from the median and MAD (0.6745 rescales MAD to match a normal's std)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros(x.shape, dtype=bool)
    return np.abs(0.6745 * (x - med) / mad) > threshold


if __name__ == "__main__":
    data = np.concatenate([np.random.default_rng(0).normal(50, 5, 200), np.linspace(150, 180, 30)])
    for name, fn in [("z-score", zscore_outliers), ("IQR", iqr_outliers), ("modified z", modified_zscore_outliers)]:
        print(f"{name:<11} flags {int(fn(data).sum())} of {len(data)}")
