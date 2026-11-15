"""Reproducibility: what a seed does and does not control, and how to tell that the data changed.

Three questions, each answered by measurement:

* How much does one reported number depend on the seed? (split seed, model seed, both)
* If you try many seeds and report the best, how much does that flatter you? (selection on the test set versus an honest
  protocol that selects on validation data and reports on a second, untouched test set)
* Can you detect that the data changed? A content hash of the dataset (sensitive to edits, and either sensitive or
  insensitive to row order, on purpose) goes into a run manifest next to the seed and library versions.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 17
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def datasets() -> dict[str, tuple[pd.DataFrame, IntArray]]:
    bc = load_breast_cancer(as_frame=True)
    X, y = make_classification(n_samples=500, n_features=12, n_informative=4, flip_y=0.1, random_state=SEED)
    noisy = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return {"breast cancer": (bc.data, np.asarray(bc.target, dtype=int)), "noisy synthetic": (noisy, np.asarray(y, dtype=int))}


def forest(seed: int, n_jobs: int = 1) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=60, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs)


def test_accuracy(X: pd.DataFrame, y: IntArray, split_seed: int, model_seed: int) -> float:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split_seed)
    return float(accuracy_score(yte, forest(model_seed).fit(Xtr, ytr).predict(Xte)))


def seed_spread(X: pd.DataFrame, y: IntArray, seeds: range = range(30)) -> pd.DataFrame:
    """Test accuracy over ``seeds`` when only the split, only the model, or both change."""
    rows = []
    for label, fn in (
        ("split seed only", lambda s: test_accuracy(X, y, s, 0)),
        ("model seed only", lambda s: test_accuracy(X, y, 0, s)),
        ("both", lambda s: test_accuracy(X, y, s, s)),
    ):
        acc = np.array([fn(s) for s in seeds])
        rows.append({"varied": label, "mean": acc.mean(), "std": acc.std(ddof=1), "min": acc.min(), "max": acc.max()})
    return pd.DataFrame(rows)


def best_of_n_inflation(X: pd.DataFrame, y: IntArray, n: int = 10, trials: int = 40, seed: int = SEED) -> dict[str, float]:
    """The cost of 'try n seeds and report the best'.

    Each trial draws ``n`` seeds. *Naive*: the best test accuracy among them is reported. *Honest*: the seed is chosen on a
    validation split, and the chosen model's accuracy is reported on a second test split that took no part in choosing.
    Compared with the average over all ``n`` seeds on that same second test split, which is what a random seed gives.
    """
    rng = np.random.default_rng(seed)
    Xdev, Xfinal, ydev, yfinal = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    naive_gain, honest_gain = [], []
    for _ in range(trials):
        seeds = [int(s) for s in rng.integers(0, 10**6, n)]
        scores = []
        for s in seeds:
            Xtr, Xte, ytr, yte = train_test_split(Xdev, ydev, test_size=0.3, stratify=ydev, random_state=s)
            model = forest(s).fit(Xtr, ytr)
            scores.append((float(accuracy_score(yte, model.predict(Xte))), float(accuracy_score(yfinal, model.predict(Xfinal)))))
        dev, fin = np.array(scores).T
        naive_gain.append(dev.max() - dev.mean())  # what the best-seed report claims over a typical seed
        honest_gain.append(fin[int(dev.argmax())] - fin.mean())  # what the chosen seed really gains on untouched data
    return {"claimed_gain": float(np.mean(naive_gain)), "real_gain": float(np.mean(honest_gain)), "real_gain_se": float(np.std(honest_gain, ddof=1) / np.sqrt(trials))}


def identical_runs(X: pd.DataFrame, y: IntArray) -> dict[str, float | bool]:
    """Bitwise and approximate agreement of predicted probabilities across repeated fits."""
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    a = forest(5).fit(Xtr, ytr).predict_proba(Xte)
    threaded = forest(5, n_jobs=2).fit(Xtr, ytr).predict_proba(Xte)
    return {
        "same seed, rerun: bitwise equal": bool(np.array_equal(a, forest(5).fit(Xtr, ytr).predict_proba(Xte))),
        "same seed, 1 vs 2 threads: bitwise equal": bool(np.array_equal(a, threaded)),
        "same seed, 1 vs 2 threads: max difference": float(np.abs(a - threaded).max()),
        "different seed: max difference": float(np.abs(a - forest(6).fit(Xtr, ytr).predict_proba(Xte)).max()),
    }


def fingerprint(df: pd.DataFrame, ignore_row_order: bool = False) -> str:
    """SHA-256 of the dataset's content: column names, dtypes and every cell. Row order counts unless ignored."""
    cols = hashlib.sha256(json.dumps([[str(c), str(t)] for c, t in df.dtypes.items()]).encode()).digest()
    rows = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)
    if ignore_row_order:
        rows = np.sort(rows)
    return hashlib.sha256(cols + rows.tobytes()).hexdigest()[:16]


def edit_cells(df: pd.DataFrame, fraction: float, seed: int = SEED) -> pd.DataFrame:
    """A copy with ``fraction`` of the values in the first numeric column nudged by 1%: a quiet upstream data change."""
    out = df.copy()
    col = out.columns[0]
    idx = np.random.default_rng(seed).choice(len(out), max(1, int(fraction * len(out))), replace=False)
    out.iloc[idx, 0] = out.iloc[idx, 0] * 1.01
    out[col] = out[col].astype(float)
    return out


@dataclass(frozen=True)
class Manifest:
    data_hash: str
    seed: int
    rows: int
    columns: int
    python: str
    scikit_learn: str
    numpy: str
    pandas: str

    @property
    def run_id(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()[:12]


def manifest(df: pd.DataFrame, seed: int) -> Manifest:
    return Manifest(fingerprint(df), seed, len(df), df.shape[1], platform.python_version(), sklearn.__version__, np.__version__, pd.__version__)


def change_report(X: pd.DataFrame, y: IntArray) -> pd.DataFrame:
    """Which kinds of change a fingerprint notices, and what the model score did."""
    shuffled = X.sample(frac=1.0, random_state=1)
    variants = {
        "unchanged": X,
        "rows reordered": shuffled,
        "2% of one column edited by 1%": edit_cells(X, 0.02),
        "one column renamed": X.rename(columns={X.columns[0]: "renamed"}),
        "one row dropped": X.iloc[1:],
    }
    base = fingerprint(X)
    rows = []
    for name, v in variants.items():
        yy = y[v.index.to_numpy()] if name != "one row dropped" else y[1:]
        rows.append(
            {
                "change": name,
                "hash changes": fingerprint(v) != base,
                "hash changes (ignoring row order)": fingerprint(v, True) != fingerprint(X, True),
                "accuracy": test_accuracy(v.reset_index(drop=True), yy, SEED, SEED),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for name, (X, y) in datasets().items():
        print(f"\n== {name} ==")
        print(seed_spread(X, y).round(4).to_string(index=False))
        print("best of 10 seeds:", {k: round(v, 4) for k, v in best_of_n_inflation(X, y).items()})
        print(identical_runs(X, y))
        print(change_report(X, y).round(4).to_string(index=False))
