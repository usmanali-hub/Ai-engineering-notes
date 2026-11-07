"""Feature selection compared: filter, wrapper and embedded methods, scored and checked against known truth.

The synthetic dataset has 300 rows and 100 columns: columns 0-4 carry the signal, columns 5-9 are redundant linear
mixtures of them, and columns 10-99 are pure noise, so what each method picked can be checked. Selection always runs
*inside* the cross-validation pipeline, fitted on training folds only (selecting on all rows first is the leak shown in
the leakage lab). Stability is how similar the chosen set is across bootstrap resamples (Jaccard, 1 = identical).
"""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 3
METHODS = ("none", "anova", "mutual_info", "rfe", "lasso", "forest")
INFORMATIVE = range(5)
REDUNDANT = range(5, 10)


SEEDS = tuple(range(3, 13))  # ten independently drawn datasets


def synthetic(seed: int = SEED) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Informative columns first (shuffle=False), then redundant, then noise."""
    X, y = make_classification(
        n_samples=300, n_features=100, n_informative=5, n_redundant=5, n_repeated=0, class_sep=1.0, shuffle=False, random_state=seed
    )
    return X, y


def selector(method: str, k: int, seed: int = SEED):
    if method == "none":
        return "passthrough"
    if method == "anova":
        return SelectKBest(f_classif, k=k)
    if method == "mutual_info":
        return SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=seed), k=k)
    if method == "rfe":
        return RFE(LogisticRegression(max_iter=2000), n_features_to_select=k, step=0.1)
    if method == "lasso":
        return SelectFromModel(Lasso(alpha=0.02, max_iter=5000), max_features=k, threshold=-np.inf)
    if method == "forest":
        return SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1), max_features=k, threshold=-np.inf)
    raise ValueError(f"unknown method {method!r}")


def build(method: str, k: int, seed: int = SEED) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("select", selector(method, k, seed)), ("clf", LogisticRegression(max_iter=2000))]
    )


def selected(method: str, X: NDArray[np.float64], y: NDArray[np.int_], k: int, seed: int = SEED) -> NDArray[np.int_]:
    pipe = build(method, k, seed).fit(X, y)
    if method == "none":
        return np.arange(X.shape[1])
    mask = pipe.named_steps["select"].get_support()
    return np.flatnonzero(mask)


def cv_auc(method: str, X: NDArray[np.float64], y: NDArray[np.int_], k: int, repeats: int = 3) -> tuple[float, float]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    s = cross_val_score(build(method, k), X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std(ddof=1))


def stability(method: str, X: NDArray[np.float64], y: NDArray[np.int_], k: int, n: int = 15, seed: int = SEED) -> float:
    """Mean pairwise Jaccard similarity of the selected sets over bootstrap resamples."""
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        sets.append(set(selected(method, X[idx], y[idx], k).tolist()))
    return float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)]))


def recovery(chosen: NDArray[np.int_]) -> dict[str, int]:
    s = set(chosen.tolist())
    return {
        "informative_found": len(s & set(INFORMATIVE)),
        "redundant_found": len(s & set(REDUNDANT)),
        "noise_picked": len(s - set(INFORMATIVE) - set(REDUNDANT)),
    }


def compare_synthetic(k: int = 10, seed: int = SEED) -> pd.DataFrame:
    X, y = synthetic(seed)
    rows = []
    for m in METHODS:
        t0 = time.perf_counter()
        mean, std = cv_auc(m, X, y, k)
        seconds = time.perf_counter() - t0
        chosen = selected(m, X, y, k)
        rows.append(
            {"method": m, "auc": mean, "auc_std": std, **recovery(chosen), "stability": stability(m, X, y, k) if m != "none" else 1.0, "seconds": seconds}
        )
    return pd.DataFrame(rows).set_index("method")


def repeated_synthetic(k: int = 10, seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    """Mean over independently drawn synthetic datasets, plus how often each method beat using every column."""
    runs = pd.concat([compare_synthetic(k, s) for s in seeds], keys=list(seeds), names=["seed", "method"])
    mean = runs.groupby("method").mean().loc[list(METHODS)]
    auc = runs["auc"].unstack("method")
    mean["auc_sd_over_datasets"] = auc.std(ddof=1).loc[list(METHODS)]
    mean["beats_all_columns"] = [int((auc[m] > auc["none"]).sum()) if m != "none" else 0 for m in METHODS]
    mean["datasets"] = len(seeds)
    mean.attrs["auc_by_dataset"] = auc
    return mean


def paired_vs(auc: pd.DataFrame, reference: str = "anova") -> pd.DataFrame:
    """Per method, the AUC difference from ``reference`` on the same dataset: mean, standard error and how often it was higher."""
    out = []
    for m in METHODS:
        if m in (reference, "none"):
            continue
        d = auc[m] - auc[reference]
        out.append({"method": m, "mean_diff": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)), "wins": int((d > 0).sum()), "datasets": len(d)})
    return pd.DataFrame(out).set_index("method")


def compare_real(ks: tuple[int, ...] = (3, 5, 10)) -> pd.DataFrame:
    """Breast cancer data (30 real columns): CV AUC for each method at several set sizes. No ground truth here."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    rows = {m: {k: cv_auc(m, X, y, k)[0] for k in ks} for m in METHODS if m != "none"}
    out = pd.DataFrame(rows).T
    out.loc["none (all 30)"] = cv_auc("none", X, y, 30)[0]
    return out


def main() -> None:
    pd.set_option("display.width", 160)
    print(f"Synthetic: 5 informative, 5 redundant, 90 noise columns; choose 10; 300 rows; mean over {len(SEEDS)} datasets")
    table = repeated_synthetic()
    print(table.round(3).to_string())
    print(f"\nPaired against ANOVA on the same {len(SEEDS)} datasets (AUC difference):")
    print(paired_vs(table.attrs["auc_by_dataset"]).round(4).to_string())
    print("\nBreast cancer (30 real columns): CV ROC-AUC by number of selected columns")
    print(compare_real().round(3).to_string())


if __name__ == "__main__":
    main()
