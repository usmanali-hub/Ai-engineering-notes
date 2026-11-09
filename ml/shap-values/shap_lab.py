"""SHAP values from first principles, checked against LightGBM's built-in TreeSHAP.

A Shapley value shares one prediction among the features. For a row x and a feature i it is the average, over every
order in which features could be "revealed", of how much the model's output changes when i is revealed. "Revealed"
means taken from x; "not yet revealed" means taken from a randomly drawn *background* row (the interventional
definition). Enumerating all 2^d feature subsets gives the exact answer, feasible for d about 10 or fewer.

What the lab checks:

* efficiency: base value plus the contributions equals the model output for that row, exactly;
* a feature the model never uses gets exactly zero, and two interchangeable features get equal credit;
* LightGBM's ``pred_contrib`` (TreeSHAP, which uses the trees' own cover counts instead of a background sample)
  is close to the exact values here, with a slightly larger gap when one feature is a near-copy of another;
* when a feature has a near-copy, the credit for it is split between the two, in SHAP and in permutation importance
  alike, so neither number for either column alone says how much the underlying signal matters;
* a feature the real process ignores still gets credit if the fitted model happens to use it (the noise columns here).
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from scipy.stats import spearmanr

Predict = Callable[[NDArray[np.float64]], NDArray[np.float64]]  # rows -> one number per row (log-odds, not probability)
SEED = 3


def exact_shapley(predict: Predict, x: NDArray[np.float64], background: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    """Base value and the exact interventional Shapley value of each feature for one row ``x``."""
    d = len(x)
    value: dict[frozenset[int], float] = {}
    for size in range(d + 1):
        for subset in combinations(range(d), size):
            mixed = background.copy()
            idx = list(subset)
            mixed[:, idx] = x[idx]
            value[frozenset(subset)] = float(predict(mixed).mean())
    phi = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for size in range(d):
            weight = 1.0 / (d * comb(d - 1, size))
            for subset in combinations(others, size):
                s = frozenset(subset)
                phi[i] += weight * (value[s | {i}] - value[s])
    return value[frozenset()], phi


def shap_matrix(predict: Predict, rows: NDArray[np.float64], background: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    base = 0.0
    out = np.zeros(rows.shape)
    for r, x in enumerate(rows):
        base, out[r] = exact_shapley(predict, x, background)
    return base, out


def make_data(n: int = 2000, correlated_copy: bool = False, seed: int = SEED) -> tuple[NDArray[np.float64], NDArray[np.int_], list[str]]:
    """Six features; the outcome depends on x0, x1 (with an interaction) and x2, and not on x3-x5.
    With ``correlated_copy`` the last column is replaced by a near-copy of x0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = 1.5 * X[:, 0] + 1.0 * X[:, 1] + 0.8 * X[:, 0] * X[:, 1] + 0.5 * (X[:, 2] > 0.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    names = [f"x{i}" for i in range(6)]
    if correlated_copy:
        X[:, 5] = X[:, 0] + 0.1 * rng.normal(size=n)
        names[5] = "x0_copy"
    return X, y, names


def fit(X: NDArray[np.float64], y: NDArray[np.int_]) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=80, num_leaves=8, learning_rate=0.1, min_child_samples=20, random_state=SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1).fit(X, y)


def raw_predict(model: LGBMClassifier) -> Predict:
    return lambda rows: np.asarray(model.predict(rows, raw_score=True), dtype=float)


def treeshap(model: LGBMClassifier, rows: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    """LightGBM's own contributions: one column per feature plus a final base-value column."""
    contrib = np.asarray(model.predict(rows, pred_contrib=True), dtype=float)
    return float(contrib[0, -1]), contrib[:, :-1]


def permutation_importance(model: LGBMClassifier, X: NDArray[np.float64], y: NDArray[np.int_], seed: int = SEED, repeats: int = 5) -> NDArray[np.float64]:
    """Fall in accuracy when one column is shuffled, averaged over ``repeats`` shuffles (evaluated on the rows given)."""
    rng = np.random.default_rng(seed)
    base = float((model.predict(X) == y).mean())
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(repeats):
            Z = X.copy()
            Z[:, j] = rng.permutation(Z[:, j])
            drops.append(base - float((model.predict(Z) == y).mean()))
        out[j] = np.mean(drops)
    return out


def compare(correlated_copy: bool, n_explain: int = 60, n_background: int = 80) -> dict[str, object]:
    X, y, names = make_data(correlated_copy=correlated_copy)
    train, test = slice(0, 1500), slice(1500, 2000)
    model = fit(X[train], y[train])
    predict = raw_predict(model)
    background = X[train][:n_background]
    rows = X[test][:n_explain]
    base, exact = shap_matrix(predict, rows, background)
    _, tree = treeshap(model, rows)
    efficiency = float(np.abs(base + exact.sum(axis=1) - predict(rows)).max())
    perm = permutation_importance(model, X[test], y[test])
    mean_abs = np.abs(exact).mean(axis=0)
    return {
        "names": names,
        "efficiency_error": efficiency,
        "mean_abs_exact": mean_abs,
        "mean_abs_tree": np.abs(tree).mean(axis=0),
        "row_correlation": float(np.corrcoef(exact.ravel(), tree.ravel())[0, 1]),
        "mean_abs_difference": float(np.abs(exact - tree).mean()),
        "permutation": perm,
        "rank_agreement": float(spearmanr(mean_abs, perm).statistic),
    }


def table(result: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean |SHAP| exact": result["mean_abs_exact"],
            "mean |SHAP| TreeSHAP": result["mean_abs_tree"],
            "permutation (accuracy drop)": result["permutation"],
        },
        index=result["names"],  # type: ignore[arg-type]
    )


def main() -> None:
    pd.set_option("display.width", 140)
    for label, flag in (("independent features", False), ("x5 replaced by a near-copy of x0", True)):
        r = compare(flag)
        print(f"\n{label}")
        print(table(r).round(3).to_string())
        print(
            f"efficiency error {r['efficiency_error']:.2e}; exact vs TreeSHAP: correlation {r['row_correlation']:.3f}, "
            f"mean absolute difference {r['mean_abs_difference']:.3f}; rank agreement SHAP vs permutation {r['rank_agreement']:.2f}"
        )


if __name__ == "__main__":
    main()
