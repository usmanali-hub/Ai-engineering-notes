"""Class imbalance: do nothing vs class weights vs SMOTE, on data with a known rare positive class.

Each method is scored by PR-AUC (ranking quality, unaffected by the cutoff) and by the best F1 reachable by moving the
cutoff. Resampling is done inside each training fold only, via an imblearn pipeline, so synthetic rows never leak into
validation. Two models (logistic regression and gradient boosting) are run on several independently drawn datasets so
differences can be compared with their spread across datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 7
METHODS = ("plain", "class_weight", "smote")
MODELS = ("logistic", "boosting")
SEEDS = tuple(range(7, 17))  # ten independently drawn datasets


def make_data(rate: float = 0.03, n: int = 6000, seed: int = SEED) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    X, y = make_classification(
        n_samples=n, n_features=12, n_informative=5, n_redundant=2, weights=[1 - rate], flip_y=0.01, class_sep=0.9, random_state=seed
    )
    return X, y


def build(method: str, model: str = "logistic") -> Pipeline | ImbPipeline:
    weight = "balanced" if method == "class_weight" else None
    clf: LogisticRegression | HistGradientBoostingClassifier
    if model == "boosting":
        clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=8, class_weight=weight, random_state=SEED)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight=weight)
    if method == "smote":
        return ImbPipeline([("scale", StandardScaler()), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def best_f1(y: NDArray[np.int_], s: NDArray[np.float64]) -> float:
    p, r, _ = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(f.max())


def out_of_fold(method: str, X: NDArray[np.float64], y: NDArray[np.int_], folds: int = 5, model: str = "logistic") -> NDArray[np.float64]:
    scores = np.zeros(len(y))
    for fit, val in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        scores[val] = build(method, model).fit(X[fit], y[fit]).predict_proba(X[val])[:, 1]
    return scores


def compare(rate: float = 0.03, model: str = "logistic", seed: int = SEED) -> pd.DataFrame:
    X, y = make_data(rate, seed=seed)
    rows = []
    for m in METHODS:
        s = out_of_fold(m, X, y, model=model)
        pred = s >= 0.5
        rows.append(
            {
                "method": m,
                "pr_auc": average_precision_score(y, s),
                "recall_at_0.5": recall_score(y, pred),
                "flagged_at_0.5": int(pred.sum()),
                "mean_score": float(s.mean()),
                "best_f1_any_cutoff": best_f1(y, s),
            }
        )
    out = pd.DataFrame(rows).set_index("method")
    out.attrs["prevalence"] = float(y.mean())
    return out


def repeated(rate: float = 0.03, model: str = "logistic", seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    """Mean and standard deviation over independently drawn datasets, plus how often each fix beat plain on PR-AUC."""
    runs = [compare(rate, model, s) for s in seeds]
    stack = pd.concat(runs, keys=list(seeds), names=["seed", "method"])
    mean = stack.groupby("method").mean().loc[list(METHODS)]
    std = stack.groupby("method").std(ddof=1).loc[list(METHODS)]
    out = mean.add_suffix("_mean").join(std[["pr_auc", "best_f1_any_cutoff"]].add_suffix("_std"))
    pr = stack["pr_auc"].unstack("method")
    out["pr_auc_wins_vs_plain"] = [int((pr[m] > pr["plain"]).sum()) if m != "plain" else 0 for m in METHODS]
    out["datasets"] = len(seeds)
    return out


def main() -> None:
    pd.set_option("display.width", 200)
    for model in MODELS:
        for rate in (0.03, 0.15):
            t = repeated(rate, model)
            print(f"\n{model}, positive rate about {rate:.0%}: mean over {len(SEEDS)} datasets of 6,000 rows (out-of-fold)")
            print(t.round(3).to_string())


if __name__ == "__main__":
    main()
