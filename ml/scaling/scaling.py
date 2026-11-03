"""Feature scaling: which models care, and by how much.

Distance-based models (KNN) and margin/kernel models (SVM) treat every feature as part of one geometry, so
a feature measured in thousands drowns out one measured in fractions. Tree ensembles split on one feature
at a time and are unaffected. Every comparison here puts the scaler inside a pipeline, so each fold learns
its scaling from its own training rows only.
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC

SCALERS: dict[str, type[BaseEstimator] | None] = {
    "none": None,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def models() -> dict[str, BaseEstimator]:
    return {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm_rbf": SVC(kernel="rbf"),
        "logistic": LogisticRegression(max_iter=1000),
        "forest": RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1),
    }


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    loader = {"breast_cancer": load_breast_cancer, "wine": load_wine}[name]
    X, y = loader(return_X_y=True)
    return np.asarray(X, dtype=float), np.asarray(y)


def with_scaler(model: BaseEstimator, scaler: str) -> BaseEstimator:
    cls = SCALERS[scaler]
    step = FunctionTransformer() if cls is None else cls()  # an identity step keeps the pipeline shape the same
    return make_pipeline(step, model)


def compare(X: np.ndarray, y: np.ndarray, repeats: int = 3, seed: int = 0) -> dict[str, dict[str, float]]:
    """Mean cross-validated accuracy for every (model, scaler) pair, on identical folds."""
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # unscaled logistic regression often stops at its limit
        return {
            name: {s: float(cross_val_score(with_scaler(est, s), X, y, cv=folds).mean()) for s in SCALERS}
            for name, est in models().items()
        }


def distance_share(X: np.ndarray, pairs: int = 2000, seed: int = 0) -> tuple[int, float]:
    """Which feature dominates Euclidean distance, and what share of squared distance it supplies.

    Random pairs of rows are drawn; the share is the largest per-feature mean squared difference over the
    total. With equal influence among p features it would be 1/p.
    """
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(X), pairs), rng.integers(0, len(X), pairs)
    per_feature = ((X[a] - X[b]) ** 2).mean(axis=0)
    top = int(np.argmax(per_feature))
    return top, float(per_feature[top] / per_feature.sum())


def scale_trap(n: int = 600, seed: int = 0) -> dict[str, float]:
    """Two informative features measured in small units and one pure-noise feature measured in large ones.

    Unscaled, the noise feature supplies almost all of the distance, so KNN neighbours are random.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    informative = rng.normal(0, 1, (n, 2)) + y[:, None] * 2.0
    noise = rng.normal(0, 1, (n, 1)) * 1000  # e.g. a salary column next to two ratios
    X = np.hstack([informative, noise])
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    knn = KNeighborsClassifier(n_neighbors=5)
    return {
        "unscaled": float(cross_val_score(with_scaler(knn, "none"), X, y, cv=folds).mean()),
        "standard": float(cross_val_score(with_scaler(knn, "standard"), X, y, cv=folds).mean()),
        "noise_share_of_distance": distance_share(X)[1],
    }


def with_outliers(X: np.ndarray, fraction: float = 0.05, factor: float = 20.0, seed: int = 0) -> np.ndarray:
    """Blow up a few values per column, as a data-entry error or a heavy tail might."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    hit = rng.random(X.shape) < fraction
    out[hit] = out[hit] * factor
    return out


def summary(repeats: int = 3) -> str:
    lines = []
    for name in ("breast_cancer", "wine"):
        X, y = load(name)
        top, share = distance_share(X)
        lines.append(f"\n{name}: {X.shape[0]} rows, {X.shape[1]} features; feature {top} supplies {share:.1%} of squared distance (equal share would be {1 / X.shape[1]:.1%})")
        lines.append(f"{'model':<10}" + "".join(f"{s:>10}" for s in SCALERS))
        for model, row in compare(X, y, repeats).items():
            lines.append(f"{model:<10}" + "".join(f"{row[s]:>10.3f}" for s in SCALERS))
    X, y = load("breast_cancer")
    lines.append("\nbreast_cancer with 5% of values multiplied by 20 (outliers):")
    lines.append(f"{'model':<10}" + "".join(f"{s:>10}" for s in SCALERS))
    for model, row in compare(with_outliers(X), y, repeats).items():
        lines.append(f"{model:<10}" + "".join(f"{row[s]:>10.3f}" for s in SCALERS))
    t = scale_trap()
    lines.append(f"\nscale trap (2 informative + 1 noise feature in thousands): KNN unscaled {t['unscaled']:.3f}, standardised {t['standard']:.3f}; noise supplies {t['noise_share_of_distance']:.1%} of distance")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
