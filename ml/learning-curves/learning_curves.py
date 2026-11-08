"""Learning curves: is a model limited by bias or variance, and would more data help?

A learning curve trains the same model on growing slices of the training data and records the score on the rows it
was trained on and the score on held-out rows (5-fold cross-validation). Reading the two curves together:

* a large, persistent gap between training and validation score means the model is fitting noise (high variance);
* a small gap with a low validation score means the model is too simple for the pattern (high bias);
* a validation curve still rising at full size means more data would help; a flat one means it would not.

Two datasets are used: a noisy synthetic problem with a nonlinear boundary (so a straight-line model is too simple
and an unlimited tree is too flexible) and the real handwritten digits data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_digits, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 11
SIZES = (0.1, 0.25, 0.5, 0.75, 1.0)
GAP_LIMIT = 0.08  # training minus validation score above this counts as overfitting
BIAS_MARGIN = 0.03  # validation score this far below the best model's counts as underfitting

MODELS = {
    "logistic (simple)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "tree, depth 3": lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED),
    "tree, unlimited": lambda: DecisionTreeClassifier(random_state=SEED),
    "forest, 100 trees": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
}


def noisy_nonlinear(n: int = 3000, seed: int = SEED) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Binary problem with two clusters per class (not linearly separable) and 8% label noise."""
    return make_classification(
        n_samples=n, n_features=12, n_informative=6, n_redundant=0, n_clusters_per_class=3, flip_y=0.08, class_sep=0.9, random_state=seed
    )


def digits() -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    data = load_digits()
    return data.data, data.target


DATASETS = {"noisy synthetic": noisy_nonlinear, "handwritten digits": digits}


@dataclass(frozen=True)
class Curve:
    dataset: str
    model: str
    sizes: NDArray[np.int_]
    train: NDArray[np.float64]  # mean accuracy on the rows trained on, per size
    validation: NDArray[np.float64]
    validation_std: NDArray[np.float64]


def curve(dataset: str, model: str, sizes: tuple[float, ...] = SIZES) -> Curve:
    X, y = DATASETS[dataset]()
    n, tr, va = learning_curve(
        MODELS[model](), X, y, train_sizes=np.array(sizes), cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy", n_jobs=1
    )[:3]
    return Curve(dataset, model, n, tr.mean(axis=1), va.mean(axis=1), va.std(axis=1, ddof=1))


def diagnose(c: Curve, best_validation: float) -> str:
    """Rule of thumb from the end of the curve (see GAP_LIMIT and BIAS_MARGIN)."""
    gap = float(c.train[-1] - c.validation[-1])
    if gap > GAP_LIMIT:
        return "overfitting (variance)"
    if float(c.validation[-1]) < best_validation - BIAS_MARGIN:
        return "underfitting (bias)"
    return "about right"


def would_more_data_help(c: Curve) -> float:
    """Validation gain from using the full training slice instead of half of it (the last doubling)."""
    half = int(np.argmin(np.abs(c.sizes - c.sizes[-1] / 2)))
    return float(c.validation[-1] - c.validation[half])


def summary(dataset: str, curves: dict[str, Curve] | None = None) -> pd.DataFrame:
    curves = curves or {m: curve(dataset, m) for m in MODELS}
    best = max(float(c.validation[-1]) for c in curves.values())
    rows = []
    for m, c in curves.items():
        rows.append(
            {
                "model": m,
                "train_score": float(c.train[-1]),
                "validation_score": float(c.validation[-1]),
                "gap": float(c.train[-1] - c.validation[-1]),
                "gain_from_last_doubling": would_more_data_help(c),
                "diagnosis": diagnose(c, best),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def plot(path: Path, curves_by_dataset: dict[str, dict[str, Curve]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(curves_by_dataset), len(MODELS), figsize=(4 * len(MODELS), 3.4 * len(curves_by_dataset)), sharey="row", squeeze=False)
    for r, (ds, cs) in enumerate(curves_by_dataset.items()):
        for k, (m, c) in enumerate(cs.items()):
            ax = axes[r][k]
            ax.plot(c.sizes, c.train, marker="o", label="training")
            ax.plot(c.sizes, c.validation, marker="o", label="validation")
            ax.fill_between(c.sizes, c.validation - c.validation_std, c.validation + c.validation_std, alpha=0.2)
            ax.set_title(f"{m}" if r == 0 else "", fontsize=10)
            ax.set_xlabel("training rows")
            if k == 0:
                ax.set_ylabel(f"{ds}\naccuracy")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    pd.set_option("display.width", 160)
    all_curves = {ds: {m: curve(ds, m) for m in MODELS} for ds in DATASETS}
    for ds, cs in all_curves.items():
        print(f"\n{ds}: accuracy at the full training slice (5-fold CV)")
        print(summary(ds, cs).round(3).to_string())
    plot(Path(__file__).with_name("learning_curves.png"), all_curves)


if __name__ == "__main__":
    main()
