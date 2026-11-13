"""PCA for visualisation: what the two plotted axes keep, what they lose, and when they mislead.

PCA rotates the data so the first axis holds the most variance, the second the most of what is left, and so on. Plotting
the first two axes is the usual way to look at high-dimensional data. The lab measures four things that decide
whether that picture can be trusted:

* how much variance the first components hold (the scree), and how many are needed for 90% and 95%
* whether feature scaling changes the picture (PCA is driven by variance, so a column with big units dominates)
* how much of the class structure survives in two dimensions (cross-validated kNN accuracy, with scaling and PCA fitted
  inside each training fold so the check does not leak) and how well local neighbourhoods are preserved (trustworthiness)
* the trap: PCA keeps *variance*, not *class information*. When the groups differ along a low-variance direction, the
  top components throw that direction away, while a supervised projection (LDA) keeps it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 3
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def datasets() -> dict[str, tuple[FloatArray, IntArray, list[str]]]:
    """name -> (features, labels, feature names)."""
    out = {}
    for name, loader in (("iris", load_iris), ("wine", load_wine), ("digits", load_digits)):
        data = loader()
        out[name] = (np.asarray(data.data, dtype=float), np.asarray(data.target, dtype=int), list(data.feature_names))
    return out


def scree(X: FloatArray, scale: bool = True) -> FloatArray:
    """Explained-variance ratio of every component (sums to 1). ``scale`` standardises each column first."""
    Z = StandardScaler().fit_transform(X) if scale else X
    return np.asarray(PCA().fit(Z).explained_variance_ratio_, dtype=float)


def components_for(ratio: FloatArray, target: float) -> int:
    """Fewest leading components whose cumulative variance reaches ``target`` (a fraction, 0 < target <= 1)."""
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    return int(np.searchsorted(np.cumsum(ratio), target - 1e-12) + 1)


def top_feature_share(X: FloatArray, scale: bool) -> tuple[int, float]:
    """The column with the largest squared loading on PC1 and the share of PC1 it carries (squared loadings sum to 1)."""
    Z = StandardScaler().fit_transform(X) if scale else X
    loadings = PCA(n_components=1).fit(Z).components_[0] ** 2
    return int(np.argmax(loadings)), float(loadings.max())


def knn_accuracy(X: FloatArray, y: IntArray, n_components: int | None, seed: int = SEED) -> float:
    """5-fold CV accuracy of 5-NN on standardised features, projected to ``n_components`` (None = no projection).

    Scaling and PCA are fitted on each training fold only.
    """
    steps = [StandardScaler()] + ([PCA(n_components=n_components)] if n_components else []) + [KNeighborsClassifier(5)]
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(make_pipeline(*steps), X, y, cv=folds).mean())


def random_pair_accuracy(X: FloatArray, y: IntArray, seed: int = SEED, draws: int = 10) -> float:
    """Mean 5-NN accuracy using two randomly chosen original columns: what a 2D view costs without PCA."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        cols = rng.choice(X.shape[1], size=2, replace=False)
        scores.append(knn_accuracy(X[:, cols], y, None, seed))
    return float(np.mean(scores))


def projection_quality(X: FloatArray, y: IntArray, seed: int = SEED) -> dict[str, float]:
    """How much class structure and local neighbourhood structure a 2D PCA view keeps."""
    Z = StandardScaler().fit_transform(X)
    embedded = PCA(n_components=2).fit_transform(Z)  # unsupervised and label-free, so this is not a leakage risk
    return {
        "variance_in_2d": float(scree(X)[:2].sum()),
        "acc_all_features": knn_accuracy(X, y, None, seed),
        "acc_pca_2d": knn_accuracy(X, y, 2, seed),
        "acc_two_random_columns": random_pair_accuracy(X, y, seed),
        "trustworthiness_2d": float(trustworthiness(Z, embedded, n_neighbors=10)),
    }


def variance_trap(n: int = 400, seed: int = SEED) -> dict[str, float]:
    """Two classes that differ only along a low-variance axis, hidden under a high-variance nuisance axis.

    Column 0 has standard deviation 10 and carries no class information. Column 1 has standard deviation 0.5 and
    separates the classes by 4 units (8 standard deviations). Returns CV accuracy of 5-NN after a 1D projection by
    PCA (which keeps column 0) and by LDA (which uses the labels and keeps column 1), plus 2D PCA.
    """
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = np.column_stack([rng.normal(0, 10, n), rng.normal(0, 0.5, n) + 4.0 * y])
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)

    def acc(projector: object) -> float:
        return float(cross_val_score(make_pipeline(projector, KNeighborsClassifier(5)), X, y, cv=folds).mean())  # type: ignore[arg-type]

    return {
        "pca_1d": acc(PCA(n_components=1)),
        "lda_1d": acc(LinearDiscriminantAnalysis(n_components=1)),
        "pca_2d": acc(PCA(n_components=2)),
        "variance_share_pc1": float(PCA().fit(X).explained_variance_ratio_[0]),
    }


def reconstruction_error(X: FloatArray, ks: tuple[int, ...]) -> pd.DataFrame:
    """Mean squared error of rebuilding the (centred) data from k components, and the share of variance lost."""
    rows = []
    centred = X - X.mean(axis=0)
    for k in ks:
        pca = PCA(n_components=k).fit(X)
        rebuilt = pca.inverse_transform(pca.transform(X))
        rows.append({"k": k, "mse": float(((X - rebuilt) ** 2).mean()), "variance_lost": float(1 - pca.explained_variance_ratio_.sum())})
    out = pd.DataFrame(rows)
    out["relative_mse"] = out["mse"] / float((centred**2).mean())
    return out


@dataclass
class Results:
    scree: pd.DataFrame
    scaling: pd.DataFrame
    quality: pd.DataFrame
    trap: dict[str, float]
    reconstruction: pd.DataFrame


def run() -> Results:
    data = datasets()
    scree_rows, scaling_rows, quality_rows = [], [], []
    for name, (X, y, features) in data.items():
        ratio = scree(X)
        scree_rows.append(
            {"dataset": name, "features": X.shape[1], "pc1": ratio[0], "pc1_pc2": ratio[:2].sum(), "k90": components_for(ratio, 0.90), "k95": components_for(ratio, 0.95)}
        )
        for scaled in (False, True):
            col, share = top_feature_share(X, scaled)
            scaling_rows.append({"dataset": name, "scaled": scaled, "pc1_variance": scree(X, scaled)[0], "top_feature": features[col], "top_feature_share": share})
        quality_rows.append({"dataset": name, **projection_quality(X, y)})
    return Results(
        pd.DataFrame(scree_rows),
        pd.DataFrame(scaling_rows),
        pd.DataFrame(quality_rows),
        variance_trap(),
        reconstruction_error(data["digits"][0], (2, 5, 10, 20, 40)),
    )


def plot(path: Path) -> None:
    """Save scatter plots of the first two components for the three datasets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, (X, y, _)) in zip(axes, datasets().items(), strict=True):
        Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
        ax.scatter(Z[:, 0], Z[:, 1], c=y, s=8, cmap="tab10")
        ax.set_title(f"{name}: first two components")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 200)
    for title, table in (("Scree", r.scree), ("Scaling", r.scaling), ("Quality of a 2D view", r.quality), ("Digits reconstruction", r.reconstruction)):
        print(f"\n{title}\n{table.round(3).to_string(index=False)}")
    print("\nVariance trap", {k: round(v, 3) for k, v in r.trap.items()})
