"""k-means and the silhouette score: how to choose k, and when the answer is misleading.

For each dataset the lab sweeps k and records the inertia (within-cluster sum of squares, which always falls as k
grows), the mean silhouette score (how much closer points are to their own cluster than to the next nearest one, from
-1 to 1), and, where the true grouping is known, the adjusted Rand index (ARI, 1 = identical to the truth, about 0 =
no better than chance). It then checks three things that decide whether to trust the result:

* does the silhouette pick the true number of groups? (blobs: yes; touching blobs and two moons: not reliably)
* does feature scaling change the clustering? (it must, because k-means is built on Euclidean distance)
* is the clustering stable across random initialisations, or does it depend on the seed?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_wine, make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 5
KS = tuple(range(2, 9))


def datasets() -> dict[str, tuple[NDArray[np.float64], NDArray[np.int_], int]]:
    """name -> (features, true labels, true number of groups)."""
    corners = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0]])  # explicit centres: random ones can land close together
    separated = make_blobs(n_samples=600, centers=corners, cluster_std=0.8, random_state=SEED)
    touching = make_blobs(n_samples=600, centers=corners * 0.5, cluster_std=1.3, random_state=SEED)
    moons = make_moons(n_samples=600, noise=0.06, random_state=SEED)
    iris, wine = load_iris(return_X_y=True), load_wine(return_X_y=True)
    return {
        "4 separated blobs": (separated[0], separated[1], 4),
        "4 touching blobs": (touching[0], touching[1], 4),
        "two moons": (moons[0], moons[1], 2),
        "iris (3 species)": (iris[0], iris[1], 3),
        "wine (3 cultivars)": (wine[0], wine[1], 3),
    }


def fit(X: NDArray[np.float64], k: int, seed: int = SEED, n_init: int = 10) -> KMeans:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def sweep(X: NDArray[np.float64], y: NDArray[np.int_] | None = None, ks: tuple[int, ...] = KS) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = fit(X, k)
        row = {"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(X, km.labels_))}
        if y is not None:
            row["ari"] = float(adjusted_rand_score(y, km.labels_))
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def best_k_by_silhouette(table: pd.DataFrame) -> int:
    return int(table["silhouette"].idxmax())


def elbow_k(table: pd.DataFrame) -> int:
    """The k where the inertia curve is furthest below the straight line joining its end points (a standard
    'knee' heuristic). It is a convenience, not a criterion: flat or smooth curves have no clear knee."""
    k = table.index.to_numpy(dtype=float)
    inertia = table["inertia"].to_numpy(dtype=float)
    x = (k - k[0]) / (k[-1] - k[0])
    y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    return int(k[int(np.argmax((1 - x) - y))])


def scaling_effect(X: NDArray[np.float64], y: NDArray[np.int_], k: int) -> dict[str, float]:
    raw = adjusted_rand_score(y, fit(X, k).labels_)
    scaled = adjusted_rand_score(y, fit(StandardScaler().fit_transform(X), k).labels_)
    return {"ari_raw": float(raw), "ari_scaled": float(scaled)}


def stability(X: NDArray[np.float64], k: int, seeds: int = 10) -> dict[str, float]:
    """Run k-means with one initialisation per seed and compare the runs: ARI between every pair of seeds, and the
    spread of the inertia. n_init=1 on purpose, to expose dependence on the starting centres."""
    runs = [KMeans(n_clusters=k, n_init=1, init="random", random_state=s).fit(X) for s in range(seeds)]
    aris = [adjusted_rand_score(runs[i].labels_, runs[j].labels_) for i in range(seeds) for j in range(i + 1, seeds)]
    inertia = np.array([r.inertia_ for r in runs])
    return {"min_pairwise_ari": float(min(aris)), "mean_pairwise_ari": float(np.mean(aris)), "inertia_spread": float((inertia.max() - inertia.min()) / inertia.min())}


@dataclass(frozen=True)
class Summary:
    name: str
    true_k: int
    silhouette_k: int
    elbow_k: int
    ari_at_true_k: float
    silhouette_at_true_k: float
    silhouette_best: float


def summarise(name: str, X: NDArray[np.float64], y: NDArray[np.int_], true_k: int, scale: bool = False) -> Summary:
    Z = StandardScaler().fit_transform(X) if scale else X
    t = sweep(Z, y)
    return Summary(name, true_k, best_k_by_silhouette(t), elbow_k(t), float(t.loc[true_k, "ari"]), float(t.loc[true_k, "silhouette"]), float(t["silhouette"].max()))


def main() -> None:
    pd.set_option("display.width", 150)
    rows = []
    for name, (X, y, k) in datasets().items():
        rows.append(summarise(name, X, y, k, scale=name.startswith(("iris", "wine"))).__dict__)
    print("\nChoosing k (iris and wine standardised first)")
    print(pd.DataFrame(rows).set_index("name").round(3).to_string())
    X, y, k = datasets()["wine (3 cultivars)"]
    print("\nScaling on wine at the true k:", {a: round(b, 3) for a, b in scaling_effect(X, y, k).items()})
    print("\nStability over 10 single-start runs (k = true k)")
    for name, (X, _, k) in datasets().items():
        s = stability(StandardScaler().fit_transform(X) if name.startswith(("iris", "wine")) else X, k)
        print(f"  {name:22s} min pairwise ARI {s['min_pairwise_ari']:.3f}  mean {s['mean_pairwise_ari']:.3f}  inertia spread {s['inertia_spread']:.3f}")


if __name__ == "__main__":
    main()
