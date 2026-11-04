"""Ridge versus Lasso, measured.

Both add a penalty on coefficient size to least squares. Ridge (L2) shrinks every coefficient toward zero but keeps
them all. Lasso (L1) shrinks some to exactly zero, which selects features. Which one predicts better depends on the
truth: Lasso should win when only a few features matter, Ridge when many matter a little. This lab builds both worlds
with a known answer, then tries a real dataset.

The penalty strength (``alpha``) is always chosen by cross-validation inside the training data, and features are
standardised inside the pipeline, because the penalty is not scale-free.
"""

import warnings
from dataclasses import dataclass

import matplotlib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    RidgeCV,
    lasso_path,
)
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHAS = np.logspace(-3, 3, 60)
GROUP = 5  # columns per correlated group


@dataclass
class World:
    X: np.ndarray
    y: np.ndarray
    coef: np.ndarray  # the true coefficients


def make_world(kind: str, n: int = 100, p: int = 50, noise: float = 3.0, seed: int = 0) -> World:
    """``sparse``: 5 of p coefficients are large and the rest exactly zero. ``dense``: all p are small and non-zero.

    ``correlated``: like sparse, but the columns come in groups of five that move together (correlation about 0.9),
    and each real feature has four near-copies that are not real. Lasso cannot tell a feature from its copies.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    if kind == "correlated":
        groups = p // GROUP
        factor = rng.normal(size=(n, groups))
        X = np.sqrt(0.9) * np.repeat(factor, GROUP, axis=1) + np.sqrt(0.1) * X
    coef = np.zeros(p)
    if kind == "correlated":
        coef[::GROUP][:5] = [4.0, -3.0, 2.5, -2.0, 1.5]  # the first column of each of the first five groups
    elif kind == "sparse":
        coef[:5] = [4.0, -3.0, 2.5, -2.0, 1.5]
    elif kind == "dense":
        coef[:] = rng.normal(0, 0.7, p)
    else:
        raise ValueError(f"unknown world: {kind}")
    return World(X, X @ coef + rng.normal(0, noise, n), coef)


def models(elastic_net: bool = False) -> dict[str, Pipeline]:
    out = {
        "ols": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
        "lasso": make_pipeline(StandardScaler(), LassoCV(alphas=ALPHAS, cv=5, max_iter=20000, random_state=0)),
    }
    if elastic_net:  # a mix of the two penalties, designed for groups of correlated features
        out["elastic_net"] = make_pipeline(
            StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=ALPHAS, cv=5, max_iter=20000, random_state=0)
        )
    return out


def original_scale_coef(pipe: Pipeline) -> np.ndarray:
    """Coefficients mapped back to the units of the raw inputs, so they can be compared with the truth."""
    scaler = pipe.named_steps["standardscaler"]
    return np.asarray(pipe[-1].coef_ / scaler.scale_, dtype=float)


def experiment(kind: str, repeats: int = 30, n: int = 100, p: int = 50, noise: float = 3.0, seed: int = 0) -> dict[str, dict[str, float]]:
    """Average over many fresh datasets: out-of-sample error, coefficient error, and how well Lasso finds the right features.

    Test error is measured on 5,000 fresh rows from the same world, as a multiple of the irreducible noise variance
    (1.0 would be a perfect model).
    """
    use_enet = kind == "correlated"
    sums: dict[str, dict[str, list[float]]] = {m: {"test_error": [], "coef_error": []} for m in models(use_enet)}
    found, wrong, zeros_kept, group_found, clean, decoys_kept = [], [], [], [], [], []
    for r in range(repeats):
        w = make_world(kind, n, p, noise, seed + r)
        fresh = make_world(kind, 5000, p, noise, seed + 10_000 + r)
        fresh_y = fresh.X @ w.coef + np.random.default_rng(seed + 20_000 + r).normal(0, noise, 5000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for name, pipe in models(use_enet).items():
                pipe.fit(w.X, w.y)
                mse = float(np.mean((pipe.predict(fresh.X) - fresh_y) ** 2))
                sums[name]["test_error"].append(mse / noise**2)
                sums[name]["coef_error"].append(float(np.linalg.norm(original_scale_coef(pipe) - w.coef)))
                if name == "lasso":
                    chosen = np.abs(original_scale_coef(pipe)) > 0
                    truth = w.coef != 0
                    found.append(float((chosen & truth).sum() / max(truth.sum(), 1)))
                    if kind == "correlated":
                        grid = chosen.reshape(-1, GROUP)[:5]  # the five groups that hold a real feature
                        real, copies = grid[:, 0], grid[:, 1:]
                        group_found.append(float(grid.any(axis=1).mean()))
                        clean.append(float((real & ~copies.any(axis=1)).mean()))
                        decoys_kept.append(float(copies.sum()))
                    else:
                        clean.append(float(chosen[:5].mean()))
                        decoys_kept.append(float(chosen[5 : 5 + 4 * 5].sum()))  # 20 noise columns, the same number as the copies above
                    wrong.append(float((chosen & ~truth).sum()))
                    zeros_kept.append(float((~chosen).sum()))
    out = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in sums.items()}
    out["lasso"].update(true_features_found=float(np.mean(found)), false_features_kept=float(np.mean(wrong)), coefficients_zeroed=float(np.mean(zeros_kept)))
    out["lasso"]["clean_selection"] = float(np.mean(clean))  # the real column chosen with none of its copies chosen alongside
    out["lasso"]["copies_kept_of_20"] = float(np.mean(decoys_kept))
    if group_found:
        out["lasso"]["right_group_found"] = float(np.mean(group_found))
    return out


def scale_dependence(seed: int = 0) -> dict[str, float]:
    """The penalty looks at coefficient size, and coefficient size depends on the units.

    Two equally useful features, one measured in thousands and one in fractions. Unscaled, Lasso charges the
    large-unit feature almost nothing (its coefficient is tiny) and the small-unit feature a lot, so it drops
    the wrong one. Standardised, it treats them alike.
    """
    rng = np.random.default_rng(seed)
    n = 300
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = 2 * a + 2 * b + rng.normal(0, 1, n)
    X = np.column_stack([a * 1000, b * 0.001])  # same information, wildly different units
    raw = Lasso(alpha=0.5, max_iter=100000).fit(X, y)
    scaled = make_pipeline(StandardScaler(), Lasso(alpha=0.5, max_iter=100000)).fit(X, y)
    return {
        "unscaled_kept": float(np.count_nonzero(raw.coef_)),
        "unscaled_small_unit_feature_kept": float(raw.coef_[1] != 0),
        "scaled_kept": float(np.count_nonzero(scaled[-1].coef_)),
    }


def real_data(repeats: int = 5, seed: int = 0) -> dict[str, float]:
    """Diabetes progression (442 patients, 10 measurements): mean squared error over repeated 5-fold CV.

    ``alpha`` is chosen by cross-validation inside each training fold, so the outer score is honest.
    """
    X, y = load_diabetes(return_X_y=True)
    folds = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return {name: float(-cross_val_score(pipe, X, y, cv=folds, scoring="neg_mean_squared_error").mean()) for name, pipe in models().items()}


def path_figure(path: str, seed: int = 0) -> None:
    """Coefficients as the penalty grows: Ridge fades smoothly, Lasso switches features off one by one."""
    from sklearn.linear_model import ridge_regression

    w = make_world("sparse", seed=seed)
    Z = StandardScaler().fit_transform(w.X)
    yc = w.y - w.y.mean()
    alphas, coefs, _ = lasso_path(Z, yc, alphas=ALPHAS)
    ridge = np.array([ridge_regression(Z, yc, alpha=a * len(yc)) for a in alphas])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    true = w.coef != 0
    for ax, c, title in ((axes[0], ridge.T, "Ridge"), (axes[1], coefs, "Lasso")):
        for j in range(c.shape[0]):
            ax.plot(alphas, c[j], color="#c44e52" if true[j] else "#999999", linewidth=1.6 if true[j] else 0.6, alpha=1 if true[j] else 0.7)
        ax.set_xscale("log")
        ax.set_xlabel("penalty strength (alpha)")
        ax.set_title(f"{title}: the 5 real features (red) and 45 noise features (grey)")
        ax.title.set_fontsize(9)
    axes[0].set_ylabel("coefficient (standardised inputs)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary() -> str:
    lines = []
    for kind in ("sparse", "dense", "correlated"):
        r = experiment(kind)
        lines.append(f"\n{kind} truth (100 rows, 50 features, 30 fresh datasets): test error as a multiple of noise variance, coefficient error")
        for name in [m for m in ("ols", "ridge", "lasso", "elastic_net") if m in r]:
            lines.append(f"  {name:<6}{r[name]['test_error']:>8.3f}{r[name]['coef_error']:>10.2f}")
        la = r["lasso"]
        if kind == "sparse":
            lines.append(f"  lasso found {la['true_features_found']:.0%} of the 5 real features, kept {la['false_features_kept']:.1f} of 45 noise features ({la['copies_kept_of_20']:.1f} of the first 20), zeroed {la['coefficients_zeroed']:.1f} of 50")
        elif kind == "correlated":
            lines.append(f"  lasso: right group found {la['right_group_found']:.0%}, real column picked {la['true_features_found']:.0%}, but picked cleanly (no copy alongside) only {la['clean_selection']:.0%}; kept {la['copies_kept_of_20']:.1f} of the 20 copies (independent noise columns: see sparse)")
        else:
            lines.append(f"  lasso zeroed {la['coefficients_zeroed']:.1f} of 50 coefficients although every one matters")
    lines.append("\ndiabetes (real, 442 rows): mean squared error " + ", ".join(f"{k} {v:.0f}" for k, v in real_data().items()))
    s = scale_dependence()
    lines.append(f"\nscale trap: unscaled Lasso kept {s['unscaled_kept']:.0f} feature(s) and {'kept' if s['unscaled_small_unit_feature_kept'] else 'DROPPED'} the small-unit one; standardised kept {s['scaled_kept']:.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    path_figure("regularization_path.png")
    print("wrote regularization_path.png")
