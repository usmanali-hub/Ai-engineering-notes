"""Bias-variance trade-off measured, not just drawn.

Many independent training sets are drawn from the same noisy curve (same x values, fresh noise each time), a polynomial of each degree is
fitted to every one, and the predictions are compared with the true curve. That gives the three terms
of expected squared error directly:

    expected test error = bias^2 + variance + noise^2

* bias^2   how far the *average* fitted curve is from the truth (a model too simple to bend enough)
* variance how much the fitted curve changes from one training set to the next (a model too flexible)
* noise^2  the part no model can remove
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NOISE = 0.3


def truth(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


@dataclass
class Result:
    degrees: list[int]
    bias2: np.ndarray
    variance: np.ndarray
    noise2: float
    train_mse: np.ndarray
    test_mse: np.ndarray  # measured against fresh noisy targets, independent of the decomposition
    grid: np.ndarray
    predictions: dict[int, np.ndarray]  # degree -> (n_datasets, n_grid) fitted curves

    @property
    def expected_error(self) -> np.ndarray:
        return self.bias2 + self.variance + self.noise2


def simulate(
    degrees: list[int] | None = None,
    n_train: int = 20,
    n_datasets: int = 300,
    noise: float = NOISE,
    seed: int = 0,
) -> Result:
    degrees = degrees or list(range(1, 16))
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 200)
    x = np.linspace(0, 1, n_train)  # fixed design: only the noise changes between datasets, and no fit has to extrapolate
    preds = {d: np.empty((n_datasets, grid.size)) for d in degrees}
    train_err = {d: np.empty(n_datasets) for d in degrees}
    test_err = {d: np.empty(n_datasets) for d in degrees}
    for i in range(n_datasets):
        y = truth(x) + rng.normal(0, noise, n_train)
        fresh_y = truth(grid) + rng.normal(0, noise, grid.size)  # a new noisy test sample for this dataset
        for d in degrees:
            fit = np.polynomial.Polynomial.fit(x, y, d)  # fits on a rescaled domain, so degree 12 stays stable
            preds[d][i] = fit(grid)
            train_err[d][i] = np.mean((fit(x) - y) ** 2)
            test_err[d][i] = np.mean((preds[d][i] - fresh_y) ** 2)
    mean_pred = {d: preds[d].mean(axis=0) for d in degrees}
    return Result(
        degrees=degrees,
        bias2=np.array([np.mean((mean_pred[d] - truth(grid)) ** 2) for d in degrees]),
        variance=np.array([np.mean(preds[d].var(axis=0)) for d in degrees]),
        noise2=noise**2,
        train_mse=np.array([train_err[d].mean() for d in degrees]),
        test_mse=np.array([test_err[d].mean() for d in degrees]),
        grid=grid,
        predictions=preds,
    )


def plot(result: Result, path: str, show_degrees: tuple[int, ...] = (1, 3, 15)) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    for ax, d in zip(axes[:3], show_degrees, strict=True):
        curves = result.predictions[d]
        for curve in curves[:25]:
            ax.plot(result.grid, curve, color="#4c72b0", alpha=0.25, linewidth=0.8)
        ax.plot(result.grid, curves.mean(axis=0), color="#c44e52", linewidth=2, label="average fit")
        ax.plot(result.grid, truth(result.grid), color="black", linewidth=2, linestyle="--", label="truth")
        ax.set_ylim(-2, 2)
        ax.set_title(f"degree {d}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="lower left", fontsize=8)
    ax = axes[3]
    ax.plot(result.degrees, result.bias2, marker="o", label="bias²", color="#dd8452")
    ax.plot(result.degrees, result.variance, marker="o", label="variance", color="#55a868")
    ax.plot(result.degrees, result.expected_error, marker="o", label="bias² + variance + noise²", color="#4c72b0")
    ax.plot(result.degrees, result.test_mse, marker="x", linestyle=":", label="measured test error", color="black")
    ax.plot(result.degrees, result.train_mse, marker="s", linestyle="--", label="training error", color="#937860")
    best = result.degrees[int(np.argmin(result.expected_error))]
    ax.axvline(best, color="grey", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree (model flexibility)")
    ax.set_ylabel("squared error (log scale)")
    ax.set_title(f"lowest expected error at degree {best}")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("Each blue curve is a fit to a different training sample of the same noisy sine wave", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary(result: Result) -> str:
    lines = ["degree   bias²   variance   noise²   expected   measured   train"]
    for i, d in enumerate(result.degrees):
        lines.append(
            f"{d:>6} {result.bias2[i]:7.3f} {result.variance[i]:10.3f} {result.noise2:8.3f} "
            f"{result.expected_error[i]:10.3f} {result.test_mse[i]:10.3f} {result.train_mse[i]:7.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    res = simulate()
    print(summary(res))
    plot(res, "bias_variance.png")
    print("wrote bias_variance.png")
