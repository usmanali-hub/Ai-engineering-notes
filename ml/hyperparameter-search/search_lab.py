"""Grid search, random search and Optuna's TPE sampler at the same trial budget.

Question: for a fixed number of model fits, which way of choosing hyperparameters finds a better configuration?
Three methods tune the same gradient boosting model (four hyperparameters) on three small datasets:

* grid: the cross product of a few evenly spaced values per hyperparameter, sized to the budget;
* random: independent draws from the same ranges (log-uniform where the range spans orders of magnitude);
* tpe: Optuna's Tree-structured Parzen Estimator, which spends later trials near the best earlier ones.

Each method gets a budget of 8, 16 or 32 trials. A trial is a 3-fold cross-validated accuracy on the training part of
a split. The configuration with the best cross-validated score is then refitted on the whole training part and scored
on a held-out test part that no search saw. The gap between the two scores is the optimism of the search.
Repeated over several seeds (each changes the split and the random draws) so that one lucky run does not decide.
"""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")  # tiny datasets: OpenMP thread start-up costs more than it saves

import numpy as np
import optuna
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

BUDGETS = (8, 16, 32)
SEEDS = tuple(range(12))
SPACE: dict[str, tuple[float, float, str]] = {  # name: (low, high, scale)
    "learning_rate": (0.01, 0.3, "log"),
    "max_leaf_nodes": (4, 32, "logint"),
    "l2_regularization": (1e-4, 10.0, "log"),
    "min_samples_leaf": (5, 40, "int"),
}
DEFAULTS = {"learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 0.0, "min_samples_leaf": 20}
MAX_ITER = 60


def datasets() -> dict[str, tuple[NDArray[np.float64], NDArray[np.int_]]]:
    noisy = make_classification(n_samples=800, n_features=20, n_informative=5, n_redundant=3, flip_y=0.1, class_sep=0.8, random_state=7)
    return {"breast cancer": load_breast_cancer(return_X_y=True), "wine": load_wine(return_X_y=True), "noisy synthetic": noisy}  # type: ignore[dict-item]


def _clip(name: str, value: float) -> float | int:
    low, high, scale = SPACE[name]
    v = min(max(value, low), high)
    return round(v) if scale in ("int", "logint") else float(v)


def levels(name: str, k: int) -> list[float | int]:
    """k evenly spaced values over the hyperparameter's range (in log space for log scales), away from the edges."""
    low, high, scale = SPACE[name]
    qs = [(i + 0.5) / k for i in range(k)]
    if scale in ("log", "logint"):
        return [_clip(name, math.exp(math.log(low) + q * (math.log(high) - math.log(low)))) for q in qs]
    return [_clip(name, low + q * (high - low)) for q in qs]


def grid_configs(budget: int) -> list[dict[str, float | int]]:
    """A lattice with exactly ``budget`` points: 8 = 2x2x2 (min_samples_leaf fixed), 16 = 2x2x2x2, 32 = 4x2x2x2."""
    counts = {8: (2, 2, 2, 1), 16: (2, 2, 2, 2), 32: (4, 2, 2, 2)}[budget]
    axes = [levels(n, c) if c > 1 else [DEFAULTS[n]] for n, c in zip(SPACE, counts, strict=True)]
    return [dict(zip(SPACE, combo, strict=True)) for combo in itertools.product(*axes)]


def random_configs(budget: int, seed: int) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(budget):
        cfg: dict[str, float | int] = {}
        for name, (low, high, scale) in SPACE.items():
            u = rng.random()
            v = math.exp(math.log(low) + u * (math.log(high) - math.log(low))) if scale in ("log", "logint") else low + u * (high - low)
            cfg[name] = _clip(name, v)
        out.append(cfg)
    return out


def model(cfg: dict[str, float | int], seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=MAX_ITER, early_stopping=False, random_state=seed, **cfg)  # type: ignore[arg-type]


def cv_score(cfg: dict[str, float | int], X: NDArray[np.float64], y: NDArray[np.int_], seed: int) -> float:
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)
    return float(cross_val_score(model(cfg, seed), X, y, cv=folds, scoring="accuracy").mean())


@dataclass(frozen=True)
class Result:
    best_cfg: dict[str, float | int]
    trial_scores: list[float]

    @property
    def best_score(self) -> float:
        return max(self.trial_scores)


def run_list(configs: list[dict[str, float | int]], X: NDArray[np.float64], y: NDArray[np.int_], seed: int) -> Result:
    scores = [cv_score(c, X, y, seed) for c in configs]
    return Result(configs[int(np.argmax(scores))], scores)


def run_tpe(budget: int, X: NDArray[np.float64], y: NDArray[np.int_], seed: int) -> Result:
    def objective(trial: optuna.Trial) -> float:
        cfg: dict[str, float | int] = {}
        for name, (low, high, scale) in SPACE.items():
            if scale == "log":
                cfg[name] = trial.suggest_float(name, low, high, log=True)
            elif scale == "logint":
                cfg[name] = trial.suggest_int(name, int(low), int(high), log=True)
            else:
                cfg[name] = trial.suggest_int(name, int(low), int(high))
        return cv_score(cfg, X, y, seed)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget)
    scores = [float(t.value) for t in study.trials if t.value is not None]
    return Result(dict(study.best_params), scores)


METHODS: dict[str, Callable[[int, NDArray[np.float64], NDArray[np.int_], int], Result]] = {
    "grid": lambda b, X, y, s: run_list(grid_configs(b), X, y, s),
    "random": lambda b, X, y, s: run_list(random_configs(b, s), X, y, s),
    "tpe": run_tpe,
}


def experiment(budgets: tuple[int, ...] = BUDGETS, seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    rows = []
    for ds, (X, y) in datasets().items():
        for seed in seeds:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            for budget in budgets:
                for method, run in METHODS.items():
                    r = run(budget, Xtr, ytr, seed)
                    test = float(model(r.best_cfg, seed).fit(Xtr, ytr).score(Xte, yte))
                    rows.append({"dataset": ds, "seed": seed, "budget": budget, "method": method, "cv_best": r.best_score, "test": test, "trials": len(r.trial_scores)})
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Per budget and method: mean CV best, mean test score, optimism (CV best minus test), and the paired
    difference in test score against random search over the same dataset and seed."""
    t = table.copy()
    t["optimism"] = t["cv_best"] - t["test"]
    ref = t[t["method"] == "random"].set_index(["dataset", "seed", "budget"])["test"]
    t["vs_random"] = t["test"].to_numpy() - ref.reindex(pd.MultiIndex.from_frame(t[["dataset", "seed", "budget"]])).to_numpy()
    g = t.groupby(["budget", "method"])
    return pd.DataFrame(
        {
            "cv_best": g["cv_best"].mean(),
            "test": g["test"].mean(),
            "optimism": g["optimism"].mean(),
            "vs_random": g["vs_random"].mean(),
            "vs_random_se": g["vs_random"].std(ddof=1) / np.sqrt(g["vs_random"].count()),
        }
    )


def main() -> None:
    pd.set_option("display.width", 140)
    table = experiment()
    print(summarise(table).round(4).to_string())
    print("\nMean best CV score by dataset and method at budget 32")
    print(table[table["budget"] == 32].groupby(["dataset", "method"])[["cv_best", "test"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
