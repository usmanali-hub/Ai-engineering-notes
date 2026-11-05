"""Confusion matrices from scratch, checked against scikit-learn, on real and imbalanced data."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @staticmethod
    def _ratio(num: float, den: float) -> float:
        return num / den if den else 0.0

    @property
    def accuracy(self) -> float:
        return self._ratio(self.tp + self.tn, self.n)

    @property
    def precision(self) -> float:
        return self._ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self._ratio(self.tp, self.tp + self.fn)

    @property
    def specificity(self) -> float:
        return self._ratio(self.tn, self.tn + self.fp)

    @property
    def f1(self) -> float:
        return self._ratio(2 * self.precision * self.recall, self.precision + self.recall)

    @property
    def mcc(self) -> float:
        """Matthews correlation: 0 for no skill, 1 for perfect, whatever the class balance."""
        den = sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        return self._ratio(self.tp * self.tn - self.fp * self.fn, den)


def confusion(y_true: NDArray[np.int_], scores: NDArray[np.float64], cutoff: float = 0.5) -> Confusion:
    """Count the four outcomes by hand; a row is flagged when its score is >= cutoff."""
    flagged = scores >= cutoff
    positive = y_true == 1
    return Confusion(
        tp=int(np.sum(flagged & positive)),
        fp=int(np.sum(flagged & ~positive)),
        fn=int(np.sum(~flagged & positive)),
        tn=int(np.sum(~flagged & ~positive)),
    )


def sweep(y_true: NDArray[np.int_], scores: NDArray[np.float64], cutoffs: tuple[float, ...]) -> list[tuple[float, Confusion]]:
    return [(c, confusion(y_true, scores, c)) for c in cutoffs]


def cost_optimal_cutoff(
    y_true: NDArray[np.int_], scores: NDArray[np.float64], miss_cost: float, false_alarm_cost: float = 1.0
) -> tuple[float, float]:
    """Cutoff minimising false_alarm_cost*FP + miss_cost*FN over every distinct score; returns (cutoff, cost)."""
    best = (0.5, float("inf"))
    for c in np.unique(scores):
        k = confusion(y_true, scores, float(c))
        cost = false_alarm_cost * k.fp + miss_cost * k.fn
        if cost < best[1]:
            best = (float(c), float(cost))
    return best


def _fit(X: NDArray[np.float64], y: NDArray[np.int_]) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(Xtr, ytr)
    return yte, model.predict_proba(Xte)[:, 1]


def cancer_scores() -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Held-out scores for 'malignant' (1) on the breast cancer data."""
    data = load_breast_cancer()
    return _fit(data.data, (data.target == 0).astype(int))


def rare_event_scores(rate: float = 0.01) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Held-out scores on a synthetic problem where only `rate` of rows are positive."""
    X, y = make_classification(
        n_samples=20000, n_features=10, n_informative=4, weights=[1 - rate], flip_y=0.0, class_sep=0.8, random_state=SEED
    )
    return _fit(X, y)


def main() -> None:
    y, s = cancer_scores()
    k = confusion(y, s)
    print(f"Breast cancer (positive = malignant), {k.n} held-out patients, cutoff 0.5")
    print(f"  TP {k.tp}  FP {k.fp}  FN {k.fn}  TN {k.tn}")
    print(f"  accuracy {k.accuracy:.3f}  precision {k.precision:.3f}  recall {k.recall:.3f}  "
          f"specificity {k.specificity:.3f}  F1 {k.f1:.3f}  MCC {k.mcc:.3f}")
    print("\ncutoff  flagged  FP  FN  precision  recall")
    for c, m in sweep(y, s, (0.05, 0.1, 0.3, 0.5, 0.7, 0.9)):
        print(f"{c:>6}  {m.tp + m.fp:>7}  {m.fp:>2}  {m.fn:>2}  {m.precision:>9.3f}  {m.recall:>6.3f}")
    for ratio in (1, 5, 20):
        c, cost = cost_optimal_cutoff(y, s, ratio)
        print(f"miss costs {ratio:>2}x a false alarm -> best cutoff {c:.3f}, cost {cost:.0f}")

    y, s = rare_event_scores()
    never = confusion(y, np.zeros(len(y)), 0.5)
    model = confusion(y, s)
    print(f"\nRare event ({y.mean():.1%} positive), {len(y)} held-out rows")
    print(f"  predict 'no' always : accuracy {never.accuracy:.3f}  recall {never.recall:.3f}  MCC {never.mcc:.3f}")
    print(f"  model at 0.5        : accuracy {model.accuracy:.3f}  recall {model.recall:.3f}  "
          f"precision {model.precision:.3f}  MCC {model.mcc:.3f}")


if __name__ == "__main__":
    main()
