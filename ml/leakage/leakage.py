"""Train/test leakage: four ways a model's score gets inflated, each measured, each with its fix.

Leakage means information from the data used to *judge* a model reaches the process that *builds* it.
The score then looks better than anything the model will do on new data. Every function here returns
the leaky score next to the honest one, so the size of the lie is visible.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def selection_leakage(n: int = 100, p: int = 5000, k: int = 20, repeats: int = 10, seed: int = 0) -> dict[str, float]:
    """Pick the best-looking features on ALL rows, then cross-validate: on pure noise this "finds" signal.

    The labels are coin flips independent of every feature, so the true accuracy is 50%. Choosing the
    k features that correlate best with the labels *before* splitting lets the test rows vote on which
    features to keep. Fix: put the selection inside a pipeline, so each fold chooses features from its
    own training rows only.
    """
    rng = np.random.default_rng(seed)
    leaky, honest = [], []
    for _ in range(repeats):
        X = rng.normal(size=(n, p))
        y = rng.integers(0, 2, n)
        folds = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1 << 30)))
        chosen = SelectKBest(f_classif, k=k).fit_transform(X, y)  # <- uses every row, test folds included
        leaky.append(cross_val_score(LogisticRegression(max_iter=1000), chosen, y, cv=folds).mean())
        pipe = make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression(max_iter=1000))
        honest.append(cross_val_score(pipe, X, y, cv=folds).mean())
    return {"leaky": float(np.mean(leaky)), "honest": float(np.mean(honest)), "chance": 0.5}


def manual_honest_selection(n: int = 100, p: int = 5000, k: int = 20, seed: int = 0) -> float:
    """The same fix written out by hand: select on the training fold, apply to the test fold.

    Exists to check that the pipeline is doing exactly this, not something subtly different.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, n)
    scores = []
    for fit_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        selector = SelectKBest(f_classif, k=k).fit(X[fit_idx], y[fit_idx])
        model = LogisticRegression(max_iter=1000).fit(selector.transform(X[fit_idx]), y[fit_idx])
        scores.append(model.score(selector.transform(X[test_idx]), y[test_idx]))
    return float(np.mean(scores))


def target_leakage(n: int = 4000, seed: int = 0) -> dict[str, float]:
    """A feature that is only known *because* the outcome happened.

    ``cancellation_ticket`` is filed when a customer cancels, so it is almost perfectly tied to churn,
    but at the moment a retention model would run, it does not exist yet. A held-out split cannot catch
    this: the leaked column is in the test rows too. Only asking "would I know this at prediction
    time?" catches it. Fix: drop columns that are consequences of the target.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.exponential(24, n)
    monthly = rng.normal(65, 20, n)
    contract_m2m = rng.random(n) < 0.5
    logit = -1.2 + 1.0 * contract_m2m - 0.03 * tenure + 0.01 * (monthly - 65)
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    ticket = np.where(churn == 1, rng.random(n) < 0.95, rng.random(n) < 0.02).astype(int)
    honest_X = np.column_stack([tenure, monthly, contract_m2m])
    leaky_X = np.column_stack([honest_X, ticket])
    out = {}
    for name, X in (("leaky", leaky_X), ("honest", honest_X)):
        X_tr, X_te, y_tr, y_te = train_test_split(X, churn, test_size=0.25, random_state=seed, stratify=churn)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
        out[name] = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    return out


def group_data(n_groups: int = 150, rows: int = 8, dims: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Customers with several rows each. Each customer has a fixed fingerprint plus per-row noise.

    The label belongs to the *customer* and is random, so nothing about a fingerprint predicts it.
    """
    rng = np.random.default_rng(seed)
    fingerprints = rng.normal(0, 3, (n_groups, dims))
    labels = rng.integers(0, 2, n_groups)
    groups = np.repeat(np.arange(n_groups), rows)
    X = fingerprints[groups] + rng.normal(0, 0.5, (n_groups * rows, dims))
    return X, labels[groups], groups


def group_leakage(seed: int = 0) -> dict[str, float]:
    """The same customer in both train and test lets the model memorise people instead of learning a pattern.

    With random folds, a customer's other rows sit in the training set, so a flexible model recognises
    the fingerprint and recalls the label. With GroupKFold, whole customers are held out together, and
    the score drops to the truth: 50%, because the labels carry no signal. Fix: split by customer.
    """
    X, y, groups = group_data(seed=seed)
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    leaky = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=seed)).mean()
    honest = cross_val_score(model, X, y, cv=GroupKFold(5), groups=groups).mean()
    return {"leaky": float(leaky), "honest": float(honest), "chance": 0.5}


def scaler_leakage(seed: int = 0) -> dict[str, float]:
    """Scaling on all rows before splitting: textbook leakage, but here the effect is tiny.

    Included so the lesson is not "every leak is dramatic". The scaler only passes along means and
    variances, a weak channel compared with choosing features by their link to the labels. It is still
    wrong, still free to fix, and on other data or with smaller samples it can matter.
    """
    X, y = load_breast_cancer(return_X_y=True)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    leaky = cross_val_score(SVC(), StandardScaler().fit_transform(X), y, cv=folds).mean()
    honest = cross_val_score(make_pipeline(StandardScaler(), SVC()), X, y, cv=folds).mean()
    return {"leaky": float(leaky), "honest": float(honest)}


def summary() -> str:
    s, t, g, c = selection_leakage(), target_leakage(), group_leakage(), scaler_leakage()
    return "\n".join(
        [
            f"feature selection before CV (noise data, truth 50%): leaky {s['leaky']:.1%}  honest {s['honest']:.1%}",
            f"column that follows the outcome (AUC):               leaky {t['leaky']:.3f}  honest {t['honest']:.3f}",
            f"same customer in train and test (truth 50%):         leaky {g['leaky']:.1%}  honest {g['honest']:.1%}",
            f"scaler fitted on all rows (breast cancer, accuracy): leaky {c['leaky']:.4f} honest {c['honest']:.4f}",
        ]
    )


if __name__ == "__main__":
    print(summary())
