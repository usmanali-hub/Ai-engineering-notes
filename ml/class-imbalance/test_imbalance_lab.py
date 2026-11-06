import numpy as np
import pytest
from sklearn.metrics import f1_score

import imbalance_lab as lab


@pytest.fixture(scope="module")
def rare():
    return lab.compare(0.03)


def test_data_has_the_stated_rarity():
    _, y = lab.make_data(0.03)
    assert 0.025 < y.mean() < 0.045
    assert len(y) == 6000


def test_plain_model_flags_few_and_misses_most_positives(rare):
    assert rare.loc["plain", "recall_at_0.5"] < 0.35
    assert rare.loc["plain", "flagged_at_0.5"] < 0.03 * 6000


def test_weights_and_smote_trade_precision_for_recall_at_the_default_cutoff(rare):
    for m in ("class_weight", "smote"):
        assert rare.loc[m, "recall_at_0.5"] > rare.loc["plain", "recall_at_0.5"] + 0.3
        assert rare.loc[m, "flagged_at_0.5"] > 10 * rare.loc["plain", "flagged_at_0.5"]


def test_neither_fix_improves_ranking_quality(rare):
    # PR-AUC ignores the cutoff, so it shows whether the model itself got better
    plain = rare.loc["plain", "pr_auc"]
    assert rare["pr_auc"].max() - rare["pr_auc"].min() < 0.05
    assert rare.loc["class_weight", "pr_auc"] < plain + 0.02
    assert rare.loc["smote", "pr_auc"] < plain + 0.02


def test_scores_stop_being_probabilities_after_rebalancing(rare):
    prevalence = rare.attrs["prevalence"]
    assert abs(rare.loc["plain", "mean_score"] - prevalence) < 0.01  # calibrated on average
    for m in ("class_weight", "smote"):
        assert rare.loc[m, "mean_score"] > 5 * prevalence


def test_best_f1_matches_a_brute_force_search():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    s = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    brute = max(f1_score(y, (s >= c).astype(int)) for c in np.unique(s))
    assert lab.best_f1(y, s) == pytest.approx(brute)


def test_smote_runs_only_on_training_folds():
    X, y = lab.make_data()
    fit = np.arange(0, 4000)
    val = np.arange(4000, 6000)
    pipe = lab.build("smote").fit(X[fit], y[fit])
    assert pipe.predict_proba(X[val]).shape == (len(val), 2)  # validation rows are scored, never resampled
    assert pipe.named_steps["smote"].sampling_strategy == "auto"
    assert lab.out_of_fold("smote", X, y).shape == y.shape


def test_unknown_method_still_builds_a_plain_pipeline():
    assert "smote" not in lab.build("plain").named_steps


@pytest.fixture(scope="module")
def paired():
    return lab.repeated(0.03, "logistic", seeds=(7, 8, 9))


def test_repeated_reports_spread_and_paired_wins(paired):
    assert list(paired.index) == list(lab.METHODS)
    assert (paired["datasets"] == 3).all()
    assert (paired["pr_auc_std"] > 0).all()  # datasets differ, so the spread is real
    assert paired.loc["plain", "pr_auc_wins_vs_plain"] == 0


def test_fixes_do_not_win_on_ranking_across_datasets(paired):
    assert paired.loc["class_weight", "pr_auc_wins_vs_plain"] <= 1
    assert paired.loc["smote", "pr_auc_wins_vs_plain"] <= 1
    assert paired.loc["class_weight", "pr_auc_mean"] < paired.loc["plain", "pr_auc_mean"] + 0.01


def test_boosting_model_runs_and_shows_the_same_recall_shift():
    t = lab.compare(0.03, "boosting", seed=7)
    assert t.loc["class_weight", "recall_at_0.5"] > t.loc["plain", "recall_at_0.5"]
    assert t.loc["smote", "flagged_at_0.5"] > t.loc["plain", "flagged_at_0.5"]
    assert t.loc["class_weight", "pr_auc"] < t.loc["plain", "pr_auc"] + 0.03
