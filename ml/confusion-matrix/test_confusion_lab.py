import confusion_lab as cl
import numpy as np
import pytest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


@pytest.fixture(scope="module")
def cancer():
    return cl.cancer_scores()


def test_hand_counts_match_sklearn(cancer):
    y, s = cancer
    for cutoff in (0.05, 0.3, 0.5, 0.9):
        k = cl.confusion(y, s, cutoff)
        tn, fp, fn, tp = confusion_matrix(y, (s >= cutoff).astype(int)).ravel()
        assert (k.tp, k.fp, k.fn, k.tn) == (tp, fp, fn, tn)


def test_metrics_match_sklearn(cancer):
    y, s = cancer
    k, pred = cl.confusion(y, s), (s >= 0.5).astype(int)
    assert k.precision == pytest.approx(precision_score(y, pred))
    assert k.recall == pytest.approx(recall_score(y, pred))
    assert k.f1 == pytest.approx(f1_score(y, pred))
    assert k.mcc == pytest.approx(matthews_corrcoef(y, pred))
    assert k.specificity == pytest.approx(recall_score(y, pred, pos_label=0))


def test_cutoff_trades_false_alarms_for_misses(cancer):
    rows = cl.sweep(*cancer, (0.05, 0.3, 0.5, 0.7, 0.9))
    fp = [m.fp for _, m in rows]
    fn = [m.fn for _, m in rows]
    assert fp == sorted(fp, reverse=True) and fn == sorted(fn)
    assert all(m.n == rows[0][1].n for _, m in rows)


def test_zero_division_is_zero_not_an_error():
    k = cl.confusion(np.array([0, 0, 1]), np.array([0.0, 0.0, 0.0]))
    assert (k.precision, k.f1) == (0.0, 0.0) and k.recall == 0.0
    assert cl.Confusion(0, 0, 0, 0).mcc == 0.0


def test_cost_optimal_cutoff_beats_every_other_cutoff(cancer):
    y, s = cancer
    for ratio in (1, 5, 20):
        cutoff, cost = cl.cost_optimal_cutoff(y, s, ratio)
        for c in np.linspace(0, 1, 101):
            k = cl.confusion(y, s, float(c))
            assert cost <= k.fp + ratio * k.fn + 1e-9
        k = cl.confusion(y, s, cutoff)
        assert cost == k.fp + ratio * k.fn


def test_pricier_misses_lower_the_cutoff(cancer):
    cheap, _ = cl.cost_optimal_cutoff(*cancer, 1)
    dear, _ = cl.cost_optimal_cutoff(*cancer, 20)
    assert dear < cheap


def test_accuracy_paradox_on_rare_events():
    y, s = cl.rare_event_scores(0.01)
    never = cl.confusion(y, np.zeros(len(y)))
    model = cl.confusion(y, s)
    assert never.accuracy > 0.98 and never.recall == 0 and never.mcc == 0
    assert model.mcc > never.mcc and model.recall > never.recall
