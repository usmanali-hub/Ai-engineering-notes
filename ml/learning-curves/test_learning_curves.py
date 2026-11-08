import learning_curves as lc
import numpy as np
import pytest


def fake(train: list[float], val: list[float]) -> lc.Curve:
    return lc.Curve("d", "m", np.array([100, 250, 500, 750, 1000]), np.array(train), np.array(val), np.zeros(5))


def test_gap_means_overfitting():
    c = fake([1.0] * 5, [0.6, 0.65, 0.7, 0.72, 0.73])
    assert lc.diagnose(c, best_validation=0.73) == "overfitting (variance)"


def test_low_score_small_gap_means_underfitting():
    c = fake([0.7] * 5, [0.69] * 5)
    assert lc.diagnose(c, best_validation=0.85) == "underfitting (bias)"


def test_close_to_best_with_small_gap_is_about_right():
    c = fake([0.9] * 5, [0.88] * 5)
    assert lc.diagnose(c, best_validation=0.9) == "about right"


def test_last_doubling_gain():
    c = fake([1] * 5, [0.5, 0.6, 0.7, 0.75, 0.8])
    assert lc.would_more_data_help(c) == pytest.approx(0.1)


def test_curve_shapes_and_training_score_not_below_validation():
    c = lc.curve("noisy synthetic", "tree, depth 3", sizes=(0.25, 0.5, 1.0))
    assert len(c.sizes) == len(c.train) == len(c.validation) == 3
    assert c.sizes[0] < c.sizes[-1]
    assert (c.train >= c.validation - 0.02).all()


def test_unlimited_tree_memorises_and_is_flagged():
    s = lc.summary("noisy synthetic", {m: lc.curve("noisy synthetic", m, sizes=(0.5, 1.0)) for m in ("tree, depth 3", "tree, unlimited")})
    assert s.loc["tree, unlimited", "train_score"] == pytest.approx(1.0)
    assert s.loc["tree, unlimited", "diagnosis"] == "overfitting (variance)"
