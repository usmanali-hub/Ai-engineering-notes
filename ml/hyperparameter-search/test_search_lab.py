import numpy as np
import pytest

import search_lab as lab


@pytest.fixture(scope="module")
def data():
    return lab.datasets()["wine"]


@pytest.mark.parametrize("budget", lab.BUDGETS)
def test_grid_has_exactly_the_budget_and_distinct_points(budget):
    g = lab.grid_configs(budget)
    assert len(g) == budget and len({tuple(sorted(c.items())) for c in g}) == budget


@pytest.mark.parametrize("budget", lab.BUDGETS)
def test_every_method_respects_the_ranges(budget):
    for cfg in lab.grid_configs(budget) + lab.random_configs(budget, 0):
        for name, (low, high, _) in lab.SPACE.items():
            assert low <= cfg[name] <= high


def test_levels_are_evenly_spaced_in_log_space_for_log_scales():
    v = lab.levels("learning_rate", 4)
    ratios = np.diff(np.log(v))
    assert ratios == pytest.approx(ratios[0] * np.ones(3))
    assert all(isinstance(x, int) for x in lab.levels("min_samples_leaf", 3))


def test_random_configs_are_reproducible_and_depend_on_the_seed():
    assert lab.random_configs(10, 1) == lab.random_configs(10, 1)
    assert lab.random_configs(10, 1) != lab.random_configs(10, 2)


def test_each_method_uses_exactly_its_budget_and_reports_the_best_trial(data):
    X, y = data
    for method, run in lab.METHODS.items():
        r = run(4 if method == "tpe" else 8, X, y, 0) if method != "grid" else run(8, X, y, 0)
        assert r.best_score == max(r.trial_scores)
        assert len(r.trial_scores) == (4 if method == "tpe" else 8)


def test_tpe_is_deterministic_for_a_seed(data):
    X, y = data
    a, b = lab.run_tpe(5, X, y, 3), lab.run_tpe(5, X, y, 3)
    assert a.trial_scores == b.trial_scores and a.best_cfg == b.best_cfg


def test_summary_compares_each_method_with_random_search_and_random_with_itself_is_zero():
    t = lab.experiment(budgets=(8,), seeds=(0,))
    s = lab.summarise(t)
    assert set(t["trials"]) == {8}
    assert s.loc[(8, "random"), "vs_random"] == pytest.approx(0.0)
    assert (s["optimism"] > -0.05).all()
