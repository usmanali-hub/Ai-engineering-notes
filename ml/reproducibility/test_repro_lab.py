import numpy as np
import pandas as pd
import pytest
import repro_lab as lab


@pytest.fixture(scope="module")
def cancer():
    return lab.datasets()["breast cancer"]


def test_rerunning_with_the_same_seed_is_bitwise_identical_and_thread_count_changes_at_most_rounding(cancer):
    r = lab.identical_runs(*cancer)
    assert r["same seed, rerun: bitwise equal"] is True
    assert r["same seed, 1 vs 2 threads: max difference"] < 1e-12  # may or may not be bitwise equal; never meaningfully different
    assert r["different seed: max difference"] > 0.05


def test_split_seed_moves_the_score_more_than_model_seed(cancer):
    s = lab.seed_spread(*cancer, seeds=range(12)).set_index("varied")
    assert s.loc["split seed only", "std"] > s.loc["model seed only", "std"]
    assert (s["max"] > s["min"]).all()


def test_best_of_n_seeds_looks_better_than_it_is(cancer):
    r = lab.best_of_n_inflation(*cancer, n=8, trials=15)
    assert r["claimed_gain"] > 0.01  # picking the max on the data used to pick always looks better than average
    assert r["claimed_gain"] > r["real_gain"] + 0.01  # and the untouched data does not agree


def test_fingerprint_is_stable_and_notices_edits_renames_and_dropped_rows(cancer):
    X, _ = cancer
    base = lab.fingerprint(X)
    assert lab.fingerprint(X.copy()) == base
    assert lab.fingerprint(lab.edit_cells(X, 0.02)) != base
    assert lab.fingerprint(X.rename(columns={X.columns[0]: "z"})) != base
    assert lab.fingerprint(X.iloc[1:]) != base


def test_row_order_changes_the_default_hash_but_not_the_order_insensitive_one(cancer):
    X, _ = cancer
    shuffled = X.sample(frac=1.0, random_state=3)
    assert lab.fingerprint(shuffled) != lab.fingerprint(X)
    assert lab.fingerprint(shuffled, True) == lab.fingerprint(X, True)


def test_a_single_changed_cell_is_enough_to_change_the_hash(cancer):
    X, _ = cancer
    edited = X.copy()
    edited.iloc[100, 3] += 1e-9
    assert lab.fingerprint(edited) != lab.fingerprint(X)


def test_manifest_run_id_depends_on_data_and_seed_only_through_their_values(cancer):
    X, _ = cancer
    a, b = lab.manifest(X, 1), lab.manifest(X.copy(), 1)
    assert a == b and a.run_id == b.run_id
    assert lab.manifest(X, 2).run_id != a.run_id
    assert lab.manifest(lab.edit_cells(X, 0.02), 1).run_id != a.run_id
    assert a.rows == len(X) and a.columns == X.shape[1]


def test_change_report_covers_each_kind_of_change(cancer):
    r = lab.change_report(*cancer).set_index("change")
    assert not r.loc["unchanged", "hash changes"]
    assert r.loc["rows reordered", "hash changes"] and not r.loc["rows reordered", "hash changes (ignoring row order)"]
    assert r.loc["2% of one column edited by 1%", "hash changes (ignoring row order)"]
    assert np.isfinite(r["accuracy"]).all() and isinstance(r, pd.DataFrame)
