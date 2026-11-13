import numpy as np
import pca_lab as lab
import pytest


def test_scree_sums_to_one_and_is_sorted():
    for X, _, _ in lab.datasets().values():
        r = lab.scree(X)
        assert r.sum() == pytest.approx(1.0)
        assert np.all(np.diff(r) <= 1e-12)


def test_components_for_counts_the_cumulative_threshold():
    r = np.array([0.5, 0.3, 0.15, 0.05])
    assert lab.components_for(r, 0.5) == 1
    assert lab.components_for(r, 0.8) == 2
    assert lab.components_for(r, 0.95) == 3
    assert lab.components_for(r, 1.0) == 4
    with pytest.raises(ValueError):
        lab.components_for(r, 0)


def test_unscaled_wine_is_dominated_by_the_column_with_big_units():
    X, _, names = lab.datasets()["wine"]
    col, share = lab.top_feature_share(X, scale=False)
    assert names[col] == "proline" and share > 0.99
    assert lab.scree(X, scale=False)[0] > 0.99
    assert lab.top_feature_share(X, scale=True)[1] < 0.3
    assert lab.scree(X, scale=True)[0] < 0.5


def test_keeping_every_component_loses_nothing_for_knn():
    X, y, _ = lab.datasets()["iris"]
    assert lab.knn_accuracy(X, y, X.shape[1]) == pytest.approx(lab.knn_accuracy(X, y, None))


def test_projection_quality_is_in_range_and_two_pca_axes_beat_two_random_columns_on_wine():
    X, y, _ = lab.datasets()["wine"]
    q = lab.projection_quality(X, y)
    assert all(0 <= v <= 1 for v in q.values())
    assert q["acc_pca_2d"] > q["acc_two_random_columns"] + 0.1


def test_pca_is_poor_when_class_information_is_in_a_low_variance_direction():
    t = lab.variance_trap()
    assert t["variance_share_pc1"] > 0.9
    assert abs(t["pca_1d"] - 0.5) < 0.1  # chance
    assert t["lda_1d"] > 0.95
    assert t["pca_2d"] > 0.95  # nothing is lost until a component is dropped


def test_reconstruction_error_falls_with_k_and_equals_the_variance_lost():
    X, _, _ = lab.datasets()["digits"]
    r = lab.reconstruction_error(X, (2, 10, 40))
    assert r["mse"].is_monotonic_decreasing
    np.testing.assert_allclose(r["relative_mse"], r["variance_lost"], atol=1e-9)


def test_run_has_a_row_per_dataset_and_plot_writes_a_file(tmp_path):
    r = lab.run()
    assert set(r.scree["dataset"]) == {"iris", "wine", "digits"} and len(r.scaling) == 6
    out = tmp_path / "pca.png"
    lab.plot(out)
    assert out.stat().st_size > 1000
