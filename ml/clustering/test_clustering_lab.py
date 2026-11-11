import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, silhouette_score

import clustering_lab as lab


@pytest.fixture(scope="module")
def data():
    return lab.datasets()


def test_inertia_never_rises_as_k_grows(data):
    X, y, _ = data["4 touching blobs"]
    t = lab.sweep(X, y)
    assert (np.diff(t["inertia"].to_numpy()) <= 1e-9).all()


def test_silhouette_and_elbow_find_four_well_separated_blobs(data):
    X, y, k = data["4 separated blobs"]
    t = lab.sweep(X, y)
    assert lab.best_k_by_silhouette(t) == k and lab.elbow_k(t) == k
    assert t.loc[k, "silhouette"] > 0.7 and t.loc[k, "ari"] == pytest.approx(1.0)


def test_silhouette_is_in_range_and_low_for_a_random_labelling(data):
    X, _, _ = data["4 separated blobs"]
    labels = np.random.default_rng(0).integers(0, 4, len(X))
    s = silhouette_score(X, labels)
    assert -1 <= s <= 1 and s < 0.05


def test_ari_ignores_label_names_and_is_near_zero_for_random_labels():
    y = np.repeat([0, 1, 2], 100)
    assert adjusted_rand_score(y, (y + 1) % 3) == pytest.approx(1.0)
    assert abs(adjusted_rand_score(y, np.random.default_rng(1).integers(0, 3, 300))) < 0.05


def test_elbow_on_a_synthetic_curve_finds_the_bend():
    table = pd.DataFrame({"inertia": [1000.0, 400.0, 100.0, 90.0, 82.0, 76.0, 71.0]}, index=pd.Index(range(2, 9), name="k"))
    assert lab.elbow_k(table) == 4


def test_kmeans_cannot_recover_two_moons_even_at_the_true_k(data):
    X, y, k = data["two moons"]
    assert lab.sweep(X, y, ks=(k,)).loc[k, "ari"] < 0.5


def test_scaling_changes_the_clustering_when_one_feature_has_a_huge_range():
    rng = np.random.default_rng(2)
    informative = np.vstack([rng.normal(0, 1, (150, 2)), rng.normal(6, 1, (150, 2))])
    y = np.repeat([0, 1], 150)
    noise = rng.normal(0, 1000, (300, 1))  # a feature measured in huge units that says nothing about the groups
    X = np.hstack([informative, noise])
    r = lab.scaling_effect(X, y, 2)
    assert r["ari_raw"] < 0.2 < 0.6 < r["ari_scaled"]


def test_single_start_runs_can_disagree_where_ten_starts_do_not(data):
    X, y, k = data["4 separated blobs"]
    assert lab.stability(X, k)["min_pairwise_ari"] < 0.9  # some single random starts land in a poor local minimum
    assert adjusted_rand_score(y, lab.fit(X, k, n_init=10).labels_) == pytest.approx(1.0)


def test_summary_reports_the_true_k_row(data):
    X, y, k = data["wine (3 cultivars)"]
    s = lab.summarise("wine", X, y, k, scale=True)
    assert s.true_k == 3 and s.ari_at_true_k > 0.8 and s.silhouette_k == 3


def test_dbscan_with_a_label_free_radius_recovers_two_moons_where_kmeans_cannot(data):
    X, y, k = data["two moons"]
    r = lab.compare_methods(X, y, k)
    assert r["dbscan_chosen"] > 0.95 and r["kmeans"] < 0.6


def test_gaussian_mixture_beats_kmeans_on_iris_where_clusters_are_elongated(data):
    X, y, k = data["iris (3 species)"]
    r = lab.compare_methods(X, y, k)
    assert r["gmm"] > r["kmeans"] + 0.15


def test_dbscan_depends_heavily_on_its_radius(data):
    X, y, _ = data["two moons"]
    t = lab.dbscan_eps_sweep(X, y)
    assert t["ari"].max() > 0.95 and t["ari"].min() < 0.1


def test_dbscan_choice_respects_the_noise_and_cluster_constraints():
    table = pd.DataFrame(
        {"clusters": [1, 3, 3, 3], "noise": [0.0, 0.5, 0.1, 0.05], "silhouette": [float("nan"), 0.9, 0.5, 0.4]}, index=pd.Index([0.1, 0.2, 0.3, 0.4], name="eps")
    )
    assert lab.dbscan_choice(table) == 0.3  # 0.2 has the best silhouette but labels half the points noise
    assert np.isnan(lab.dbscan_choice(table.assign(noise=0.9)))
