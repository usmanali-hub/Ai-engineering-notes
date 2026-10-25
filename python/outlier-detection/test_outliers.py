import numpy as np
from outliers import iqr_outliers, modified_zscore_outliers, zscore_outliers


def data_with_clustered_outliers():
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(50, 5, 200), np.linspace(150, 180, 30)])


def test_zscore_is_masked_by_clustered_outliers():
    # 30 huge values (13% of the data) inflate the standard deviation so much that they hide themselves.
    flagged = zscore_outliers(data_with_clustered_outliers(), threshold=3.0)
    assert flagged[-30:].sum() == 0


def test_robust_methods_catch_all_clustered_outliers():
    data = data_with_clustered_outliers()
    assert iqr_outliers(data)[-30:].all()
    assert modified_zscore_outliers(data)[-30:].all()


def test_single_extreme_value_is_caught_by_all_methods():
    data = np.append(np.random.default_rng(1).normal(0, 1, 100), 25.0)
    for fn in (zscore_outliers, iqr_outliers, modified_zscore_outliers):
        assert fn(data)[-1]


def test_hand_computed_iqr_fences():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 100.0])
    # Q1=3, Q3=7, IQR=4 -> fences are -3 and 13
    assert iqr_outliers(x).tolist() == [False] * 8 + [True]


def test_constant_data_has_no_outliers():
    x = np.full(10, 4.2)
    for fn in (zscore_outliers, iqr_outliers, modified_zscore_outliers):
        assert not fn(x).any()
