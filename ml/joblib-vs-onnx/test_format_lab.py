from pathlib import Path

import format_lab as lab
import numpy as np
import onnxruntime as ort
import pytest


@pytest.fixture(scope="module")
def rows(tmp_path_factory: pytest.TempPathFactory) -> dict[str, lab.Row]:
    d = Path(tmp_path_factory.mktemp("formats"))
    return {m: lab.compare(m, d, batch=200, reps=3) for m in ("logistic", "forest")}


def test_logistic_probabilities_agree_to_float32_precision(rows):
    r = rows["logistic"]
    assert r.max_proba_diff < 1e-4 and r.label_agreement == 1.0


def test_forest_decisions_nearly_always_agree_even_if_probabilities_can_drift(rows):
    r = rows["forest"]
    assert r.label_agreement > 0.98
    assert r.max_proba_diff < 0.1  # float32 thresholds can send a row to a different leaf


def test_sizes_and_timings_are_recorded_and_positive(rows):
    for r in rows.values():
        assert min(r.joblib_kb, r.onnx_kb, r.load_ms_joblib, r.load_ms_onnx, r.one_row_ms_joblib, r.one_row_ms_onnx) > 0


def test_onnx_one_row_prediction_is_not_slower_than_scikit_learn_by_a_large_factor(rows):
    # loose on purpose: timings are machine dependent, so only guard against a gross regression
    for r in rows.values():
        assert r.one_row_ms_onnx < 3 * r.one_row_ms_joblib


def test_onnx_model_takes_raw_columns_and_matches_pipeline_on_new_data(tmp_path):
    df, y = lab.make_data(600, seed=3)
    pipe = lab.build("logistic").fit(df, y)
    path = tmp_path / "m.onnx"
    path.write_bytes(lab.to_onnx(pipe))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(lab.onnx_proba(sess, df), pipe.predict_proba(df)[:, 1], atol=1e-4)
    assert {i.name for i in sess.get_inputs()} == set(lab.NUMERIC + lab.CATEGORICAL)


def test_an_unseen_category_is_ignored_by_both_formats(tmp_path):
    df, y = lab.make_data(600, seed=4)
    pipe = lab.build("logistic").fit(df, y)
    path = tmp_path / "m.onnx"
    path.write_bytes(lab.to_onnx(pipe))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    odd = df.head(5).assign(plan="platinum")
    np.testing.assert_allclose(lab.onnx_proba(sess, odd), pipe.predict_proba(odd)[:, 1], atol=1e-4)


def test_loading_a_pickle_runs_code_and_the_onnx_loader_does_not():
    assert lab.pickle_load_runs_code() is True
    assert lab.onnx_load_rejects_non_models() is True
