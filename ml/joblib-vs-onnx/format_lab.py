"""Saving a fitted model as joblib (a pickle) or as ONNX: size, speed, agreement, and what loading can do.

The same scikit-learn pipeline (one-hot encoding and scaling, then a model) is saved both ways and compared on:

* file size
* prediction agreement: ONNX runs in float32, so probabilities differ slightly; do the decisions?
* load time and prediction latency for one row and for a batch (machine dependent; use the ratios, not the numbers)
* safety: loading a pickle can run arbitrary code, loading an ONNX file cannot (demonstrated with a harmless payload)
"""

from __future__ import annotations

import io
import pickle
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from numpy.typing import NDArray
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 11
NUMERIC = ["age", "income", "visits"]
CATEGORICAL = ["plan", "region"]
FloatArray = NDArray[np.float64]


def make_data(n: int = 3000, seed: int = SEED) -> tuple[pd.DataFrame, NDArray[np.int_]]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "income": rng.lognormal(10, 0.5, n),
            "visits": rng.poisson(4, n).astype(float),
            "plan": rng.choice(["basic", "plus", "pro"], n),
            "region": rng.choice(["north", "south", "east", "west"], n),
        }
    )
    z = 0.03 * (df["age"] - 45) - 0.5 * np.log(df["income"] / 22000) + 0.2 * df["visits"] + (df["plan"] == "pro") * 0.8 - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df, y


def build(model: str) -> Pipeline:
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    est = LogisticRegression(max_iter=500) if model == "logistic" else RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=SEED)
    return Pipeline([("prep", prep), ("model", est)])


def to_onnx(pipe: Pipeline) -> bytes:
    """Convert a fitted pipeline. Numbers go in as one float column each and strings as one string column each."""
    types = [(c, FloatTensorType([None, 1])) for c in NUMERIC] + [(c, StringTensorType([None, 1])) for c in CATEGORICAL]
    model = convert_sklearn(pipe, initial_types=types, options={id(pipe.steps[-1][1]): {"zipmap": False}}, target_opset=17)
    return bytes(model.SerializeToString())


def onnx_inputs(df: pd.DataFrame) -> dict[str, NDArray[np.generic]]:
    out: dict[str, NDArray[np.generic]] = {}
    for c in NUMERIC:
        out[c] = df[c].to_numpy(dtype=np.float32).reshape(-1, 1)
    for c in CATEGORICAL:
        out[c] = df[c].to_numpy(dtype=object).reshape(-1, 1).astype(str)
    return out


def onnx_proba(session: ort.InferenceSession, df: pd.DataFrame) -> FloatArray:
    _, proba = session.run(None, onnx_inputs(df))
    return np.asarray(proba, dtype=float)[:, 1]


def median_ms(fn: object, reps: int) -> float:
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()  # type: ignore[operator]
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


@dataclass
class Row:
    model: str
    joblib_kb: float
    onnx_kb: float
    max_proba_diff: float
    share_rows_diff_over_1e3: float
    label_agreement: float
    load_ms_joblib: float
    load_ms_onnx: float
    one_row_ms_joblib: float
    one_row_ms_onnx: float
    batch_ms_joblib: float
    batch_ms_onnx: float


def compare(model: str, workdir: Path, batch: int = 1000, reps: int = 15) -> Row:
    df, y = make_data()
    pipe = build(model).fit(df.iloc[:2000], y[:2000])
    test = df.iloc[2000:]
    jpath, opath = workdir / f"{model}.joblib", workdir / f"{model}.onnx"
    joblib.dump(pipe, jpath)
    opath.write_bytes(to_onnx(pipe))
    sess = ort.InferenceSession(str(opath), providers=["CPUExecutionProvider"])
    loaded = joblib.load(jpath)
    p_j = loaded.predict_proba(test)[:, 1]
    p_o = onnx_proba(sess, test)
    one, many = test.iloc[:1], test.iloc[:batch]
    return Row(
        model=model,
        joblib_kb=jpath.stat().st_size / 1024,
        onnx_kb=opath.stat().st_size / 1024,
        max_proba_diff=float(np.abs(p_j - p_o).max()),
        share_rows_diff_over_1e3=float((np.abs(p_j - p_o) > 1e-3).mean()),
        label_agreement=float(((p_j >= 0.5) == (p_o >= 0.5)).mean()),
        load_ms_joblib=median_ms(lambda: joblib.load(jpath), reps),
        load_ms_onnx=median_ms(lambda: ort.InferenceSession(str(opath), providers=["CPUExecutionProvider"]), reps),
        one_row_ms_joblib=median_ms(lambda: loaded.predict_proba(one), reps * 4),
        one_row_ms_onnx=median_ms(lambda: sess.run(None, onnx_inputs(one)), reps * 4),
        batch_ms_joblib=median_ms(lambda: loaded.predict_proba(many), reps),
        batch_ms_onnx=median_ms(lambda: sess.run(None, onnx_inputs(many)), reps),
    )


# ---- safety demonstration (harmless payload) ----------------------------------------------------------------------
EXECUTED: list[str] = []


def _payload() -> str:
    EXECUTED.append("code from the file ran during load")
    return "model-looking object"


class Malicious:
    """A pickle can name any callable to run on load. This one only appends to a list; a real attack would not."""

    def __reduce__(self) -> tuple[object, tuple[()]]:
        return (_payload, ())


def pickle_load_runs_code() -> bool:
    EXECUTED.clear()
    pickle.loads(pickle.dumps(Malicious()))
    return bool(EXECUTED)


def onnx_load_rejects_non_models() -> bool:
    """Feeding the same malicious pickle bytes to the ONNX loader fails; it parses a graph, it does not execute."""
    EXECUTED.clear()
    blob = pickle.dumps(Malicious())
    try:
        onnx.load_model_from_string(blob)
        parsed = True
    except Exception:  # noqa: BLE001 - any parse failure counts as a rejection
        parsed = False
    return not parsed and not EXECUTED


def stringify(rows: list[Row]) -> str:
    buf = io.StringIO()
    pd.DataFrame([r.__dict__ for r in rows]).round(4).to_string(buf, index=False)
    return buf.getvalue()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        print(stringify([compare(m, Path(d)) for m in ("logistic", "forest")]))
    print("pickle load ran code:", pickle_load_runs_code())
    print("onnx loader rejected pickle bytes without running code:", onnx_load_rejects_non_models())
