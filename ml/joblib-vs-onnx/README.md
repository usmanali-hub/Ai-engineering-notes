# Saving models: joblib vs ONNX

The same scikit-learn pipeline (scaling and one-hot encoding, then a model) saved as a joblib file (a pickle) and as ONNX,
for a logistic regression and a 100-tree random forest on a synthetic mixed-type dataset (3 numeric columns, 2 text
columns; 2,000 training rows, 1,000 scored).

```
python format_lab.py
pytest
```

Needs `skl2onnx` and `onnxruntime` (in `requirements-dev.txt`).

## What was measured

| model | joblib size | ONNX size | max probability difference | rows differing by more than 0.001 | decision agreement |
|---|---|---|---|---|---|
| logistic | 3.4 KB | 1.2 KB | 0.0000 | 0% | 100% |
| forest | 2.9 MB | 1.4 MB | 0.026 | 22% | 99.6% |

* **ONNX files are smaller** here (about a third of the size for the logistic model, half for the forest).
* **The logistic model converts exactly** to float32 precision.
* **The forest does not.** The probabilities differ by up to 0.026 and about a fifth of the rows differ by more than 0.001,
  while 99.6% of the 0.5-threshold decisions agree. ONNX compares in float32, so a feature value very close to a split
  point can fall on the other side and land in a different leaf. That is small in aggregate and real per row: if a
  downstream system uses the probability as a score, or a cutoff close to where many rows sit, check agreement on your
  own data before switching format.

## Speed (machine dependent: read the ratios, not the milliseconds)

Medians over repeated calls on a 2-core cloud container:

| | scikit-learn via joblib | ONNX Runtime |
|---|---|---|
| predict one row (logistic) | about 3 ms | about 0.1 ms |
| predict one row (forest) | about 8 ms | about 0.1 ms |
| predict 1,000 rows (logistic) | about 4 ms | about 0.5 ms |
| predict 1,000 rows (forest) | about 16 ms | about 4-16 ms (varied run to run) |
| load the file (forest) | about 16-19 ms | about 21-50 ms |

* **Single-row latency is where ONNX wins clearly** (a factor of 30 or more here), because scikit-learn's per-call overhead
  dominates for one row.
* **Large batches narrow the gap** for the forest, and loading was slower for ONNX in these runs. These timings varied by
  a factor of two between runs, so treat them as indicative only.

## Safety: loading a pickle runs code

A pickle can name any function to call when it is loaded. The lab builds one whose payload only appends to a list, loads it,
and shows the payload ran. Feeding the same bytes to the ONNX loader fails to parse and runs nothing, because an ONNX file is
a data description (a graph of operators and weights), not a program. This is why `ml-end-to-end` documents that its joblib
models must only be loaded from sources you trust. ONNX removes that specific risk; it does not remove the need to know
where a model came from, since a malicious graph can still be wrong on purpose.

## When to pick which

* **joblib**: your own pipeline, same Python environment, anything scikit-learn can fit. Zero conversion work. Pickles are
  not portable across library versions, and they run code on load.
* **ONNX**: serving from another language or runtime, low single-row latency, or loading models from a less trusted place.
  Costs a conversion step that supports only some estimators and transformers, input types that must be declared
  (here one column per feature), and the agreement check above.

## Limits

* Two models, one synthetic dataset, one machine. The agreement gap in particular depends on the model and the data.
* Only models the converter supports can be exported; custom transformers need extra work.
* The conversion fixed the ONNX opset at 17 and ZipMap off (plain probability array); other settings behave differently.
* The safety demonstration is a toy payload; it shows the mechanism, not how much damage is likely.
