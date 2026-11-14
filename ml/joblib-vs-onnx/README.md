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
* **The forest does not, and the cause is measured, not guessed.** The probabilities differ by up to 0.026, and about a fifth
  of the rows differ by more than 0.001, while 99.6% of the 0.5-threshold decisions agree. Three checks locate it:
  * the converted forest alone, given exactly the same float32 inputs as scikit-learn, agrees to 3e-7, so the ONNX trees
    are right;
  * the ONNX preprocessing differs from scikit-learn's by about 3e-7 per feature (float32 arithmetic);
  * scikit-learn's *own* forest, given float32-rounding-sized noise (relative 2e-7) on its inputs, moves by up to 0.024 and
    changes 17% of rows by more than 0.001, which is the same size as the ONNX gap. A logistic regression moves by 1e-7.
  A deep forest has many thresholds, a continuous feature lands near one of them in some tree about one row in five,
  and each such flip moves the average a little. So the gap is the forest's sensitivity to last-digit differences, not a
  conversion defect, and no converter setting removes it: double-precision inputs and the newer ML opset were tried and left
  it at 17% of rows over 0.001. Check agreement on your own data before switching format, and compare decisions, not
  probabilities, if exactness matters.

## Speed (min to max over 5 independent measurements, 2-core cloud container)

| | scikit-learn via joblib | ONNX Runtime |
|---|---|---|
| predict one row (logistic) | 2.8-3.9 ms | 0.07-0.09 ms |
| predict one row (forest) | 7.6-10.4 ms | 0.08-0.14 ms |
| predict 1,000 rows (logistic) | 3.0-3.3 ms | 0.37-0.42 ms |
| predict 1,000 rows (forest) | 16.4-20.7 ms | 4.5-5.0 ms |
| load the file (logistic) | 0.71-0.78 ms | 0.68-0.81 ms |
| load the file (forest) | 16.7-23.3 ms | 16.8-20.8 ms |

* **Single-row latency is where ONNX wins most** (30 to 100 times here), because scikit-learn's per-call overhead
  dominates for one row.
* **Batches of 1,000 are 8 times faster for the logistic model and 4 times for the forest.** Loading takes about the same.
* An earlier single run showed a wide spread because other jobs were using the same cores; with nothing else running the
  ranges above are narrow. They are still one machine: use the ratios, not the milliseconds.

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

* Two models, one synthetic dataset, one machine. The size of the forest agreement gap depends on the model and the data.
* Only models the converter supports can be exported; custom transformers need extra work.
* The conversion fixed the ONNX opset at 17 and ZipMap off (plain probability array); other settings behave differently.
* The safety demonstration is a toy payload; it shows the mechanism, not how much damage is likely.
