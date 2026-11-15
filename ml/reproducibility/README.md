# Reproducibility: seeds and data versioning

What a random seed controls, what it does not, how much a single reported number depends on it, and how to notice that the
data underneath changed. Two datasets: breast cancer (569 rows) and a noisy synthetic problem (500 rows, 10% label noise),
a 60-tree random forest, 70/30 stratified splits.

```
python repro_lab.py
pytest
```

## How much one number depends on the seed (test accuracy over 30 seeds)

| dataset | what changes with the seed | mean | std | min | max |
|---|---|---|---|---|---|
| breast cancer | split only | 0.954 | 0.017 | 0.930 | 0.983 |
| breast cancer | model only | 0.941 | 0.007 | 0.924 | 0.953 |
| breast cancer | both | 0.953 | 0.016 | 0.924 | 0.983 |
| noisy synthetic | split only | 0.784 | 0.039 | 0.707 | 0.853 |
| noisy synthetic | model only | 0.782 | 0.012 | 0.760 | 0.807 |
| noisy synthetic | both | 0.776 | 0.034 | 0.707 | 0.847 |

* **Which rows land in the test set matters more than the model's own randomness**, by a factor of 2 to 3 here. A single
  score from a single split carries a spread of several accuracy points; on the noisy data the same code gives anywhere
  from 0.71 to 0.85. Report a mean and spread over splits, or cross-validate.

## "Try ten seeds and report the best"

Each trial draws ten seeds. *Claimed* is the best test accuracy among them minus the average of the ten. *Real* is how much
better the chosen seed's model is on a second test set that took no part in the choice.

| dataset | claimed gain | real gain on untouched data (± s.e.) |
|---|---|---|
| breast cancer | +0.029 | +0.001 ± 0.001 |
| noisy synthetic | +0.050 | +0.002 ± 0.003 |

The best-seed number is a gain of 3 to 5 points that is not there: on fresh data the chosen seed is no better than a typical
one. This is the same selection effect as tuning on the test set. A seed is a nuisance parameter, not something to optimise.

## What fixing a seed gives you

* Re-running with the same seed, data, code and libraries reproduces the predicted probabilities **bit for bit**.
* Changing the **thread count** (`n_jobs` 1 vs 2) with the same seed does **not** reproduce them bit for bit, because summing
  trees in a different order rounds differently, but the difference was about 1e-16, which is nothing. Compare with a
  tolerance (`np.allclose`), not `==`.
* A different seed moved the probabilities by up to 0.16 (cancer) and 0.22 (synthetic): the seed matters, so record it.
* A seed pins one library version on one platform. Different versions of scikit-learn or NumPy can change what a seed
  produces, so the run manifest records them too.

## Data versioning: a content hash in the run manifest

`fingerprint(df)` hashes the column names, dtypes and every cell (SHA-256, first 16 hex characters). It is row-order
sensitive by default, with an option to ignore order. What it notices, with breast cancer:

| change | default hash changes | order-insensitive hash changes | accuracy (same seed and code) |
|---|---|---|---|
| unchanged | no | no | 0.971 |
| rows reordered | **yes** | no | 0.959 |
| 2% of one column edited by 1% | yes | yes | 0.971 |
| one column renamed | yes | yes | 0.971 |
| one row dropped | yes | yes | 0.965 |

* **Reordering rows changes the result with no data change at all**, because the same seed then splits different rows into
  the test set (0.971 to 0.959). That is why the default hash counts row order: it is part of what "same data" means
  for a reproducible run. Use the order-insensitive hash to ask "is this the same set of records?".
* **A change can be invisible in the score and still matter.** Editing 2% of a column left the accuracy unchanged to three
  decimals, so a metric-only check would not have noticed. The hash did.
* A `Manifest` bundles the data hash, seed, row and column counts and the Python, scikit-learn, NumPy and pandas versions;
  its `run_id` changes when any of them does.

## Limits

* One model family and two small datasets. The size of the seed spread depends on the data and the model.
* A hash says *that* the data changed, not *what* changed. For that you need the data itself (DVC, lakeFS, or dated copies).
* The hash covers values, not their source: it cannot tell you the file was downloaded from the right place.
* Bit-for-bit reproducibility across machines, GPUs or library versions is not promised by a seed; tolerance-based checks
  are the realistic standard.
