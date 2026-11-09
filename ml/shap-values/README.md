# SHAP values from first principles

A Shapley value splits one prediction among the features: for a row, it is the average (over every order in which
features could be revealed) of how much the model's output moves when that feature is revealed. Unrevealed features
take values from a background sample. `shap_lab.py` computes it exactly by enumerating all 2^d subsets (fine for six
features), then checks it against LightGBM's built-in TreeSHAP (`pred_contrib=True`).

```
python shap_lab.py
pytest
```

No `shap` package is needed. The point is to see what the number is, so the real library's output is not a black box.

## What the tests pin down

* **Efficiency**: base value plus the contributions equals the model output, to machine precision (1e-15 here).
* **Linear model**: the value of feature *i* is exactly weight × (x − background mean), so the idea reduces to the
  familiar one when the model is simple.
* **Dummy and symmetry**: a feature the model ignores gets exactly zero; two interchangeable features get equal credit.
* **Interactions**: the credit for a product of two features is shared equally between them.
* **TreeSHAP**: its contributions sum to the model's raw score.

## Results (LightGBM, 80 trees, 60 explained rows, 80 background rows)

Six features; the outcome depends on x0, x1 (with an interaction) and a little on x2; x3 to x5 are noise.

| feature | mean abs SHAP (exact) | mean abs SHAP (TreeSHAP) | permutation (accuracy drop) |
|---|---|---|---|
| x0 | 1.356 | 1.330 | 0.198 |
| x1 | 0.650 | 0.636 | 0.109 |
| x2 | 0.194 | 0.205 | 0.006 |
| x3 | 0.127 | 0.128 | 0.014 |
| x4 | 0.103 | 0.105 | -0.006 |
| x5 | 0.134 | 0.133 | 0.008 |

Exact and TreeSHAP values correlate 0.996 (mean absolute difference 0.046). With x5 replaced by a near-copy of x0 the
credit splits (x0 falls from 1.356 to 0.884; the copy gets 0.528), and the gap to TreeSHAP rises a little to 0.063
(correlation 0.993).

## Things to take from it

* SHAP and permutation importance tell the same broad story here (x0 and x1 matter, the rest barely) but are not the
  same quantity: SHAP is in log-odds units and per row; permutation is an accuracy drop on one dataset.
* **Noise columns are not zero.** x3 to x5 get 0.10 to 0.13 because the fitted model uses them to chase noise. SHAP
  explains the model, not the world.
* **Correlated copies split the credit.** Neither column's value alone says how much the underlying signal matters;
  permutation importance splits it too.

## Limits

* One dataset, one seed, 60 explained rows; the 0.77 and 0.94 rank agreements with permutation in the script output
  are over six features and move with the seed, so they are not evidence of anything general.
* Exact enumeration costs 2^d model calls per row; it cannot scale past roughly ten features, which is why TreeSHAP and
  sampling approximations exist. Only the interventional definition is implemented.
