# Class imbalance: weights vs SMOTE vs doing nothing

When one class is rare, a default classifier flags few cases. The usual fixes are class weights (make rare-class
mistakes cost more) and SMOTE (invent synthetic rare-class rows). `imbalance_lab.py` compares both with doing
nothing on synthetic data with a known rare class (6,000 rows per dataset), out-of-fold, with SMOTE applied inside
each training fold only so synthetic rows never reach validation. Two models (logistic regression and histogram
gradient boosting) are run on ten independently drawn datasets at two positive rates (about 3% and 15%).

## What I measured (mean over ten datasets)

| model | positives | method | PR-AUC | best F1 over cutoffs | recall at 0.5 | rows flagged at 0.5 | mean score | beat plain on PR-AUC |
|---|---|---|---|---|---|---|---|---|
| logistic | ~3% | plain | 0.325 | 0.371 | 0.124 | 30 | 0.035 | - |
| logistic | ~3% | class weights | 0.275 | 0.321 | 0.703 | 1,659 | 0.366 | 0 of 10 |
| logistic | ~3% | SMOTE | 0.282 | 0.328 | 0.697 | 1,604 | 0.349 | 0 of 10 |
| logistic | ~15% | plain | 0.607 | 0.580 | 0.362 | 426 | 0.154 | - |
| logistic | ~15% | class weights | 0.575 | 0.563 | 0.760 | 1,877 | 0.373 | 0 of 10 |
| logistic | ~15% | SMOTE | 0.577 | 0.563 | 0.755 | 1,838 | 0.363 | 0 of 10 |
| boosting | ~3% | plain | 0.554 | 0.565 | 0.335 | 82 | 0.030 | - |
| boosting | ~3% | class weights | 0.503 | 0.521 | 0.622 | 435 | 0.159 | 0 of 10 |
| boosting | ~3% | SMOTE | 0.481 | 0.504 | 0.668 | 630 | 0.192 | 0 of 10 |
| boosting | ~15% | plain | 0.871 | 0.804 | 0.686 | 703 | 0.150 | - |
| boosting | ~15% | class weights | 0.860 | 0.792 | 0.861 | 1,172 | 0.254 | 0 of 10 |
| boosting | ~15% | SMOTE | 0.856 | 0.788 | 0.863 | 1,197 | 0.252 | 1 of 10 |

- **Recall at the default cutoff rises a lot** (for example 0.124 to about 0.70 for logistic at 3%), but the model flags
  more than 1,600 rows to do it, so precision falls sharply.
- **Ranking does not improve, and usually gets slightly worse.** Neither fix beat the plain model on PR-AUC in 79 of
  the 80 dataset comparisons it was tried on (one SMOTE win, boosting at 15%). The best F1 reachable by moving the cutoff is also lower, so reweighting
  or resampling moved the scores around without teaching the model anything; moving the cutoff on the plain model is at least as good.
- **Scores stop being probabilities.** The plain models' mean score matches the base rate (0.035 at about 3.6%
  positives); after reweighting or SMOTE it is 0.16 to 0.37, so "0.7" no longer means a 70% chance. If you need
  probabilities, recalibrate.
- Spread: PR-AUC varies by 0.06 to 0.17 between datasets (the synthetic datasets differ in how separable they are), so only the paired comparison
  within a dataset, shown in the last column, is meaningful.

## Limits

- Synthetic data with well-behaved features; real data with categorical or noisy features can behave differently, and
  SMOTE in particular can do worse there.
- Two models and default settings. Tuning the class weight, or using SMOTE variants, might do better.
- Ten datasets per setting is enough to see a consistent direction, not to put a precise number on a small effect.

## Run

```bash
pip install -r ../../requirements-dev.txt
python imbalance_lab.py   # about 3 minutes
pytest
```
