# Class imbalance: weights vs SMOTE vs doing nothing

When one class is rare, a default classifier flags few cases. The usual fixes are class weights (make rare-class
mistakes cost more) and SMOTE (invent synthetic rare-class rows). `imbalance_lab.py` compares both with doing
nothing on synthetic data with a known rare class (3.6% positives after label noise, 6,000 rows), out-of-fold, with
SMOTE applied inside each training fold only so synthetic rows never reach validation.

## What I measured

| method | PR-AUC | recall at cutoff 0.5 | rows flagged at 0.5 | mean score | best F1 over cutoffs |
|---|---|---|---|---|---|
| plain | 0.474 | 0.228 | 53 | 0.036 | 0.496 |
| class weights | 0.458 | 0.667 | 1,300 | 0.363 | 0.477 |
| SMOTE | 0.460 | 0.676 | 1,306 | 0.354 | 0.490 |

- **Recall at 0.5 triples**, but the model flags 1,300 rows to find about 145 of the roughly 216 positives, so precision falls sharply.
- **Ranking does not improve.** PR-AUC and the best reachable F1 are the same or slightly worse. Both fixes move the
  scores, they do not teach the model anything new; moving the cutoff on the plain model reaches the same F1.
- **Scores stop being probabilities.** The plain model's mean score (0.036) matches the 3.6% base rate; after
  reweighting or SMOTE it is about 0.35, so "0.7" no longer means a 70% chance. If you need probabilities, recalibrate.
- At a 15.6% positive rate the pattern is the same (PR-AUC 0.731, 0.727, 0.727), so it is not specific to very rare classes.

## Limits

- One synthetic dataset, a logistic regression and default SMOTE settings; a different model (for example boosted
  trees) or other resamplers could behave differently.
- No error bars from repeated splits; differences of 0.01 to 0.02 PR-AUC are within noise.
- Synthetic features are well behaved. SMOTE interpolates between neighbours and can do worse on categorical or
  noisy features.

## Run

```bash
pip install -r ../../requirements-dev.txt
python imbalance_lab.py
pytest
```
