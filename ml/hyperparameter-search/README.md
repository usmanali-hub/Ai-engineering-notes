# Hyperparameter search: grid, random and Optuna at the same budget

Tunes a gradient boosting model (four hyperparameters: learning rate, leaf count, L2 penalty, minimum leaf size) on
three small datasets (breast cancer, wine, a noisy synthetic problem) with three methods, each given 8, 16 or 32
trials. A trial is a 3-fold cross-validated accuracy. The best configuration is then refitted on the whole training
part and scored on a test part that no search saw. Four seeds (each changes the split and the random draws).

* **grid**: evenly spaced values per hyperparameter (log-spaced for the ranges spanning orders of magnitude), a lattice
  with exactly the budget's number of points (2x2x2, 2x2x2x2, 4x2x2x2);
* **random**: independent draws from the same ranges;
* **tpe**: Optuna's Tree-structured Parzen Estimator.

```
python search_lab.py      # about 5 minutes on 2 cores
pytest
```

## Results (mean over 3 datasets x 4 seeds = 12 runs per cell)

| budget | method | best CV accuracy | test accuracy | test minus random (paired) |
|---|---|---|---|---|
| 8 | grid | 0.9090 | 0.9226 | +0.007 ± 0.007 |
| 8 | random | 0.9100 | 0.9153 | 0 |
| 8 | tpe | 0.9096 | 0.9242 | +0.009 ± 0.009 |
| 16 | grid | 0.9104 | 0.9166 | +0.007 ± 0.008 |
| 16 | random | 0.9130 | 0.9099 | 0 |
| 16 | tpe | 0.9146 | 0.9230 | +0.013 ± 0.009 |
| 32 | grid | 0.9123 | 0.9145 | +0.006 ± 0.010 |
| 32 | random | 0.9154 | 0.9089 | 0 |
| 32 | tpe | 0.9178 | 0.9245 | +0.016 ± 0.009 |

(± is the standard error of the paired difference over the 12 runs.)

## What to take from it

* **At these budgets the three methods are close.** The best cross-validated score rises slowly with budget for all of
  them (about 0.910 at 8 trials to 0.912-0.918 at 32), and TPE is highest at 16 and 32 trials. The grid is lowest in
  best-CV score at 16 and 32, as expected when only a few values per axis can be tried.
* **Test-set differences are within noise.** The paired test difference against random search is at most 0.016 and the
  standard errors are 0.007 to 0.010, so none of it is distinguishable from zero with 12 runs. The ordering of test
  scores (TPE first at every budget) is suggestive, not established.
* **The "optimism" column here is negative or small** (test accuracy is often above the best CV score) because 3-fold
  CV trains on two thirds of the training part while the final refit uses all of it. A search is optimistic about the
  score it found, but that effect is smaller than this difference in training size on small data.
* Grid search wastes trials on hyperparameters that do not matter (it repeats the same learning rate for every
  combination of the others). Random and TPE try a new value of every hyperparameter on every trial, which is why they
  are generally preferred when only a few of the hyperparameters matter.

## Limits

* Three small datasets, four seeds, one model family, accuracy only. Four hyperparameters and budgets up to 32 are
  small; the gap between methods usually grows with more hyperparameters and larger budgets.
* The grid lattices are one particular choice of levels (centred in each range, not at the edges). A grid built around
  known-good values could do better.
* Optuna's default TPE settings were used; its first trials are random, so at a budget of 8 it is mostly random search.
