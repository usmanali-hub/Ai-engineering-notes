# Hyperparameter search: grid, random and Optuna at the same budget

Tunes a gradient boosting model (four hyperparameters: learning rate, leaf count, L2 penalty, minimum leaf size) on
three small datasets (breast cancer, wine, a noisy synthetic problem) with three methods, each given 8, 16 or 32
trials. A trial is a 3-fold cross-validated accuracy. The best configuration is then refitted on the whole training
part and scored on a test part that no search saw. Twelve seeds (each changes the split and the random draws).

* **grid**: evenly spaced values per hyperparameter (log-spaced for the ranges spanning orders of magnitude), a lattice
  with exactly the budget's number of points (2x2x2, 2x2x2x2, 4x2x2x2);
* **random**: independent draws from the same ranges;
* **tpe**: Optuna's Tree-structured Parzen Estimator.

```
python search_lab.py      # about 15 to 20 minutes on 2 cores
pytest
```

## Results (mean over 3 datasets x 12 seeds = 36 runs per cell)

| budget | method | best CV accuracy | test accuracy | test minus random (paired) |
|---|---|---|---|---|
| 8 | grid | 0.9083 | 0.9248 | +0.007 ± 0.004 |
| 8 | random | 0.9112 | 0.9174 | 0 |
| 8 | tpe | 0.9099 | 0.9221 | +0.005 ± 0.004 |
| 16 | grid | 0.9117 | 0.9213 | +0.002 ± 0.003 |
| 16 | random | 0.9141 | 0.9189 | 0 |
| 16 | tpe | 0.9144 | 0.9248 | +0.006 ± 0.003 |
| 32 | grid | 0.9135 | 0.9210 | +0.001 ± 0.004 |
| 32 | random | 0.9156 | 0.9200 | 0 |
| 32 | tpe | 0.9175 | 0.9269 | +0.007 ± 0.004 |

(± is the standard error of the paired difference over the 36 runs.)

An earlier version of this lab used 4 seeds and showed TPE ahead of random by +0.016 at a budget of 32. With 12 seeds
that gap is +0.007: the larger number was mostly noise, which is the reason the lab now uses more seeds.

## What to take from it

* **At these budgets the three methods are close.** The best cross-validated score rises slowly with budget for all of
  them (0.908-0.912 at 8 trials to 0.914-0.918 at 32). TPE has the highest best-CV score at 16 and 32 trials and the grid
  the lowest at every budget, as expected when only a few values per axis can be tried.
* **Test-set differences are small and mostly not distinguishable from zero.** The largest paired difference against random
  search is +0.007, with standard errors of 0.003 to 0.004. TPE's +0.007 at a budget of 32 is about 2 standard errors, but
  six comparisons were made, so one near 2 is what chance produces; and the grid is ahead of random at a budget of 8 (+0.007)
  and level at 32, which no one would predict from theory. TPE ahead of random at every budget is suggestive, not established.
* **The "optimism" column here is negative** (test accuracy is above the best CV score for every method) because 3-fold
  CV trains on two thirds of the training part while the final refit uses all of it. A search is optimistic about the
  score it found, but that effect is smaller than this difference in training size on small data.
* Grid search wastes trials on hyperparameters that do not matter (it repeats the same learning rate for every
  combination of the others). Random and TPE try a new value of every hyperparameter on every trial, which is why they
  are generally preferred when only a few of the hyperparameters matter.

## Limits

* Three small datasets, twelve seeds, one model family, accuracy only. Four hyperparameters and budgets up to 32 are
  small; the gap between methods usually grows with more hyperparameters and larger budgets.
* The grid lattices are one particular choice of levels (centred in each range, not at the edges). A grid built around
  known-good values could do better.
* Optuna's default TPE settings were used; its first trials are random, so at a budget of 8 it is mostly random search.
