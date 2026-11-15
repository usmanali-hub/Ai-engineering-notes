# What a careful churn model project looks like, and what it does not prove

A write-up of [ml-end-to-end](https://github.com/usmanali-hub/ml-end-to-end): a customer-churn model on the public Telco
dataset (7,043 customers, 26.5% churn), built so that every number can be traced to a command and a test.

## 1. Fix the test set first, then leave it alone

The data was split once (stratified, seed 42) into 5,634 training rows and 1,409 test rows. Every choice afterwards used
5-fold cross-validation on the training rows only. A guard in the experiment tracker refuses to log runs on data that
contains test customers, and the final evaluation command refuses to run twice.

## 2. Choose the metric before the model

About one customer in four churns, so a model that flags nobody is 73.5% accurate. Precision-recall AUC was the primary
metric, with a base-rate baseline (0.265) as the floor every model has to beat.

## 3. Report ties as ties

| model | cross-validation PR-AUC |
|---|---|
| logistic regression (L2, C = 0.03) | 0.674 |
| LightGBM | 0.672 |
| neural network (MLP) | 0.672 |
| random forest | 0.666 |

The gaps are about a tenth of the fold-to-fold standard deviation (0.018 to 0.022). Repeating cross-validation over several
seeds showed that the choice of folds moves the score about as much as most of these differences. (In the separate
[hyperparameter search lab](../hyperparameter-search), a smarter search beat random search by only about 0.007.) The
simplest model, logistic regression, was kept as primary.

## 4. Spend the test set once

On the held-out rows the three chosen models scored 0.665, 0.662 and 0.663 PR-AUC (95% bootstrap interval roughly 0.61 to
0.71). That is 0.004 to 0.009 under cross-validation, the small optimism of choosing on the same data. The
models are still tied. At the cutoff picked from training scores (0.33) the logistic model caught 275 of the 374 churners
among 493 flagged customers.

## 5. Make it runnable by someone else

Models are saved with a SHA-256 version, the library versions they were trained with, and the cutoff. Loading checks the
hash and refuses a mismatched library version, because a pickle runs code when it is loaded. An ONNX export is refused
unless it matches the pickle on the training rows, and `predict --runtime onnx` never unpickles. Input validation takes
about 3 ms for one customer after profiling showed it, not the model, was the slow part (the ONNX model itself takes
about 0.4 ms). Those timings are one shared 2-core machine, in-process, with no network.

## What this does not show

* One public tutorial dataset. `TotalCharges` is almost `MonthlyCharges` x `tenure`, which real billing rarely is, so the
  scores show the method, not what a model would do at a real telecom.
* One test split of 1,409 rows is one draw; the interval is about the sample, not about a different company.
* No deployed service, monitoring, or drift detection. The latency numbers are for the scoring function only.
* Ties mean the extra complexity of boosting or a neural net bought nothing here. That is a finding about this dataset.

## Lessons worth reusing

1. Decide the metric and the split before looking at models.
2. Quote the noise next to every comparison; if the gap is smaller than the noise, say it is a tie.
3. Profile before optimising: the 9.5 ms was in input validation, not in the model.
4. Make the unglamorous parts (hashing, version checks, refusing to rerun) part of the code, with tests.
