# Feature scaling: which models care

KNN and SVM compare rows by distance, so a feature measured in the hundreds drowns out one measured in
fractions. Trees split on one feature at a time and never notice. This lab measures it on two real datasets
(`python scaling.py`, about 40 seconds), with every scaler fitted inside the cross-validation pipeline so a
fold never sees its held-out rows. Numbers are mean accuracy over 5-fold CV repeated 3 times.

## Results

**Wine** (178 rows, 13 features). One feature, proline, supplies 99.8% of squared distance; equal influence would be 7.7%.

| model | none | standard | minmax | robust |
|---|---|---|---|---|
| KNN | 0.700 | 0.961 | 0.968 | 0.953 |
| SVM (RBF) | 0.672 | 0.983 | 0.989 | 0.981 |
| logistic regression | 0.949 | 0.983 | 0.985 | 0.985 |
| random forest | 0.981 | 0.981 | 0.981 | 0.981 |

**Breast cancer** (569 rows, 30 features). One feature supplies 70.9% of distance; equal would be 3.3%.

| model | none | standard | minmax | robust |
|---|---|---|---|---|
| KNN | 0.933 | 0.967 | 0.970 | 0.964 |
| SVM (RBF) | 0.917 | 0.974 | 0.977 | 0.971 |
| logistic regression | 0.948 | 0.975 | 0.965 | 0.980 |
| random forest | 0.962 | 0.962 | 0.962 | 0.962 |

**A constructed trap**: two informative features plus one pure-noise feature in the thousands. Unscaled, the noise
supplies 100% of the distance and KNN scores 0.467 (a coin flip); standardised, 0.903.

## What to take from it

- **Scaling is not optional for KNN and SVM.** On wine it moves them from about 0.7 to about 0.97. The random forest is identical to the last digit in every column (a test proves its predictions do not change).
- **Which scaler barely matters.** Standard, min-max and robust land within 3 points of each other, tested. Pick one, and fit it inside the pipeline.
- **Logistic regression is a middle case.** It still reaches a good score unscaled, but gains a few points scaled, and without scaling the solver often stops at its iteration limit.
- **Scaling does not clean outliers, and no scaler is safe across patterns.** Breast-cancer accuracy under four corruption patterns, averaged over three corrupted copies:

  | corruption | model | none | standard | minmax | robust |
  |---|---|---|---|---|---|
  | 5% of all values x20 | KNN | 0.904 | 0.733 | 0.727 | 0.732 |
  | 5% of all values x20 | SVM | 0.865 | 0.725 | 0.715 | 0.702 |
  | 1% of all values x20 | KNN | 0.914 | 0.854 | 0.857 | 0.858 |
  | 1% of all values x20 | SVM | 0.893 | 0.867 | 0.861 | 0.858 |
  | 3% of rows in 10 columns x50 | KNN | 0.919 | 0.927 | 0.947 | 0.887 |
  | 3% of rows in 10 columns x50 | SVM | 0.868 | 0.960 | 0.964 | 0.666 |
  | 3% of rows in 5 columns x100 | KNN | 0.917 | 0.946 | 0.958 | 0.912 |
  | 3% of rows in 5 columns x100 | SVM | 0.822 | 0.972 | 0.971 | 0.643 |

  When damage is spread across many columns, scaling makes things *worse* for every scaler: each corrupted value is still a huge outlier afterwards, and now all 30 features carry equal weight instead of a few large-unit ones. When it is concentrated in a few columns, standard and min-max scaling help. The robust scaler, the one named for outliers, is the worst for SVM there (0.64 to 0.67): it measures spread by the interquartile range, which ignores the outliers, so they are left enormous. Clip or repair outliers before scaling; a forest does not need either (0.950 on the first pattern).

## Limits

Two small benchmark datasets and default hyperparameters (k = 5, default SVM). The outlier results come from four hand-picked corruption patterns on one dataset. They show scaling is
not a fix-all and that scalers can disagree sharply, not which scaler to prefer in general. Accuracy on 178 or 569 rows moves by a point or two between runs.
