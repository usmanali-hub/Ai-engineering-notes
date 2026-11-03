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
- **Scaling does not clean outliers, and can make things worse.** In a stress test where 5% of all values are multiplied by 20, scaling *lowered* KNN accuracy from 0.907 to about 0.74 for all three scalers, including the robust one. A scaler rescales; a value 20 times too large is still 20 times too large afterwards. Unscaled KNN looked better there only because the few large-unit features it already relies on happened to be the strong predictors. Clip or fix outliers first, or use a model that does not care, as the forest (0.950) does not.

## Limits

Two small benchmark datasets and default hyperparameters (k = 5, default SVM). The outlier result is one
corruption pattern, and a different pattern would favour different scalers, so treat it as a warning that scaling
is not a fix-all rather than a ranking of scalers. Accuracy on 178 or 569 rows moves by a point or two between runs.
