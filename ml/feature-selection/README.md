# Feature selection compared

Filter (ANOVA F-test, mutual information), wrapper (recursive feature elimination) and embedded (lasso, random-forest
importance) selection, all run *inside* the cross-validation pipeline so the selector only sees training folds.
`selection_lab.py` checks them on synthetic data where the true columns are known and on a real dataset where they are not.

## Synthetic data (mean over ten datasets)

300 rows, 100 columns: columns 0-4 carry the signal, 5-9 are redundant mixtures of them, 90 are pure noise. Each
method chooses 10 columns; the classifier is logistic regression; ROC-AUC from 5-fold CV repeated 3 times, on ten
independently drawn datasets.

| method | ROC-AUC | informative found (of 5) | redundant found (of 5) | noise picked (of 10) | stability | beat all columns |
|---|---|---|---|---|---|---|
| all 100 columns | 0.804 | - | - | - | - | - |
| ANOVA F-test | 0.875 | 2.9 | 3.8 | 3.3 | 0.51 | 10 of 10 |
| mutual information | 0.876 | 2.5 | 3.8 | 3.7 | 0.28 | 9 of 10 |
| RFE (logistic) | 0.872 | 3.4 | 2.5 | 4.1 | 0.36 | 10 of 10 |
| lasso | 0.868 | 3.0 | 1.4 | 5.6 | 0.32 | 10 of 10 |
| random forest importance | 0.890 | 4.4 | 4.8 | 0.8 | 0.61 | 10 of 10 |

AUC varies a lot between datasets (sd 0.06 to 0.09), so methods are compared on the *same* dataset. Difference from
ANOVA, paired over the ten datasets:

| method | mean AUC difference | standard error | higher than ANOVA on |
|---|---|---|---|
| mutual information | +0.001 | 0.004 | 7 of 10 |
| RFE (logistic) | -0.003 | 0.008 | 2 of 10 |
| lasso | -0.006 | 0.008 | 2 of 10 |
| random forest importance | **+0.015** | 0.005 | **10 of 10** |

- **With 100 columns and 300 rows, almost any selection beats keeping everything** (0.80 to 0.87-0.89): 10 of 10
  datasets for four methods, 9 of 10 for mutual information.
- **Forest importance is the only method clearly ahead of the others**: +0.015 over ANOVA, higher in every dataset, and
  it found the most true columns while picking the least noise, at about 3.6 seconds per dataset against 0.13 for ANOVA.
  ANOVA, mutual information, RFE and lasso are not separated from each other (differences within about two standard errors).
- **Stability is poor.** Stability is the mean Jaccard similarity of the chosen set across bootstrap resamples (1 means
  identical): 0.28 to 0.61, so on 300 rows the exact columns chosen change a lot with the sample. A selected list
  should not be read as "these are the important features".

## Real data: breast cancer, 30 columns

| method | 3 columns | 5 columns | 10 columns |
|---|---|---|---|
| ANOVA F-test | 0.986 | 0.987 | 0.988 |
| mutual information | 0.975 | 0.988 | 0.989 |
| RFE (logistic) | 0.987 | 0.993 | 0.995 |
| lasso | 0.989 | 0.992 | 0.993 |
| random forest importance | 0.985 | 0.988 | 0.988 |
| **all 30 columns** | **0.995** | | |

Here selection never beats using everything: with 569 rows and 30 columns, logistic regression is not hurt by the
extra columns. Selection pays off when columns greatly outnumber rows, not by default. RFE with 10 columns ties it.

## Limits

- The synthetic data has independent noise columns and a known linear signal; real noise is correlated.
- One classifier, one set size (10) for the synthetic data, default settings for each selector.
- Ten datasets separate forest importance from the rest but not the other four from each other; a different number of selected columns could change the ranking.
- Stability is measured on 300 rows; with more data it would rise.

## Run

```bash
python selection_lab.py   # about 2 minutes
pytest
```
