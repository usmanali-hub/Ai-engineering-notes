# Feature selection compared

Filter (ANOVA F-test, mutual information), wrapper (recursive feature elimination) and embedded (lasso, random-forest
importance) selection, all run *inside* the cross-validation pipeline so the selector only sees training folds.
`selection_lab.py` checks them on synthetic data where the true columns are known and on a real dataset where they are not.

## Synthetic data (mean over five datasets)

300 rows, 100 columns: columns 0-4 carry the signal, 5-9 are redundant mixtures of them, 90 are pure noise. Each
method chooses 10 columns; the classifier is logistic regression; ROC-AUC from 5-fold CV repeated 3 times.

| method | ROC-AUC | informative found (of 5) | redundant found (of 5) | noise picked (of 10) | stability | beat all columns |
|---|---|---|---|---|---|---|
| all 100 columns | 0.820 | - | - | - | - | - |
| ANOVA F-test | 0.904 | 3.0 | 3.8 | 3.2 | 0.51 | 5 of 5 |
| mutual information | 0.909 | 2.4 | 4.4 | 3.2 | 0.30 | 5 of 5 |
| RFE (logistic) | 0.895 | 3.2 | 2.4 | 4.4 | 0.36 | 5 of 5 |
| lasso | 0.894 | 3.2 | 1.4 | 5.4 | 0.33 | 5 of 5 |
| random forest importance | 0.916 | 4.4 | 5.0 | 0.6 | 0.63 | 5 of 5 |

- **With 100 columns and 300 rows, any selection beats keeping everything** (0.82 to 0.89-0.92), in every dataset.
- **The forest importance picks found the most true columns and the least noise**, at about 3 seconds per dataset
  against 0.1 seconds for ANOVA. The spread between selectors (0.894 to 0.916) is about the size of the spread
  between datasets (0.014 to 0.020), so their order is not established; the gap to using every column is.
- **Stability is poor.** Stability is the mean Jaccard similarity of the chosen set across bootstrap resamples (1 means
  identical): 0.30 to 0.63, so on 300 rows the exact columns chosen change a lot with the sample. A selected list
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
- Five datasets show a consistent direction but do not rank the selectors against each other.
- Stability is measured on 300 rows; with more data it would rise.

## Run

```bash
python selection_lab.py   # about 2 minutes
pytest
```
