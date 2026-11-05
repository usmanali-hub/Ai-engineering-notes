# Confusion matrix, measured

A confusion matrix counts four outcomes: true positives, false positives, false negatives, true negatives.
`confusion_lab.py` builds it by hand (plain NumPy), checks every number against scikit-learn, and then
shows where the single-number summaries mislead.

## What I measured

**Breast cancer data** (positive = malignant, 171 held-out patients, logistic regression, cutoff 0.5):
TP 60, FP 1, FN 4, TN 106. Accuracy 0.971, precision 0.984, recall 0.938, specificity 0.991, F1 0.960, MCC 0.938.

**The cutoff is a dial.** The model never changes; only which mistake it prefers does.

| cutoff | flagged | false alarms | missed | precision | recall |
|---|---|---|---|---|---|
| 0.05 | 78 | 14 | 0 | 0.821 | 1.000 |
| 0.3 | 63 | 1 | 2 | 0.984 | 0.969 |
| 0.5 | 61 | 1 | 4 | 0.984 | 0.938 |
| 0.9 | 51 | 0 | 13 | 1.000 | 0.797 |

**Cost picks the cutoff.** With a miss costing 1x or 5x a false alarm the best cutoff is 0.243; at 20x it drops to 0.059.
Costs are assumptions I chose, not data.

**The accuracy paradox.** On a synthetic problem with 1% positives (6,000 held-out rows), always answering "no" scores
0.990 accuracy with recall 0 and MCC 0. The model at 0.5 scores 0.991 accuracy, recall 0.083, precision 1.000, MCC 0.287.
Accuracy cannot tell these two apart; recall and MCC can.

## Limits

- One split and one model per dataset, so no error bars. The numbers are a demonstration, not a benchmark.
- On 171 patients a single case moves recall by about 1.6 points.
- The 1% problem is synthetic, with hand-picked class separation.
- Scores are not calibrated probabilities, so "0.5" has no special meaning.

## Run

```bash
python confusion_lab.py
pytest
```
