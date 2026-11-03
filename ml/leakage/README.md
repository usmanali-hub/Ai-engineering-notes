# Train/test leakage: four cases, measured

Leakage is when information from the data used to *judge* a model reaches the process that *builds* it.
The score then looks better than the model will ever do on new data. Each case below runs the leaky
version and the fixed one side by side (`python leakage.py`, about five seconds).

| Case | What goes wrong | Leaky score | Honest score | Fix |
|---|---|---|---|---|
| **Feature selection before cross-validation** | Keep the 20 features most correlated with the labels, *then* cross-validate. Data: 100 rows, 5,000 pure-noise features, coin-flip labels (truth: 50%) | 87.7% | 50.5% | Put selection inside a `Pipeline`, so every fold picks features from its own training rows |
| **A column that follows the outcome** | `cancellation_ticket` is filed *because* a customer cancels; it does not exist when a retention model would run | AUC 0.977 | AUC 0.667 | Drop columns that are consequences of the target. A holdout split cannot catch this: the column is in the test rows too |
| **Same customer in train and test** | Several rows per customer; a random forest memorises each customer's fingerprint. Labels are random per customer (truth: 50%) | 98.8% | 51.1% | `GroupKFold`: split by customer, never by row |
| **Selection on real data** | Same selection mistake, but on real breast-cancer rows with 3,000 junk columns added (60 rows) | 93.0% | 93.0% | Same pipeline fix. Here it changes nothing, because the real columns are strong enough that selection finds them either way |
| **Scaler fitted on all rows** | Standardise before splitting (breast-cancer data, SVC) | 97.54% | 97.71% | `make_pipeline(StandardScaler(), model)` |

## What to take from it

- **The inflation can be enormous.** The first case turns a coin flip into an apparent 88% model. With thousands of features and a hundred rows this is a common way to publish results that do not replicate.
- **It depends on the setting.** Rerun the first case with 2,000 rows and 50 features and the gap falls below 3 points (tested): leakage is worst when features outnumber rows.
- **Not every leak is dramatic, and the honest number is not always lower.** The scaler leak is wrong and free to fix, but here it changes accuracy by well under a point, and in this run the leaky score is even slightly *lower*. The lesson is to fix it by habit, not to expect a big swing.
- **No split protects against target leakage.** Random hold-out leaves the leaked column in both halves. The defence is asking, for every feature, "would I know this at the moment the prediction is made?"

## How it is checked

`pytest ml/leakage` verifies each gap with tolerances loose enough for any machine (for example "leaky
above 75%, honest within 8 points of 50%"), and that the pipeline fix equals hand-written per-fold selection
to 12 decimal places, that `GroupKFold` never puts a customer on both sides while random folds nearly always do,
and that results are reproducible for a seed.

## Limits

Three of the five cases use synthetic data, built so the truth is known (50% for random labels). That
is what makes the leak measurable, and the two real-data cases (scaler, selection) show the leak is small when there is strong real signal, but real datasets rarely offer a known ground truth, so there the defence
has to be process: split first, fit every transformation on training rows only, and split by the entity
(customer, patient, device) rather than by row.
