# ai-engineering-notes

Small, runnable experiments and short write-ups from my AI engineering practice. Each folder has
code, tests and a README with what I tried, what I learned, and how to run it.

## Index

| Area | Note | What it covers |
|---|---|---|
| Python | [Config and logging](python/config-and-logging) | Typed settings from environment variables, safe secret handling, logging setup |
| Python | [Decorators: timing and retry](python/decorators-timing-retry) | A timing decorator and a retry decorator with exponential backoff |
| Python | [Streaming chunks](python/streaming-chunks) | Generators and context managers for processing large inputs lazily |
| Python | [Strict typing](python/strict-typing) | Common mypy strict errors and how to fix them |
| NumPy | [Vectorization benchmark](python/vectorization-benchmark) | Python loops vs NumPy, with measured speedups |
| SQL | [Window function cheat sheet](sql/window-functions) | Eight runnable window-function queries with hand-checked results |
| SQL | [Index vs no index](sql/index-explain) | EXPLAIN QUERY PLAN before and after an index: scan vs search, ~200x faster lookups, when an index cannot help, and the write cost |
| SQL | [Recursive CTEs](sql/recursive-cte) | Series, org chart, bill of materials, cycle-safe graph reachability and calendar gap filling, each checked against plain Python recursion |
| SQL | [Parameterized queries](sql/parameterized-queries) | Placeholders vs string formatting: injection and UNION-leak demos, IN lists, LIKE wildcard escaping, whitelisted sort columns, rollback on failure |
| ML | [Bias-variance, measured](ml/bias-variance) | Bias², variance and noise² estimated from 300 resampled training sets, checked against directly measured test error; with the plot |
| ML | [Confusion matrix, measured](ml/confusion-matrix) | Hand-counted TP/FP/FN/TN checked against scikit-learn on real breast cancer data: precision, recall, specificity, F1 and MCC, the cutoff as a dial, a cost-based cutoff, and why 'always no' scores 99% accuracy on a 1% event |
| ML | [Class imbalance](ml/class-imbalance) | Class weights vs SMOTE vs doing nothing, two models on ten datasets at two positive rates: recall at 0.5 rises sharply, but ranking (PR-AUC) almost never improves and scores stop being probabilities |
| ML | [Feature selection compared](ml/feature-selection) | ANOVA, mutual information, RFE, lasso and forest importance on ten synthetic datasets with known true columns (paired comparisons) and on breast cancer: selection beats 100 columns on 300 rows, forest importance is the one clear winner, none beats all 30 columns on real data, and the chosen sets are unstable |
| ML | [Learning curves](ml/learning-curves) | Train and validation accuracy against training size for four models on noisy synthetic data and digits: tells high bias (curves together, flat, more data does nothing) from high variance (large gap), and shows a big gap does not make a model bad (the forest has the best validation score despite the second-largest gap) |
| ML | [SHAP values from first principles](ml/shap-values) | Exact Shapley values by subset enumeration, tested for efficiency, linear-model, dummy, symmetry and interaction properties, then compared with LightGBM TreeSHAP (correlation 0.996): noise columns still get credit if the model uses them, and a correlated copy splits credit |
| ML | [Hyperparameter search: grid vs random vs Optuna](ml/hyperparameter-search) | Gradient boosting tuned by a grid, random search and Optuna TPE at budgets of 8, 16 and 32 trials on three datasets and four seeds, scored on held-out test data: TPE comes out ahead at every budget but the paired differences (up to 0.016, standard error about 0.009) are within noise, and grid search is lowest in cross-validated score |
| ML | [k-means and the silhouette score](ml/clustering) | Choosing k on five datasets with known labels: silhouette and elbow agree on compact blobs, fail on two moons (ARI 0.27 even at the true k) and merge iris species; scaling changes wine from ARI 0.37 to 0.90; single random starts end in poor minima that n_init=10 avoids |
| ML | [PCA for visualisation](ml/pca-visualization) | What the first two components keep: standardising turns wine from a 1-D plot (proline carries 100% of PC1) into a real one; 2D PCA classifies wine as well as all 13 columns (0.961) but digits only 0.550 vs 0.976; PCA does not beat two random columns on iris; a low-variance class signal is dropped by 1D PCA (0.50) and kept by LDA (1.00) |
| ML | [Saving models: joblib vs ONNX](ml/joblib-vs-onnx) | Same pipeline saved both ways: ONNX is smaller and 30-100x faster for single-row prediction; the logistic model converts exactly, while the forest differs by up to 0.026 on 17-22% of rows, the same as scikit-learn's own forest moves under float32-sized noise (the converted trees alone agree to 3e-7); loading a pickle runs code, loading ONNX does not |
| ML | [Reproducibility: seeds and data versioning](ml/reproducibility) | One split seed moves test accuracy by several points (0.71 to 0.85 on noisy data) and more than the model's own seed; best-of-10-seeds claims +3 to +5 points that vanish on untouched data (+0.001 and +0.002); the same seed with different thread counts agrees to 1e-16, not bit for bit; a dataset content hash catches a 2% edit that the accuracy did not show |
| ML | [Churn project write-up](ml/churn-project-writeup) | A blog-style summary of the ml-end-to-end repo: fix the test set first, report ties as ties, spend the test set once (PR-AUC 0.665 vs 0.674 in cross-validation), profile before optimising, and a list of what the project does not prove; a test checks the quoted figures appear in the text |
| LLM | [Tokenizer exploration](llm/tokenizer-exploration) | A byte-level BPE trained from scratch: English costs 0.41 tokens per byte, code and JSON 0.58 to 0.65, a 64-digit number almost one token per digit, and Urdu, Chinese and emoji one token per byte (no merges learned); each doubling of the vocabulary saves fewer tokens; not a real model's counts |
| LLM | [Temperature vs top-p](llm/sampling-temperature-top-p) | On three toy next-token distributions: temperature reshapes the whole distribution but never removes a token, top-p cuts the tail and keeps 1 token on a confident distribution but 10 on an open-ended one (top-k 5 keeps 5 on both), and applying temperature before or after top-p gives different results; no real model involved |
| ML | [Train/test leakage](ml/leakage) | Five leaks measured next to their fixes: selection before CV turns noise into 88% accuracy, a column that follows the outcome, the same customer on both sides, and a scaler fitted on all rows (barely matters) |
| ML | [Feature scaling](ml/scaling) | KNN, SVM, logistic regression and a forest compared under four scalers on real data: scaling moves KNN and SVM from about 0.7 to 0.97 on wine, trees do not care, and scaling does not rescue outliers |
| ML | [Ridge vs Lasso](ml/regularization) | Both penalties on worlds with known truth: Lasso wins when 5 of 50 features matter (error 1.23 vs 1.77), Ridge wins when all matter a little, and on a real 10-feature dataset neither helps |
| pandas | [Clean a messy CSV](python/pandas-clean-csv) | Header, text, currency, date and duplicate cleanup in one short chain |
| pandas | [Time-series resampling](python/timeseries-resampling) | Tick prices to OHLC bars, log returns and rolling volatility |
| Visualization | [Matplotlib chart set](python/matplotlib-chart-set) | Line, bar and histogram helpers with one consistent, honest style |
| NumPy | [Outlier detection](python/outlier-detection) | z-score vs IQR vs median/MAD, and why z-scores can mask outliers |

## Run everything

```bash
pip install -r requirements-dev.txt
pytest
```
