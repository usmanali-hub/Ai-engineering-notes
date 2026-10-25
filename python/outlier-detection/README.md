# Outlier detection: z-score vs IQR

**Goal:** compare three common outlier rules and show the case where the popular one fails.

## What's here

- `outliers.py`: `zscore_outliers`, `iqr_outliers`, `modified_zscore_outliers`
- `test_outliers.py`: a masking test, a robustness test, hand-computed IQR fences and constant-data handling

## Key ideas

- **Z-scores can mask themselves.** The mean and standard deviation are pulled by the very outliers you want to find. With 4 or 10 extreme values, z-scores still catch them all. With 30 values between 150 and 180 among 200 values near 50 (13% of the data), the standard deviation grows so much that a threshold of 3 flags none of them, while IQR and median/MAD flag all 30 (see `test_zscore_is_masked_by_clustered_outliers`).
- **IQR and median/MAD are robust.** Quartiles and the median ignore the extremes, so clustered outliers still stand out.
- **Rules are not truths.** An "outlier" can be a data entry error or the most important sale of the year. Flag first, then decide; never silently drop.
- **Check the degenerate case.** Constant data has zero spread, so a naive z-score divides by zero.

## Run

```bash
pytest python/outlier-detection
python python/outlier-detection/outliers.py
```
