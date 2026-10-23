# Clean a messy CSV with pandas

**Goal:** turn a messy CSV (padded headers, mixed case, currency strings, duplicate rows, a missing
value) into a tidy DataFrame in a handful of readable lines.

## What's here

- `clean_csv.py`: `clean()` does the whole job; running the file prints the result
- `test_clean_csv.py`: checks column names, deduplication, dtypes and the missing-fee rule

## Key ideas

- **Fix column names first.** Everything after that can use clean, predictable names.
- **Strip, then change case.** `str.strip()` before `str.title()` or `str.lower()` stops `' PRO'` and `'pro'` becoming different categories.
- **`errors="coerce"` makes bad numbers visible** as NaN instead of crashing. Decide deliberately what a missing value means; here a missing fee is filled with 0.0 because the plan is free.
- **Drop duplicates after normalizing**, not before, otherwise near-duplicates survive.
- **`format="mixed"` parses each date on its own.** With `dayfirst=False`, an ambiguous value like `05/02/2026` is read as May 2. Know your source's convention, or use explicit formats as in the [data-analysis-pipeline](https://github.com/usmanali-hub/data-analysis-pipeline) cleaning module.

## Run

```bash
pytest python/pandas-clean-csv
```
