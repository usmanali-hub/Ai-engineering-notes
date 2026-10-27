# Matplotlib chart set

**Goal:** three reusable chart helpers (line, bar, histogram) that share one style and avoid the
common ways charts mislead.

## What's here

- `charts.py`: `line_chart`, `bar_chart`, `histogram` and a `demo()` that writes three PNGs
- `test_charts.py`: checks the properties that matter (sorted bars, zero baseline, direct labels, median marker) rather than pixels

## Key ideas

- **Bars start at zero.** Bar length encodes value, so a truncated axis exaggerates differences. `bar_chart` fixes the left limit at 0.
- **Sort bars by value** so the ranking is readable at a glance.
- **Label lines directly** at their last point instead of using a legend the eye has to look up. One series in colour and the rest muted keeps focus on the story.
- **Mark the median on histograms.** For skewed data (order values, latencies) the median describes a typical value better than the mean.
- **Use the `Agg` backend for scripts and CI.** It needs no display, so the same code runs on a laptop and in GitHub Actions.
- **Test chart properties, not images.** Asserting limits, labels and ordering catches real regressions without brittle pixel comparisons.

## Run

```bash
pytest python/matplotlib-chart-set
python python/matplotlib-chart-set/charts.py   # writes line.png, bar.png, hist.png
```
