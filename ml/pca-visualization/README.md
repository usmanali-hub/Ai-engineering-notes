# PCA for visualisation

PCA rotates the data so the first axis holds the most variance, the second the most of what remains, and so on.
Plotting the first two is the standard way to look at high-dimensional data. This lab measures what that picture keeps,
what it throws away, and when it misleads, on iris (4 features), wine (13) and the 8x8 digits (64).

```
python pca_lab.py        # prints every table below
pytest
```

`pca_lab.plot(path)` saves the three two-component scatter plots as a PNG.

## How much variance the leading components hold (features standardised)

| dataset | features | PC1 | PC1+PC2 | components for 90% | for 95% |
|---|---|---|---|---|---|
| iris | 4 | 0.730 | 0.958 | 2 | 2 |
| wine | 13 | 0.362 | 0.554 | 8 | 10 |
| digits | 64 | 0.120 | 0.216 | 31 | 40 |

A two-dimensional picture of iris keeps almost everything. For wine it keeps 55% of the variance and for digits 22%, so
the plot shows a small slice of the data, and there is no rule that the missing slice is unimportant.

## Scaling changes the picture

| dataset | scaled? | PC1 variance share | column with the largest PC1 loading | its share of PC1 |
|---|---|---|---|---|
| wine | no | 0.998 | proline | 1.000 |
| wine | yes | 0.362 | flavanoids | 0.179 |
| iris | no | 0.925 | petal length | 0.734 |
| iris | yes | 0.730 | petal length | 0.337 |

Proline is measured in hundreds or thousands and the other wine columns in single digits, so without scaling PC1 is
essentially the proline column and the "13-dimensional" plot is a 1-dimensional one. PCA ranks directions by variance,
and variance depends on units. Standardise first unless the units are comparable on purpose.

## What a 2D view keeps (5-fold CV, 5-NN accuracy; scaling and PCA are fitted inside each training fold)

| dataset | variance in 2D | accuracy, all features | accuracy, 2 PCA axes | accuracy, 2 random columns | trustworthiness (k=10) |
|---|---|---|---|---|---|
| iris | 0.958 | 0.940 | 0.907 | 0.907 | 0.978 |
| wine | 0.554 | 0.961 | 0.961 | 0.728 | 0.888 |
| digits | 0.216 | 0.976 | 0.550 | 0.249 | 0.816 |

* **Wine: two PCA axes classify as well as all 13 columns (0.961), far better than two random columns (0.728).** PCA picks
  directions worth looking at.
* **Digits: the 2D view loses most of what separates the classes** (0.550 against 0.976 with all pixels). Two random pixels
  are close to useless (0.249; chance is 0.1), so PCA is far better than nothing, but a scatter plot of digits shows
  overlapping clouds that are not as overlapping in the full data.
* **Iris: PCA does not beat two random columns** (both 0.907). With four strongly related columns, almost any pair works.
  PCA is not automatically better than picking two features; the margin has to be measured.
* Trustworthiness (1 = every point's nearest neighbours in the plot were its neighbours in the data) falls from 0.978 to
  0.816 as the retained variance falls, the same story from a label-free angle.

## The trap: variance is not class information

Two classes that differ only along a column with standard deviation 0.5 (separated by 4 units, 8 standard deviations),
hidden under a nuisance column with standard deviation 10 that carries no class information. CV accuracy of 5-NN after
projecting to one dimension:

| projection | accuracy |
|---|---|
| PCA, 1 component (keeps the nuisance column; holds 96% of the variance) | 0.500 |
| LDA, 1 component (uses the labels) | 1.000 |
| PCA, 2 components (nothing dropped) | 0.990 |

PCA is unsupervised: it cannot know which variance matters. When you drop components you can drop the signal, and the
plot then looks like a structureless cloud. If the goal is class separation, use a supervised projection or check the
accuracy as above.

## Reconstruction (digits)

| components | mean squared error | variance lost |
|---|---|---|
| 2 | 13.42 | 0.715 |
| 5 | 8.54 | 0.455 |
| 10 | 4.91 | 0.262 |
| 20 | 1.98 | 0.106 |
| 40 | 0.22 | 0.012 |

The reconstruction error divided by the total variance equals the variance lost (tested), so the scree is also the
error budget. Forty of 64 components keep 99% of the pixels' variance.

## Limits

* Three small, clean datasets. The numbers show the behaviour, not how much it matters on your data.
* PCA is linear. Curved structure (two moons, say) needs other methods, and trustworthiness is only one view of that.
* The accuracy columns use 5-NN with k fixed at 5 and no tuning, so they compare projections, not classifiers.
* Using the first two components for a plot is a choice of display, not evidence that two dimensions are enough.
