# k-means and the silhouette score

Sweeps k (2 to 8) on five datasets and records the inertia (within-cluster sum of squares, always falling as k grows),
the mean silhouette score (how much closer points are to their own cluster than to the next nearest, from -1 to 1) and,
where the true grouping is known, the adjusted Rand index (ARI: 1 = same as the truth, about 0 = chance). Then checks
whether the choice of k can be trusted, what scaling does, and how stable the result is across initialisations.

```
python clustering_lab.py
pytest
```

## Choosing k

| dataset | true k | silhouette picks | elbow picks | ARI at true k | silhouette at true k |
|---|---|---|---|---|---|
| 4 separated blobs | 4 | 4 | 4 | 1.000 | 0.815 |
| 4 touching blobs | 4 | 4 | 4 | 0.752 | 0.458 |
| two moons | 2 | 8 | 4 | 0.266 | 0.486 |
| iris (standardised) | 3 | 2 | 3 | 0.620 | 0.460 |
| wine (standardised) | 3 | 3 | 3 | 0.897 | 0.285 |

* **When the groups are compact and separated, both methods work.** Touching blobs still get the right k but a
  worse match to the truth (ARI 0.752).
* **Two moons: k-means is the wrong tool.** Even at the true k=2 the ARI is 0.266, because k-means cuts space into
  convex regions around centres and the moons interlock. The silhouette's maximum is at k=8, the edge of the sweep, so
  the score is not even pointing at a stable answer. A high silhouette says the clusters are compact, not that they
  are the real groups.
* **Iris: the silhouette prefers 2.** Setosa is far from the other two species and versicolor and virginica touch, so
  the mean silhouette is highest when those two are merged. The elbow gets 3 here, but that is a heuristic, not a rule.
* **The wine silhouette is low (0.285) even though the clustering is good** (ARI 0.897). A low silhouette in many
  dimensions is normal; compare it across k on the same data, not against an absolute scale.

## Scaling matters

On wine at k=3, k-means on the raw columns gives ARI **0.371**; on standardised columns, **0.897**. The `proline`
column spans hundreds to thousands while most others span a few units, so unscaled distance is almost entirely
proline. The unit test shows the extreme case: a pure-noise feature on a huge scale drops ARI to near zero.

## Initialisation matters

Ten runs with a single random start each (`n_init=1`), k equal to the true k:

| dataset | worst pairwise ARI between runs | mean | inertia spread (max/min - 1) |
|---|---|---|---|
| 4 separated blobs | 0.306 | 0.748 | 6.567 |
| 4 touching blobs | 0.996 | 0.998 | 0.000 |
| two moons | 0.980 | 0.989 | 0.000 |
| iris | 0.430 | 0.799 | 0.366 |
| wine | 0.865 | 0.959 | 0.004 |

On the separated blobs some single starts end in a poor local minimum (two blobs merged, another split), with an
inertia up to 6.6 times the best. Taking the best of ten starts (`n_init=10`, the setting used in the sweeps) finds the
true grouping (ARI 1.000). A clustering that was run once and not checked for seed dependence is not a result.

## Limits

* Small, clean datasets with known labels. On real unlabelled data there is no ARI to check against, and the silhouette
  and the elbow are the only guides, with the weaknesses shown above.
* One seed for the data and one sweep range (2 to 8). The two-moons silhouette maximum sits at the edge of that range.
* Only k-means with Euclidean distance. DBSCAN or a Gaussian mixture handle shapes k-means cannot, and are not compared.
