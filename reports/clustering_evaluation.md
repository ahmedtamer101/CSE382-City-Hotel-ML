# Fixed K-Means baseline evaluation

## Objective and reproduction

Evaluate and describe the committed untuned City Hotel booking segmentation in the exact feature space used for fitting. Run from the repository root:

```powershell
python -m src.clustering.evaluation --dataset data/H2.csv --output-dir outputs/clustering --sample-size 3000 --seed 42
python -m pytest -q
```

The runner reproduces the Phase 3 fit in memory, checks the dataset SHA-256, parameters, row and feature counts, cluster counts, iterations, and inertia against the committed baseline files, and only then computes evaluation metrics. It retains the fitted sparse feature matrix and source rows in their original positional order. No row or model artifact is saved.

The fixed baseline is `KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42, max_iter=300)`. All 79,330 H2 bookings are fitted, with 31 source variables, 16 engineered fitting fields, and 48 transformed features. Its inertia is `597856.3237440714`, with clusters of 40,174, 33,752, and 5,404 bookings, exactly matching Phase 3. The dataset SHA-256 is `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.

## Internal metrics and computation

| Metric | Actual value | Scope | Meaning |
| --- | ---: | --- | --- |
| Silhouette Score | 0.1213919294 | Fixed 3,000 row stratified sample | Compares each row's average within-cluster distance with its nearest other cluster; larger is usually better. |
| Davies-Bouldin Index | 2.1907288369 | All 79,330 rows | Compares within-cluster scatter with distance between cluster centers; smaller is usually better. |

The silhouette sample was drawn without replacement using seed 42 and fitted cluster labels as strata. It contains 1,519 rows from cluster 0, 1,277 from cluster 1, and 204 from cluster 2. The sample approximately preserves the full cluster proportions. The metric uses the **same scaled and encoded 48-feature Euclidean space** as K-Means. A 3,000 by 3,000 float64 distance array is approximately 72,000,000 bytes (68.7 MiB), versus roughly 50.3 GB for a full 79,330 by 79,330 array. The reported silhouette is a sample estimate and can vary with the chosen sample even when the fit is fixed.

Davies-Bouldin uses a dense copy of the complete 79,330 by 48 matrix because the estimated float64 feature array is 30,462,720 bytes (29.05 MiB), below the configured 64 MiB array budget. This budget addresses the explicit matrix conversion, not total process peak memory or library temporaries. If a future feature matrix exceeds the budget, the code uses a bounded stratified sample and records that scope. It rejects samples that cannot represent at least two clusters. The silhouette sample has a hard upper limit of 3,000 rows.

Neither metric is an accuracy percentage or a proof that `k=3` is best. Inertia measures summed squared distances to assigned centroids and generally falls as `k` grows, so inertia alone cannot choose a cluster count. This phase does not compare alternatives.

## Original-scale cluster profiles

Quartiles are 25th–75th percentiles. The numerical values below come from unscaled booking characteristics, with labels joined by row position. Proportions are within each cluster.

| Cluster | Bookings | LeadTime median [Q1, Q3] | TotalNights median [Q1, Q3] | PartySize median | WeekendShare median | Top MarketSegment | Top DistributionChannel |
| ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| 0 | 40,174 (50.64%) | 79 [30, 153] days | 3 [3, 4] nights | 2 | 0.40 | Online TA 61.03% | TA/TO 90.18% |
| 1 | 33,752 (42.55%) | 57 [13, 150] days | 2 [1, 3] nights | 2 | 0.00 | Online TA 40.84% | TA/TO 82.76% |
| 2 | 5,404 (6.81%) | 239 [68, 312] days | 2 [2, 3] nights | 2 | 0.00 | Groups 53.65% | TA/TO 88.53% |

Additional observed distinctions:

| Cluster | Meal | Deposit | Guest history | Reserved room | Zero-person / zero-night records |
| ---: | --- | --- | --- | --- | ---: |
| 0 | BB 77.97% | No Deposit 91.02% | Repeated guest 0.56%; previous cancellations median 0 | A 71.92%, D 19.70% | 55 / 0 |
| 1 | BB 77.62% | No Deposit 82.17%; Non Refund 17.81% | Repeated guest 3.14%; previous cancellations median 0 | A 84.54%, D 11.06% | 107 / 326 |
| 2 | BB 88.53%, HB 10.86% | Non Refund 60.34% | Repeated guest 13.82%; previous cancellations median 1 [1, 1] | A 95.58% | 5 / 5 |

Cluster 0 mainly groups three-to-four-night bookings with a larger weekend share, a high Online TA proportion, and mostly no deposit. Cluster 1 tends toward shorter, weekday oriented bookings and includes nearly all zero-night records. Cluster 2 has a substantially longer planning horizon and disproportionately many `Groups`, `Non Refund`, repeated guest, and `Contract` bookings: `Groups` is 53.65% versus 17.62% overall, and `Contract` is 16.67% versus 2.90% overall. These are associations within the fitted groups. They do not identify unique customers or explain why bookings have these properties.

The full original-scale quartiles, top two categories, and zero-row counts are in `outputs/clustering/cluster_profiles.csv`. The separate category table contains counts, within-cluster percentages, overall percentages, and lift for every observed level of MarketSegment, DistributionChannel, Meal, DepositType, IsRepeatedGuest, ReservedRoomType, and CustomerType. A lift above 1 means a category is more common in that cluster than in the dataset overall.

## Leakage and alignment

The model and internal metrics use only the Phase 2 fitting matrix. `IsCanceled`, `ADR`, `ReservationStatus`, `ReservationStatusDate`, `AssignedRoomType`, `BookingChanges`, and `DaysInWaitingList` do not enter the fit or metrics. This report does not calculate optional outcome comparisons. Source data are preserved for profiling, and positional joins keep repeated DataFrame index labels and duplicate booking rows aligned with their labels. No supervised final test set is accessed.

## Limitations and remaining work

- H2 may include booking-lifecycle updates, so the groups cannot be described as guaranteed initial booking snapshots.
- Euclidean distance is sensitive to numerical scaling and the number of one-hot fields. `MarketSegment` and `DistributionChannel` can overlap conceptually; guest-history fields may also overlap.
- The sampled silhouette is an estimate tied to seed 42 and the observed cluster proportions. Rare or boundary patterns may be underrepresented.
- Cluster IDs have no intrinsic order, and small differences in percentages should not be overstated. The 6.81% group needs further stability checks.
- The current metrics describe one fixed fit; they do not establish an optimal cluster count or demonstrate stability across seeds, time, or algorithms.
- Member 1 still owns the full EDA decisions on duplicate records, zero-person and zero-night rows, and unusual values. Member 2 owns shared data and supervised split protocols.

Phase 5 will add the required two-dimensional visualization. Subsequent Milestone 2 work must investigate cluster count and stability, compare at least three clustering algorithms, optimize each using unsupervised criteria, and produce a dendrogram where practical. No outcome variable should guide those choices.
