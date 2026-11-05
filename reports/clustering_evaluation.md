# Revised K-Means Baseline Evaluation

## Reproduction

```powershell
python -m src.clustering.evaluation --dataset data/H2.csv --output-dir outputs/clustering --sample-size 3000 --seed 42
```

The workflow reproduces the fixed fit, verifies checksum, parameters, dimensions, counts, iterations, and inertia against the baseline files, then evaluates the same 45-dimensional matrix. Labels are joined to preserved booking rows by position, including when DataFrame index labels repeat.

## Internal metrics

| Metric | Actual value | Scope |
| --- | ---: | --- |
| Silhouette Score | 0.1849722777 | Fixed proportional stratified sample of 3,000 rows |
| Davies-Bouldin Index | 2.0046722671 | All 79,330 rows |

The silhouette sample uses seed 42 without replacement: 743 rows from cluster 0, 2,053 from cluster 1, and 204 from cluster 2. Its pairwise distance array is estimated at 72,000,000 bytes. This is a sample estimate, not an accuracy percentage.

Davies-Bouldin uses the full 79,330 × 45 matrix after a memory check. Its dense equivalent is 28,558,800 bytes, below the configured 64 MiB explicit-array budget. If a future matrix exceeds that budget, the code uses a bounded stratified sample and records the scope.

Neither metric proves that `k=3` is optimal. No cancellation outcome was used to optimize the model or select a cluster count.

## Original-scale profiles

| Cluster | Bookings | LeadTime median [Q1, Q3] | TotalNights median [Q1, Q3] | PartySize median | WeekendShare median | Top market segment | Top channel |
| ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| 0 | 19,651 (24.77%) | 8 [2, 25] days | 1 [1, 2] nights | 2 | 0.00 | Online TA 46.86% | TA/TO 67.94% |
| 1 | 54,289 (68.43%) | 104 [53, 182] days | 3 [2, 4] nights | 2 | 0.25 | Online TA 53.59% | TA/TO 93.59% |
| 2 | 5,390 (6.79%) | 239 [68, 312] days | 2 [2, 3] nights | 2 | 0.00 | Groups 53.78% | TA/TO 88.76% |

| Cluster | Meal | DepositType, post-fit only | Guest history | Reserved room | Zero-person / zero-night |
| ---: | --- | --- | --- | --- | ---: |
| 0 | BB 75.05%, SC 20.48% | No Deposit 96.53% | Repeated guest 6.37%; prior cancellations median 0 | A 83.81%, D 11.19% | 123 / 326 |
| 1 | BB 78.81%, SC 11.99% | No Deposit 83.51%; Non Refund 16.46% | Repeated guest 0.07%; prior cancellations median 0 | A 75.46%, D 17.40% | 39 / 0 |
| 2 | BB 88.50%, HB 10.89% | Non Refund 60.41% | Repeated guest 13.77%; prior cancellations median 1 [1, 1] | A 95.77% | 5 / 5 |

Interpretation:

- Cluster 0 is a short-lead, short-stay group. It contains nearly all zero-night records, has more direct-channel activity than the other clusters, and is mostly `No Deposit` when described after fitting.
- Cluster 1 is the dominant advance-planning group, with longer three-night stays, some weekend share, and a very high TA/TO channel share.
- Cluster 2 is a small long-lead history/group segment. It has the largest `Groups`, repeated-guest, prior-cancellation, and `Contract` shares. Its large `Non Refund` share is descriptive only and did not create the labels.

These are tendencies, not causal explanations or customer identities. Exact quartiles and category shares/lifts are in `cluster_profiles.csv` and `cluster_category_profiles.csv`.

## Leakage, timing, and alignment

`DepositType` is absent from fitting, metric, and PCA inputs. It is read from the preserved source frame only after labels exist. Its exclusion addresses a verified timing problem: the publication derives it from payments observed before arrival or cancellation, so its window is post-booking and outcome-conditioned. `IsCanceled`, final reservation status fields, transaction-derived `ADR`, assigned room, booking changes through check-in/cancellation, and complete waiting-list duration are also not used here.

The published dataset is a day-prior-to-arrival/change-log snapshot. Selected party composition, nights, meal, reserved room, arrival month, market segment, distribution channel, and customer type may therefore reflect lifecycle updates. They are interpreted as retrospective record characteristics, not guaranteed first-booking values. Market/channel and repeated-guest/history fields also overlap conceptually.

## Limitations and next phase

- The sampled silhouette depends on the fixed sample policy.
- Cluster IDs have no intrinsic order.
- PCA is only a two-dimensional view of 45 fitted dimensions.
- This is one fixed untuned algorithm; stability, alternative `k`, and other algorithms belong to Milestone 2 and are intentionally not implemented here.
- The clustering interface with Member 2 is still pending team approval.
