# Revised Untuned K-Means Baseline

## Scope

This is the corrected Milestone 1 baseline after removing `DepositType` from fitting. The original baseline remains in Git history at commit `6ba6e29`; its results must not be attributed to this model. No cluster-count search, hyperparameter tuning, extra algorithm, or model serialization is performed here.

## Reproduction and identity safety

```powershell
python --version  # exact recorded environment: Python 3.14.7
python -m pip install -r requirements.txt -c constraints/clustering-milestone1.txt
python -m src.clustering.experiment --dataset data/H2.csv --output-dir outputs/clustering
```

The command validates the exact 31-column schema, 79,330 rows, and official SHA-256 `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4` before official result files are overwritten. The official directory is anchored to the repository rather than process CWD. An alternate dataset requires explicit opt-in and an output path outside that directory, including its child paths and symlink aliases.

## Fixed configuration

| Parameter | Value |
| --- | --- |
| `n_clusters` | 3 |
| `init` | `k-means++` |
| `n_init` | 10 |
| `random_state` | 42 |
| `max_iter` | 300 |

`k=3` is an untuned reference point, not evidence that three clusters are optimal.

## Revised representation

The 15 engineered fitting fields are `LeadTime`, `TotalNights`, `PreviousCancellations`, `PreviousBookingsNotCanceled`, `WeekendShare`, `PartySize`, `ArrivalDateMonth`, `Meal`, `MarketSegment`, `DistributionChannel`, `IsRepeatedGuest`, `ReservedRoomType`, `CustomerType`, `HasChildren`, and `HasBabies`.

After imputation, scaling, cyclic encoding, and one-hot encoding, the exact matrix is 79,330 × 45 in CSR format. It contains 1,265,544 stored nonzero values, has density `0.354509`, occupies 15,503,852 bytes in CSR form, and would occupy 28,558,800 bytes as a dense float64 array. It contains no NaN or infinite values.

## Actual revised result

| Result | Value |
| --- | ---: |
| Rows | 79,330 |
| Source variables | 31 |
| Engineered fitting fields | 15 |
| Transformed features | 45 |
| K-Means iterations | 31 |
| Inertia | 575,905.1184983611 |
| Convergence warnings | None |

Runtime fields are recorded in `outputs/clustering/baseline_metrics.json`, but they are machine- and load-dependent and are not treated as deterministic. In the constrained recorded environment, counts, assignments, inertia, metrics, profiles, and PCA variance are the numerical reproducibility targets.

| Cluster ID | Bookings | Percentage |
| ---: | ---: | ---: |
| 0 | 19,651 | 24.771209% |
| 1 | 54,289 | 68.434388% |
| 2 | 5,390 | 6.794403% |
| **Total** | **79,330** | **100.000000%** |

Cluster IDs are arbitrary labels. Removing `DepositType` changes the geometry substantially, so the old labels, counts, and inertia were not reused.

## Original versus revised baseline

| Item | Original at `6ba6e29` | Revised |
| --- | ---: | ---: |
| `DepositType` in fitting | Yes | No; post-fit description only |
| Engineered fields | 16 | 15 |
| Transformed columns | 48 | 45 |
| Cluster counts | 40,174 / 33,752 / 5,404 | 19,651 / 54,289 / 5,390 |
| Iterations | 15 | 31 |
| Inertia | 597,856.3237440714 | 575,905.1184983611 |

Inertia is not directly comparable across different feature spaces; the lower revised value does not prove improvement. It only describes compactness in the new 45-dimensional representation.

## Leakage and row policy

The fitting input excludes `IsCanceled`, `ReservationStatus`, and `ReservationStatusDate` as direct outcomes. It also excludes verified post-booking or outcome-conditioned fields: the publication derives `DepositType` from payments before arrival or cancellation, counts `BookingChanges` until check-in or cancellation, defines `DaysInWaitingList` through confirmation, and constructs `ADR` from lodging transactions and stay nights. `AssignedRoomType` is a later operational assignment. The strong cancellation association of `DepositType` reinforces—but is not the sole basis for—its exclusion.

All 79,330 schema-valid rows are retained. Exact duplicates, zero-person bookings, zero-night bookings, and outliers are not silently removed.

## Limitations

- H2 is a retrospective day-prior-to-arrival/change-log extract, not a pristine initial snapshot. Selected party, night, meal, reserved-room, market, channel, customer-type, and arrival-month fields can reflect lifecycle updates, so the clusters are interpreted as published-record segments.
- `MarketSegment` and `DistributionChannel` overlap; prior-history fields also overlap.
- Euclidean distance depends on scaling and category weighting.
- A 6.79% cluster requires later stability analysis.
- No outcome may be used to tune `k` or select clustering parameters.
- The Member 2/Member 5 clustering interface remains pending team approval; see `clustering_design.md`.
