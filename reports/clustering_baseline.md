# Untuned K-Means Baseline

## Scope

This report records Member 5's first clustering baseline for Milestone 1. It uses the clustering-specific Phase 2 feature and preprocessing contract. It does not search for the best number of clusters, optimize hyperparameters, calculate Phase 4 clustering metrics, create visualizations, or save a fitted model.

The official project specification requires an initial untuned clustering experiment for Milestone 1 and says K-Means should use the full H2 dataset where practical. The final project will require at least three clustering algorithms, optimized configurations, internal evaluation, visualization, and interpretation; those tasks remain outside this baseline.

## Reproduction command

Run from the repository root:

```powershell
python -m src.clustering.experiment --dataset data/H2.csv --output-dir outputs/clustering
```

The command fails clearly if the dataset is missing, has a different row count, or does not have the exact ordered 31-column H2 schema.

## Workflow

1. Load `data/H2.csv` using the documented `NULL` and `NA` missing markers.
2. Validate 79,330 rows and the exact 31 source columns.
3. Construct the committed 16-column leakage-safe booking representation with `build_booking_characteristics`.
4. Fit the committed imputation, cyclic month encoding, one-hot encoding, log transformations, and numerical scaling.
5. Keep the resulting 79,330 by 48 matrix in CSR sparse form.
6. Fit one fixed K-Means configuration on every row.
7. Write only small metrics, cluster-size, and metadata files. No model or row-level label artifact is saved.

## What K-Means does

K-Means divides observations into `k` groups. It repeatedly assigns each booking to its nearest cluster center and then moves each center to the mean position of its assigned bookings. It stops when the centers stabilize or `max_iter` is reached.

The algorithm uses Euclidean distance. Scaling is necessary because an unscaled large-range variable such as lead time could dominate a small-range variable such as weekend share. One-hot categorical columns are not individually standardized because that would give rare categories excessive weight.

## Baseline configuration

| Parameter | Value |
| --- | --- |
| `n_clusters` | 3 |
| `init` | `k-means++` |
| `n_init` | 10 |
| `random_state` | 42 |
| `max_iter` | 300 |

`k=3` is an initial reference point, not a claim that three is optimal. Phase 4 must compare sensible cluster counts using Silhouette Score, Davies-Bouldin Index, cluster stability, sizes, and interpretability without using supervised outcomes to select `k`.

## Actual measured results

Results below were generated on October 30, 2026 from the final Phase 3 code.

| Result | Measured value |
| --- | ---: |
| Dataset rows used | 79,330 |
| Source variables | 31 |
| Engineered fitting features | 16 |
| Transformed numerical features | 48 |
| Preprocessing duration | 0.227950 seconds |
| K-Means fit duration | 0.501787 seconds |
| Total experiment duration | 0.857138 seconds |
| K-Means iterations | 15 |
| Inertia | 597,856.3237440714 |
| Convergence warnings | None |

Runtime values are machine- and load-dependent; they are recorded as observations, not performance guarantees.

### Cluster sizes

| Cluster ID | Bookings | Percentage |
| ---: | ---: | ---: |
| 0 | 40,174 | 50.641624% |
| 1 | 33,752 | 42.546325% |
| 2 | 5,404 | 6.812051% |
| **Total** | **79,330** | **100.000000%** |

Cluster IDs are arbitrary identifiers, not ordered rankings or business names. Profiles and meaningful labels belong to Phase 4.

## What inertia means

Inertia is the sum of squared Euclidean distances from every booking to the center of its assigned cluster. A smaller value means observations are more compact around their centers. Inertia always tends to decrease when more clusters are added, so the baseline value cannot prove that `k=3` is good and cannot be compared fairly across different datasets or feature representations.

## Reproducibility record

- Dataset SHA-256: `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`
- Python: `3.14.7`
- NumPy: `2.5.4`
- pandas: `3.1.0`
- SciPy: `1.18.1`
- scikit-learn: `1.9.1`

Two additional in-memory runs with the same data and configuration produced identical row-level labels, identical cluster-size tables, and exactly the same inertia (`597856.3237440714`; absolute difference `0.0`). Timing metadata was expected to vary.

## Sparse storage and memory observation

The transformed matrix has 1,344,874 stored nonzero values and density `0.353186`.

- CSR storage measured from data, index, and pointer arrays: 16,455,812 bytes (approximately 15.69 MiB).
- Dense-equivalent array: 30,462,720 bytes (approximately 29.05 MiB).

The full matrix was not densified. These figures describe the feature matrix only; they do not measure complete process memory or temporary allocations inside K-Means.

## Leakage and row policy

The fitting input cannot contain:

- `IsCanceled`
- `ReservationStatus`
- `ReservationStatusDate`
- `ADR`
- `AssignedRoomType`
- `BookingChanges`
- `DaysInWaitingList`

All 79,330 input rows were retained. Exact duplicate rows, zero-person bookings, zero-night bookings, and other unusual but schema-valid records were not silently removed.

## Limitations and Phase 4 work

- H2 values may reflect booking-lifecycle updates and are not guaranteed to be pristine initial-booking snapshots.
- Euclidean distance depends on preprocessing and feature weighting. One-hot category mismatches and multiple related variables can influence the result.
- `MarketSegment` and `DistributionChannel` may partly represent the same behavior.
- `IsRepeatedGuest` may overlap with prior-booking counts.
- A 6.81% cluster is not automatically invalid, but it requires profiling and stability analysis.
- Inertia alone is insufficient for model selection.
- No outcome variable may be used to select the number of clusters or tune clustering parameters.

Phase 4 must add Silhouette Score, Davies-Bouldin Index, cluster profiles in original business units, stability checks, and a justified cluster-count investigation. PCA visualization and later multi-algorithm comparison remain separate future requirements.
