# Clustering Feature Preparation and Integration Contract

## Scope and analysis point

This document defines Member 5's clustering-owned feature policy. The unit is a booking record, not a unique customer, because H2 has no stable customer identifier. This is a retrospective segmentation of the published H2 record, not a deployable initial-booking customer segmentation. The [original H2 publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) defines the observation point as the day before arrival and explains that change-log values were used where available. Some values can therefore reflect booking-lifecycle updates, and some engineered source fields use arrival or cancellation as their boundary.

The official requirements call for genuinely unsupervised clustering, justified source variables, suitable encoding and scaling, a leakage audit, internal metrics, visualization, and interpretation. Outcomes, final statuses, and uncertain post-booking fields do not enter fitting.

## Verified source contract

The official file has 79,330 rows, 31 source columns, and SHA-256 `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`. The loader requires the exact ordered schema and row count. The official output directory cannot be overwritten unless this checksum matches.

An alternate 79,330-row file with the same schema requires both `--allow-alternate-dataset` and a separate `--output-dir`. This prevents an exploratory run from replacing the official baseline.

## Revised fitting features

The exact source allowlist is:

1. `LeadTime`
2. `ArrivalDateMonth`
3. `StaysInWeekendNights`
4. `StaysInWeekNights`
5. `Adults`
6. `Children`
7. `Babies`
8. `Meal`
9. `MarketSegment`
10. `DistributionChannel`
11. `IsRepeatedGuest`
12. `PreviousCancellations`
13. `PreviousBookingsNotCanceled`
14. `ReservedRoomType`
15. `CustomerType`

Deterministic engineering produces these exact 15 fitting columns:

`LeadTime`, `TotalNights`, `PreviousCancellations`, `PreviousBookingsNotCanceled`, `WeekendShare`, `PartySize`, `ArrivalDateMonth`, `Meal`, `MarketSegment`, `DistributionChannel`, `IsRepeatedGuest`, `ReservedRoomType`, `CustomerType`, `HasChildren`, and `HasBabies`.

`TotalNights`, `WeekendShare`, and `PartySize` replace their raw component counts so those concepts are not double weighted. `HasChildren` and `HasBabies` preserve family-composition signals. Zero-night and zero-person rows remain present and are separately flagged for description.

## DepositType exclusion

`DepositType` is excluded from the source allowlist, engineered fitting frame, one-hot vocabulary, transformed matrix, K-Means fit, internal metrics, and PCA fit. Changing only `DepositType` is covered by automated feature-frame, transformed-matrix, and fitted-label invariance tests and cannot change the prepared feature matrix or clustering result.

The timing risk is verified rather than merely suspected. The original H2 publication states that `DepositType` is calculated from payments identified before the booking's arrival or cancellation date. That gives canceled and arriving bookings outcome-conditioned observation windows and is incompatible with an initial-booking interpretation. The local H2 audit also found that 99.81% of `Non Refund` bookings were canceled versus 30.48% for other deposit categories. In the original baseline, cluster 2 was 60.34% `Non Refund` and 92.86% canceled. Cancellation was used only to identify this safety risk; it was not supplied to K-Means, used to optimize `k`, or used to select a better-scoring result.

The untouched source copy still contains `DepositType`. It is joined to fixed labels by row position only after fitting for profiles and the deposit-type chart.

## Leakage and timing classification

### A. Direct outcome fields

| Excluded columns | Evidence and reason |
| --- | --- |
| `IsCanceled`, `ReservationStatus`, `ReservationStatusDate` | Direct outcome or final-status information |

### B. Verified post-booking or outcome-conditioned fields

| Excluded columns | Evidence and reason |
| --- | --- |
| `DepositType` | Calculated from payments recorded before arrival or cancellation; the observation window depends on the booking outcome |
| `BookingChanges` | Counts amendments from PMS entry until check-in or cancellation |
| `ADR` | Derived from lodging transactions divided by staying nights, rather than being a pristine initial-booking value |
| `AssignedRoomType` | Operational assignment that can differ because of overbooking or later customer requests |
| `DaysInWaitingList` | Complete duration from PMS entry until customer confirmation; unavailable at first entry |
| `Country` | The publication warns that the correct nationality is often not known until check-in; it is also high-cardinality |

### C. Selected fields that may reflect lifecycle updates

H2 uses booking/change-log values relative to the day before arrival. The publication specifically cautions that party composition, stay duration, meal, and room preferences can change after booking. Therefore `PartySize`, `HasChildren`, `HasBabies`, `TotalNights`, `WeekendShare`, `Meal`, and `ReservedRoomType` are valid only as retrospective published-record characteristics. Arrival month, market segment, distribution channel, and customer type also come from booking/change-log tables and may reflect the latest recorded value rather than first entry. `PreviousCancellations`, `PreviousBookingsNotCanceled`, and `IsRepeatedGuest` are defined relative to history before the current booking, but they overlap conceptually and can overweight guest-history behavior.

### D. Conservative or design-driven exclusions

| Excluded columns | Reason |
| --- | --- |
| `ArrivalDateYear` | Can create collection-period rather than behavioral clusters |
| `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth` | Redundant or weak calendar dimensions for this objective |
| `Agent`, `Company` | High-cardinality anonymous identifiers with substantial missingness |
| `RequiredCarParkingSpaces`, `TotalOfSpecialRequests` | Availability timing still needs team confirmation |

These distinctions do not prove that every lifecycle-updated field is outcome leakage. They limit the interpretation: the fitted groups describe the published H2 booking snapshots, while direct outcomes and established outcome-conditioned proxies remain excluded.

## Preprocessing

- `LeadTime`, `TotalNights`, and prior-booking counts: median imputation, `log1p`, then standardization.
- `WeekendShare` and `PartySize`: median imputation and standardization.
- Arrival month: cyclic sine/cosine values plus a missingness indicator.
- Categorical fields: explicit `__MISSING__` imputation and full one-hot encoding with unknown levels ignored.
- Sparse CSR output is retained for K-Means.
- No clipping or row removal is performed.

The revised official matrix is 79,330 × 45. All stored values are checked for NaN and infinity before fitting.

## Shared preprocessing integration contract

Inspection of the fetched `origin/main` at `17118fbd5f7cb6d361415d2d612f4312ccf39344` shows that Member 2's current clustering export is not API-compatible with this baseline. Its shared cleaning removes one `ADR > 1000` record and 167 zero-guest records from the official file, leaving 79,162 rows. Its clustering contract also uses `ADR`, `DaysInWaitingList`, `TotalOfSpecialRequests`, `HasAgent`, and `HasCompany`, and produces a different transformed representation. Its names differ for similar concepts (`TotalGuests`/`PartySize`, `WeekendNightRatio`/`WeekendShare`, and `HasKids` versus separate child/baby indicators).

Proposed ownership boundary:

| Owner | Responsibility |
| --- | --- |
| Member 2 | Shared H2 loading, exact source-schema validation, missing-marker normalization, and supervised preprocessing/splits |
| Member 5 | Clustering feature selection, clustering engineering and preprocessing, clustering algorithms, evaluation, and visualization |

The canonical interface still requires explicit team approval. The recommended interface is a shared loader that returns all 79,330 H2 rows in the exact documented 31-column order, enforces schema and dataset identity, and performs agreed safe string/missing-marker normalization only. It must not delete rows or add task-specific features. That normalized source frame can then be passed to Member 5's existing `build_booking_characteristics`. Member 5 should not consume Member 2's current `clustering_all.csv` or clustering preprocessor because that would change the row policy and feature contract. The team must explicitly decide whether category rewrites such as `Undefined` to `SC` or missing belong to the shared normalization layer.

This interface agreement is **pending team approval**. No Member 2 file was modified, and Member 2's cleaning or clustering configuration was not silently adopted.

### Separate main-branch archive blocker

The same fetched `origin/main` contains tracked `Classification.zip` (blob `9c897eebb7ec832eef6b1419c868686c7ec5c8aa`). In-memory archive inspection found `data/splits/train.csv`, `val.csv`, and `test.csv`, plus three fitted `artifacts/classification/*.joblib` models. This bypasses the repository's ignore rules and the submission rule excluding fitted models and pipelines. It is a team-level integration blocker, not a clustering change. The repository coordinator should remove the archive from tracked `main`, retain only source/report/aggregate outputs, and add an archive-content check or an approved submission-archive policy.

## Reproduction

```powershell
python --version  # exact recorded environment: Python 3.14.7
python -m pip install -r requirements.txt -c constraints/clustering-milestone1.txt
python -m src.clustering.experiment --dataset data/H2.csv --output-dir outputs/clustering
python -m src.clustering.evaluation --dataset data/H2.csv --output-dir outputs/clustering --sample-size 3000 --seed 42
python -m src.clustering.visualization --dataset data/H2.csv --output-dir outputs/clustering --sample-size 3000 --seed 42
python -m pytest -q
```

The fixed seed and constrained numerical environment reproduce the counts, assignments, inertia, internal metrics, profiles, and PCA variance. Wall-clock preprocessing, fitting, metric, and total durations remain machine- and load-dependent and are not reproducibility targets.
