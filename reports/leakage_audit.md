# Leakage Audit — Preprocessing & Feature Design (Member 2)

> All numbers come from `outputs/preprocessing/*.csv`, produced by `python -m src.preprocessing.run`
> on H2.csv (79,330 rows × 31 columns). Leakage checks use the **training split only** (47,719 rows).

## 1. Prediction points

| Task | Prediction / analysis point | What is unknown at that moment |
|---|---|---|
| ADR regression | **At booking confirmation**: the booking is entered in the PMS and has left the waiting list; the guest has not arrived. | Whether the booking is cancelled, the room finally assigned, later amendments, parking used at arrival. |
| IsCanceled classification | **At booking confirmation** (same point, so both models can run on the same input form in the app). | Same as above, plus the final reservation status. |
| Clustering | Descriptive segmentation of bookings using attributes of the booking itself. | The outcome (cancelled / stayed) must not shape the clusters. |

## 2. Feature-by-feature decisions

### Removed from every task

| Column | Type of leakage | Reason |
|---|---|---|
| `ReservationStatus` | **Direct** | Values Canceled / No-Show / Check-Out *are* the outcome. Train split: 20,052 rows are Canceled/No-Show and **100 %** of them have IsCanceled = 1; every other row is 0 %. A model would just copy it. |
| `ReservationStatusDate` | **Direct** | The date the final status was set; it only exists after a cancellation or a check-out. |
| `AssignedRoomType` | Post-booking | Chosen by the hotel at or near check-in. Mismatch with `ReservedRoomType` mostly happens to guests who actually arrive (upgrades), so it indirectly encodes "did not cancel". Train split: cancel rate is **5.93 %** when assigned ≠ reserved (4,270 rows) vs **45.57 %** otherwise. |
| `BookingChanges` | Post-booking | Counts amendments made until check-in **or cancellation**, so its value depends on the booking's future. Train split: **16.87 %** cancelled when BookingChanges > 0 (6,147 rows) vs **45.74 %** when 0. |
| `RequiredCarParkingSpaces` | Post-booking / indirect | Parking is usually requested or recorded at arrival. Train split: **0.00 %** of the 1,181 bookings with parking > 0 were cancelled vs **43.09 %** of the rest — a perfect "not cancelled" signal that would not exist at booking time. |

### Task-specific exclusions

| Column | Task | Reason |
|---|---|---|
| `IsCanceled` | Regression, Clustering | Future outcome at the ADR prediction point; outcome label for clustering. |
| `ADR` | Regression | Target. |
| `Country` | Classification only | Indirect leak: the guest's country is often only confirmed at check-in, so a default value (e.g. PRT) is more common for bookings that never arrive. Train split: PRT bookings are **65.44 %** cancelled (18,673 rows) vs **26.96 %** for all other countries. Kept for regression because its value is not determined by the ADR target. |
| `ArrivalDateYear` | Classification, Clustering | Calendar trend that will not repeat for future bookings. Kept for regression because prices rise year on year. |
| `DepositType` | Clustering | "Non Refund" bookings are **99.79 %** cancelled (train split, 7,966 rows) vs 30.45 % otherwise, so clusters would be organised around an outcome proxy. It is kept for classification because the deposit type is genuinely known at booking time; **Member 4 should report an ablation without it**, since this pattern is unusual. |
| `Company` | All | 95.35 % missing → replaced by `HasCompany`. |
| `Agent` | Clustering | 224 IDs dominate distance metrics → `HasAgent` used instead. |

### Accepted with a documented risk

* `TotalOfSpecialRequests`: requests can be added after booking. It is kept (strongest legitimate signal in EDA, r = −0.29 with cancellation), assuming the value is the snapshot at confirmation. Mention this in the report as a residual risk.
* `ADR` as a classification input: the agreed rate is known at booking, so it is allowed.
* `DaysInWaitingList`: known once the booking is confirmed, which is our prediction point.

## 3. How the workflow prevents leakage

1. **Deterministic cleaning only before the split.** String stripping, "NULL" → missing, removal of impossible rows (zero guests, ADR > 1000) learn nothing from the data.
2. **Feature engineering is row-wise** (`TotalNights`, `TotalGuests`, cyclical month, weekday, flags). No means, frequencies or target statistics are used.
3. **Grouped, stratified 60/20/20 split** (`StratifiedGroupKFold`, seed 42). 32.65 % of rows are exact duplicates; identical bookings share a `GroupId` and always fall in the same split, so the test set never contains a copy of a training row. Stratified on `IsCanceled` so the cancellation rate is similar in every split.

| Split | Rows | Share | Cancel rate | Mean ADR | Unique groups |
|---|---|---|---|---|---|
| train | 47,719 | 60.28 % | 42.02 % | 105.89 | 31,867 |
| val | 15,814 | 19.98 % | 42.73 % | 104.12 | 10,621 |
| test | 15,629 | 19.74 % | 40.11 % | 105.39 | 10,622 |

The test cancel rate is ~2 points lower because whole duplicate groups (often large, cancelled group bookings) must stay together; this is the expected cost of preventing duplicate leakage.
4. **All learned preprocessing is fitted on the training split only**: median imputation, log1p + `StandardScaler`, one-hot categories and the "infrequent" grouping threshold. Validation/test are only transformed. A unit test checks that scaler means equal the training means.
5. **Inside cross-validation**, members must put `build_preprocessor(task)` *inside* a `Pipeline` so it is re-fitted on each training fold, and pass `groups=GroupId` to `GroupKFold` / `StratifiedGroupKFold`.
6. **Test set is untouched in Milestone 1.** Tuning (M2) uses validation or CV on train(+val) only; the test split is scored once for the final table.
7. **PCA / LDA (Member 5)** must be added as a pipeline step after the preprocessor, so they are fitted on training folds only; LDA must use training labels only.
8. **Clustering** never receives `IsCanceled` or `ReservationStatus`; cluster hyper-parameters are chosen with silhouette / Davies–Bouldin, not with labels.

## 4. Automated checks (tests/test_preprocessing.py)

* No target or leaky column appears in any task's input columns.
* Train / val / test are disjoint and no duplicate group crosses splits.
* The split is reproducible with the same seed.
* Preprocessors produce no NaN and handle unseen categories (e.g. a new country) at inference.
* Scaler statistics come from the training split only.

## 5. Cleaning log (rules applied before the split)

| Step | Rows removed | Rows remaining |
|---|---|---|
| Raw H2.csv | 0 | 79,330 |
| ADR < 0 | 0 | 79,330 |
| ADR > 1000 (single 5,400 data-entry error) | 1 | 79,329 |
| Zero guests (Adults + Children + Babies = 0) | 167 | 79,162 |
| Regression only: 0-night stays (ADR undefined) | 264 | 78,898 |

## 6. Model input size after preprocessing (no missing values remain)

| Task | Rows (train / val / test) | Input columns | Features after encoding |
|---|---|---|---|
| Regression (ADR) | 47,572 / 15,768 / 15,558 | 31 | 104 |
| Classification (IsCanceled) | 47,719 / 15,814 / 15,629 | 30 | 82 |
| Clustering | 79,162 (all rows) | 17 | 32 |
