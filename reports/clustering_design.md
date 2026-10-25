# Clustering Feature Preparation and Preprocessing Design

## Scope

This document records Member 5's Phase 2 implementation for City Hotel booking clustering. It covers fixed feature construction, fitted preprocessing, targeted data-quality checks, and leakage controls. It does not train K-Means or any other clustering algorithm and does not report clustering performance.

The official project specification requires genuinely unsupervised clustering, justified source variables, appropriate categorical handling and scaling, leakage auditing, internal evaluation, visualization, and eventually at least three baseline-and-optimized clustering algorithms. Phase 2 implements only the reusable preparation required before those experiments.

## Objective and analysis point

The objective is to discover interpretable groups of booking records from planning horizon, arrival season, stay pattern, party composition, product choices, booking channel, and prior guest history.

The unit is a booking, not a unique customer. H2 has no stable customer identifier. The analysis is described as booking-characteristics segmentation rather than a guaranteed initial-booking snapshot because published H2 fields may reflect updates made during the booking lifecycle.

Final outcomes and later operational information are not used to fit clusters. Excluded outcome fields may be used only after clustering to describe results, with that distinction stated explicitly.

## Source data contract

The source boundary retains the exact 31 H2 column labels. `validate_source_schema` rejects missing or unexpected source columns. Load the CSV with whitespace trimming and explicit textual missing markers:

```python
pd.read_csv(
    "data/H2.csv",
    skipinitialspace=True,
    na_values=["NULL", "NA"],
    low_memory=False,
)
```

The shared loader is owned by Member 2 and is not modified in this phase.

## Selected source features

| Concept | Exact source columns | Representation |
| --- | --- | --- |
| Planning horizon | `LeadTime` | Nonnegative count; median imputation, `log1p`, standardization |
| Arrival season | `ArrivalDateMonth` | Cyclic sine/cosine coordinates plus missing indicator |
| Stay pattern | `StaysInWeekendNights`, `StaysInWeekNights` | `TotalNights` and `WeekendShare` |
| Party composition | `Adults`, `Children`, `Babies` | `PartySize`, `HasChildren`, and `HasBabies` |
| Product | `Meal`, `ReservedRoomType`, `DepositType`, `CustomerType` | Explicit-missing-value imputation and full one-hot encoding |
| Route to market | `MarketSegment`, `DistributionChannel` | Explicit-missing-value imputation and full one-hot encoding |
| Prior history | `IsRepeatedGuest`, `PreviousCancellations`, `PreviousBookingsNotCanceled` | Binary category plus log-scaled nonnegative counts |

The derived representation avoids keeping both all raw parts and their totals, which would give stay length or party size excessive Euclidean-distance weight.

Zero-night and zero-person records are retained. They produce zero `TotalNights` or `PartySize`, `WeekendShare` is defined as zero for a zero-night record, and separate flags are preserved outside the model matrix for EDA and interpretation.

## Exclusions and leakage policy

| Excluded columns | Reason |
| --- | --- |
| `IsCanceled`, `ReservationStatus`, `ReservationStatusDate` | Direct current-booking outcomes or outcome timing |
| `AssignedRoomType` | Later room assignment may reflect upgrades, overbooking, or operations |
| `BookingChanges` | Accumulates after initial booking entry |
| `DaysInWaitingList` | Final waiting duration is not known at initial booking |
| `ADR` | Regression target and uncertain booking-time availability; profile only |
| `ArrivalDateYear` | Risks separating collection periods rather than behavior |
| `ArrivalDateWeekNumber`, `ArrivalDateDayOfMonth` | Redundant or weak calendar dimensions for this objective |
| `Country`, `Agent`, `Company` | High-cardinality codes; `Agent` and `Company` also have substantial missingness |
| `RequiredCarParkingSpaces`, `TotalOfSpecialRequests` | Timing must be agreed with Members 1 and 2 before possible inclusion |

If clustering labels later become inputs to a supervised model, feature construction, fitted preprocessing, and clustering must be fitted inside each supervised training fold. A clusterer fitted on the complete dataset must not be reused as a leakage-free supervised feature generator.

## Targeted H2 observations

The local file parsed successfully with 79,330 rows and the exact 31-column schema.

Semantic missing values after trimming padded fields and interpreting `NULL`/`NA` are:

| Column | Missing |
| --- | ---: |
| `Children` | 4 |
| `Country` | 24 |
| `Agent` | 8,131 |
| `Company` | 75,641 |

All other source columns have zero missing values under this policy. There are 25,902 exact duplicate rows. They are not removed because the anonymized data has no booking identifier; Member 1 must decide whether identical rows are true duplicates or distinct but indistinguishable bookings.

Targeted validity findings:

- 167 rows have zero total people.
- 331 rows have zero total nights.
- 67 rows have both zero people and zero nights.
- No negative values were found in the selected count features.
- Selected category cardinalities are modest except for deliberately excluded identifiers/geography: `Country` has 166, `Agent` 223, and `Company` 207 non-missing levels.

Targeted distribution findings:

| Feature | Skewness | Maximum | Decision |
| --- | ---: | ---: | --- |
| `LeadTime` | 1.390 | 629 | Apply `log1p` |
| `TotalNights` | 3.924 | 57 | Apply `log1p` |
| `PartySize` | 0.722 | 12 | Scale without log |
| `PreviousCancellations` | 16.585 | 21 | Apply `log1p` |
| `PreviousBookingsNotCanceled` | 23.117 | 72 | Apply `log1p` |
| `WeekendShare` | 0.991 | 1 | Bounded ratio; scale without log |

No clipping is performed. Simple IQR rules label many legitimate zero-inflated count observations as outliers, so clipping requires Member 1's full EDA and a business justification.

## Fixed engineering versus fitted preprocessing

`src/clustering/features.py` performs fixed deterministic work:

- schema and domain validation;
- the source-column allowlist;
- leakage exclusions;
- `TotalNights`, `WeekendShare`, `PartySize`, `HasChildren`, and `HasBabies`;
- preservation of an untouched source copy and separate edge-case flags.

`src/clustering/pipeline.py` contains transformations that learn from fit data:

- median numerical imputation;
- categorical missing-value imputation with `__MISSING__`;
- one-hot category vocabulary;
- numerical means and standard deviations.

The month encoder and `log1p` are fixed transformations placed inside the sklearn pipeline for reproducible ordering.

## Preprocessing decisions

- Missing numerical values use the fit-data median. The four missing `Children` values therefore lead to missing derived `PartySize`, which is imputed without consulting any outcome.
- Missing ordinary categories use an explicit `__MISSING__` level.
- Missing month values map to zero sine/cosine coordinates and a separate missing indicator.
- `OneHotEncoder(handle_unknown="ignore")` allows future unseen categories without changing the matrix width.
- Full one-hot columns are not standardized. Standardizing rare dummy columns would give rare categories excessive influence.
- Numerical variables are standardized after justified log transformation. Scaling is needed because K-Means later uses Euclidean distance.
- Sparse output is retained. It avoids unnecessary dense one-hot storage and is accepted by scikit-learn K-Means.
- No random operation occurs in preprocessing, so identical fit data produces deterministic output.

## Risks and dependencies

Member 1 must provide the full EDA decision on duplicates, invalid-looking zero rows, outliers, unusual categories, and temporal drift. These records are preserved until that review.

Member 2 must coordinate the shared CSV loader, missing-marker convention, split/validation protocol, and reusable shared transformations. This implementation does not change `src/data/` or `src/preprocessing/`.

`MarketSegment` and `DistributionChannel` may encode overlapping channel information. `IsRepeatedGuest` may overlap with prior-booking counts. Member 1 should quantify these relationships before Phase 3; retaining both initially is documented and testable.

The H2 extraction process does not guarantee that every value is the original value at booking creation. This limits claims about true booking-time segmentation.

## Testing

From the repository root:

```powershell
python -m pytest -q
```

The focused tests cover schema and allowlist behavior, leakage exclusion, derived features, missing values, negative inputs, unknown categories, determinism, sparse output dimensions, input immutability, and zero-person/zero-night records. Synthetic fixtures are used only for unit testing and are not experimental results.
