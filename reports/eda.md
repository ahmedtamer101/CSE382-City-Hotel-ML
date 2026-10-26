# City Hotel H2 — Data Understanding and Exploratory Data Analysis

**Course:** CSE382 | **Milestone:** 1 | **Contributor:** mohamedbasyouni16  
**Dataset:** Hotel Booking Demand, City Hotel (H2), arrivals July 2015–August 2017

## 1. Objective and scope

This analysis profiles the original H2 booking dataset before preprocessing and predictive modeling. It identifies data-quality issues, describes booking patterns, and provides actionable handoffs to the preprocessing, classification, regression, and clustering teams. All figures are descriptive; correlations do not establish causation. No rows or columns were removed during EDA.

The source contains **79,330 bookings and 31 variables**. The source labels are preserved. Dataset mirror: https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv. Original dataset publication: António, Almeida, and Nunes (2019), *Hotel booking demand datasets*, Data in Brief, https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/.

## 2. Data quality audit

| Finding | Observed result | Recommended action |
|---|---|---|
| Actual missing `Country` | 24 rows (~0.03%) | Investigate and document unknown-country handling. |
| Actual missing `Children` | 4 rows (~0.005%) | Consider imputation only within an approved preprocessing design. |
| Textual `NULL` in `Agent` | 8,131 rows (10.25%) | Determine whether `NULL` means a booking without an agent; consider a distinct missing/direct-booking category. |
| Textual `NULL` in `Company` | 75,641 rows (95.35%) | Establish whether `NULL` means no company affiliation; do not blindly drop or mean-impute. |
| Fully identical records after the first occurrence | 25,902 rows (32.65%) | Investigate before deduplicating: no unique booking ID is provided, so identical rows may be separate bookings. |
| Bookings with zero recorded guests | 167 (0.211%) | Inspect cancellation status and source conventions before exclusion. |
| Bookings with zero recorded nights | 331 (0.417%) | Inspect cancellation status and booking lifecycle. |
| Negative `LeadTime`, `ADR`, or guest counts | None found | No action indicated by these checks. |
| `ADR` above 1,000 | 1 record; maximum 5,400 | Inspect extreme record; statistical rarity alone does not prove an error. |

**Missing-value caveat:** Default `pandas.read_csv()` does not necessarily interpret all textual sentinels identically. The above counts were obtained by testing object values for markers including `NULL`, while `isna()` reported `Country` and `Children`. The cleaning contract must explicitly define recognized sentinels.

## 3. Descriptive statistics and distributions

- The approximate cancellation rate is **42%**; neither target class is rare enough to justify accuracy-only evaluation.
- `LeadTime` has mean **109.74 days**, median **74 days**, and maximum **629 days**. Its long right tail indicates relatively fewer bookings made far in advance.
- `ADR` has mean **105.30**, median **99.9**, and maximum **5,400**. Most visible observations in the 0–500 histogram cluster around approximately 60–150. A spike at zero and the extreme maximum need review.
- The largest market segment is **Online TA** (**38,748 bookings**); the most common customer type is **Transient** (**59,404 bookings**).
- **No Deposit** dominates the deposit categories (**66,442 bookings**).
- **August** has the largest raw arrival count (**8,983 bookings**). Monthly counts are not exposure-adjusted: the dataset includes partial first and final years, so seasonality cannot be inferred from raw counts alone.

## 4. IQR outlier screening

The conventional rule flags values outside `[Q1 − 1.5 × IQR, Q3 + 1.5 × IQR]`.

| Variable | Lower bound | Upper bound | Flagged rows | Share |
|---|---:|---:|---:|---:|
| `LeadTime` | −187.0 | 373.0 | 2,576 | 3.25% |
| `ADR` | 9.0 | 196.2 | 3,387 | 4.27% |
| `StaysInWeekNights` | −2.0 | 6.0 | 719 | 0.91% |
| `StaysInWeekendNights` | −3.0 | 5.0 | 82 | 0.10% |
| `DaysInWaitingList` | 0.0 | 0.0 | 3,443 | 4.34% |

For `DaysInWaitingList`, both quartiles are zero. The IQR method therefore flags **every positive value**, which is a property of this zero-inflated variable rather than evidence of bad data. Preserve original data and assess any transformation or filtering with the model owners.

## 5. Cancellation by booking attributes

| Attribute / category | Bookings | Cancellation rate |
|---|---:|---:|
| `MarketSegment`: Groups | 13,975 | 68.86% |
| `MarketSegment`: Offline TA/TO | 16,747 | 42.83% |
| `MarketSegment`: Online TA | 38,748 | 37.40% |
| `MarketSegment`: Direct | 6,093 | 17.33% |
| `MarketSegment`: Undefined | 2 | 100% (unstable: n=2) |
| `DepositType`: Non Refund | 12,868 | 99.81% |
| `DepositType`: Refundable | 20 | 70% (unstable: n=20) |
| `DepositType`: No Deposit | 66,442 | 30.47% |
| `CustomerType`: Contract | 2,300 | 48.04% |
| `CustomerType`: Transient | 59,404 | 45.62% |
| `CustomerType`: Transient-Party | 17,333 | 28.10% |
| `CustomerType`: Group | 293 | 9.90% |

The highest observed monthly cancellation rate is **April (46.32%)** and the lowest is **March (36.95%)**. Month-to-month patterns warrant year-stratified analysis because the observation windows are unequal.

**Interpretation:** Deposit type and market segment show pronounced descriptive associations with cancellation. In particular, the near-total cancellation of `Non Refund` bookings is unusual and should be audited for source semantics, timing, confounding, and potential leakage before interpretation or model use. `MarketSegment=Groups` and `CustomerType=Group` are different fields and should not be conflated.

## 6. Numeric Pearson correlations

Selected coefficients with `IsCanceled`:

| Feature | Pearson r |
|---|---:|
| `LeadTime` | +0.31 |
| `PreviousCancellations` | +0.17 |
| `DaysInWaitingList` | +0.06 |
| `TotalOfSpecialRequests` | −0.29 |
| `BookingChanges` | −0.15 |
| `RequiredCarParkingSpaces` | −0.13 |
| `IsRepeatedGuest` | −0.07 |

Longer lead times tend to coincide with more cancellations, whereas higher special-request counts tend to coincide with fewer. These are **unadjusted associations**, not causal claims. Pearson correlation also misses nonlinear associations and does not adequately evaluate categorical predictors.

Other notable numeric correlations: `IsRepeatedGuest` vs. `PreviousBookingsNotCanceled` (~+0.45), `PreviousCancellations` vs. `PreviousBookingsNotCanceled` (~+0.39), `Children` vs. `ADR` (~+0.31), and `Adults` vs. `ADR` (~+0.26). The apparent `ArrivalDateYear`/`ArrivalDateWeekNumber` correlation (~−0.54) reflects dataset coverage and should not be treated as a causal feature relationship.

## 7. Handoff and leakage risks

1. **Agree on prediction time.** For booking-time cancellation prediction, availability of each feature must be verified at booking creation, not at departure or after cancellation.
2. **Exclude direct outcome leakage.** `ReservationStatus` and `ReservationStatusDate` encode final outcomes and should not be used as predictors of cancellation at booking time.
3. **Audit possible post-booking fields.** `BookingChanges`, `AssignedRoomType`, `TotalOfSpecialRequests`, and possibly other mutable fields may be unavailable or different at the chosen prediction point. `DepositType` also needs a timing and semantic audit before modeling.
4. **Define missing-value semantics.** Textual `NULL` in `Agent` and `Company` can carry business meaning. Document conversions and retain original source data.
5. **Avoid automatic row deletion.** Identical bookings, zero stays, zero guests, and IQR outliers require domain checks and sensitivity analysis.
6. **Preprocessing leakage prevention.** Fit learned imputers, encoders, scalers, selectors, and transformations on training data or training CV folds only. Keep a final held-out supervised test set untouched during model selection.
7. **Share schema and checks.** The preprocessing, regression, classification, and clustering owners should receive these findings before final feature contracts are selected.

## 8. Reproducibility

Run the accompanying script from the repository root after installing `pandas`, `numpy`, `matplotlib`, and `seaborn`:

```bash
python -m src.data.eda --data data/H2.csv --out outputs/eda
```

If the CSV is absent, the script can read the documented public mirror directly. It writes summary CSV files and figures under `outputs/eda/` without editing the source dataset. The exploratory notebook is `notebooks/01_Data_Understanding_EDA.ipynb`.

**Status:** EDA findings documented for Milestone 1. Model-specific feature selection and cleaning decisions remain pending joint review with the preprocessing owner.
