# ADR Regression Baseline

## Objective and prediction point

The task predicts `ADR` at booking confirmation, before the guest arrives and before the final booking outcome is known. The model uses the regression feature set defined by the preprocessing contract.

## Data and preprocessing

The experiment uses the preprocessing handoff's transformed City Hotel data. The training split contains 47,572 rows and the validation split contains 15,768 rows, with 104 transformed features. The preprocessor was fitted on training data only. Regression records with zero total nights were excluded because ADR is undefined for a zero-night stay.

Features unavailable at the prediction point or identified as leakage risks were excluded, including `ReservationStatus`, `ReservationStatusDate`, `IsCanceled`, `AssignedRoomType`, `BookingChanges`, and `RequiredCarParkingSpaces`.

## Baseline model

An untuned `LinearRegression` model was fitted on the training split and evaluated on the validation split.

| Metric | Validation result |
| --- | ---: |
| MAE | 15.2808 |
| RMSE | 21.7479 |
| R² | 0.6856 |

The validation R² indicates that the model explains about 68.56% of ADR variance in this split. The RMSE is higher than the MAE, suggesting some larger prediction errors affect the squared-error metric more strongly.

## Reproduction

From the repository root, run:

```bash
python src/regression/regression_baseline.py
the script expect the transformed train and validation file under data/processed/regression/ .
scope and limitation
this is an untuned baseline evaluated on validation data; the test split has not been used. linear regression may not capture nonlinear relationships or feature interactions. further model comparisons and tuning are planned for milestone 2  
