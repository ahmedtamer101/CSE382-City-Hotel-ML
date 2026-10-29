# Preprocessing Contract — hand-off to Members 3, 4 and 5

## Regenerate everything

```bash
pip install -r requirements.txt
# put the City Hotel file at data/raw/H2.csv
python -m src.preprocessing.run --config configs/preprocessing.yaml
python -m pytest -q tests
```

## What you get

| Path | Content |
|---|---|
| `data/interim/h2_clean_features_split.csv` | Full cleaned table + engineered features + `GroupId` + `Split` |
| `data/processed/regression/{train,val,test}.csv` | Raw (unscaled) model inputs + `ADR` + `GroupId` (0-night stays removed) |
| `data/processed/classification/{train,val,test}.csv` | Raw model inputs + `IsCanceled` + `GroupId` |
| `data/processed/clustering/clustering_all.csv` | All cleaned rows, clustering columns only (no outcome) |
| `*_transformed.csv.gz` | Same rows already imputed/scaled/encoded (quick experiments only) |
| `artifacts/preprocessing/*_preprocessor.joblib` | Preprocessors fitted on TRAIN only (for the app) |
| `outputs/preprocessing/feature_contract.json` | Input columns, column groups and every exclusion with its reason |
| `outputs/preprocessing/split_summary.csv`, `cleaning_log.csv`, `leakage_checks_train.csv` | Tables for the report |

`data/`, `artifacts/` are **not committed and not zipped** — anyone can regenerate them with the command above.

## Rules for every member

1. Use **only** these splits. Do not call `train_test_split` yourself.
2. Fit the preprocessor on training data only — the easiest way is to put it inside your model pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from src.preprocessing import build_preprocessor, load_task_split

X_tr, y_tr, g_tr = load_task_split("classification", "train", with_groups=True)
X_val, y_val = load_task_split("classification", "val")

clf = Pipeline([("prep", build_preprocessor("classification")),
                ("model", LogisticRegression(max_iter=1000))])
clf.fit(X_tr, y_tr)                 # baseline -> evaluate on X_val
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cross_validate(clf, X_tr, y_tr, groups=g_tr, cv=cv)   # tuning: always pass groups
```

3. Regression (Member 3): `load_task_split("regression", ...)`; for CV use `GroupKFold(5)` with `groups`. A log-transform of the ADR target (`TransformedTargetRegressor`) is allowed and is your decision.
4. Clustering (Member 5): `X, _ = load_task_split("clustering", None)` then `build_preprocessor("clustering").fit_transform(X)`; for hierarchical clustering sample rows with `random_state=42` and document it.
5. Do **not** score on `test` during Milestone 1. Use `val` or CV.
6. App / inference: `prepare_inference_frame(raw_booking_dict, task)` → saved pipeline `.predict(...)`. It accepts the original H2 column names and fills missing optional fields.

## Input columns per task

See `feature_contract.json` for the exact lists. Summary:

* **Regression (target ADR)**: arrival week/day/year, cyclical month, weekday, nights (weekend/week/total, weekend ratio), guests (adults/children/babies/total, HasKids), lead time, waiting-list days, repeated guest + previous bookings, special requests, Meal, Country, MarketSegment, DistributionChannel, ReservedRoomType, DepositType, CustomerType, Agent, HasAgent, HasCompany.
* **Classification (target IsCanceled)**: same as regression **minus** Country and ArrivalDateYear, **plus** ADR.
* **Clustering**: LeadTime, TotalNights, WeekendNightRatio, TotalGuests, HasKids, ADR, TotalOfSpecialRequests, DaysInWaitingList, IsRepeatedGuest, HasAgent, HasCompany, cyclical month, MarketSegment, CustomerType, Meal, DistributionChannel.

## Preprocessing per column group

| Group | Steps (all fitted on train) |
|---|---|
| numeric | median imputation → StandardScaler |
| skewed (LeadTime, nights, waiting days, previous bookings) | median imputation → log1p → StandardScaler |
| binary flags | most-frequent imputation, kept as 0/1 |
| categorical | missing → "Unknown" → OneHotEncoder, categories < 0.5 % of train grouped as "infrequent", unseen categories at inference → infrequent |
