"""Row-level cleaning for the City Hotel (H2) dataset.

Everything here is deterministic and learns nothing from the data, so it is safe
to apply before the train/validation/test split and again at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Exact source labels from the project specification (Section 4).
EXPECTED_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]

NUMERIC_SOURCE_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "IsRepeatedGuest", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "BookingChanges", "DaysInWaitingList", "ADR",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
]

# Text values that actually mean "missing" (EDA: Agent 10.25 %, Company 95.35 % are "NULL").
MISSING_MARKERS = {"NULL", "null", "None", "none", "N/A", "n/a", "NA", "nan", "NaN", ""}


def load_raw(path: str) -> pd.DataFrame:
    """Load H2.csv and verify it is the City Hotel schema."""
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame, expected_rows: int | None = None) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"H2.csv is missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        raise ValueError(f"Unexpected columns (did you load H1 or a merged file?): {extra}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")


def _text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NUMERIC_SOURCE_COLUMNS]


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Fix formatting problems. Used for training AND inference (no rows dropped).

    * strip padded strings (e.g. Meal "BB       ", room types "A               ")
    * turn hidden-missing markers such as "NULL" into real NaN
    * Agent: missing means "booked without an agent" -> explicit category "None"
    * Meal: "Undefined" and "SC" both mean no meal package (dataset paper) -> "SC"
    * MarketSegment / DistributionChannel "Undefined" -> NaN (imputed later)
    * cast numeric columns to numbers
    """
    out = df.copy()
    for col in _text_columns(out):
        if col not in out.columns:
            continue
        s = out[col].astype("object")
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.map(lambda v: np.nan if (isinstance(v, str) and v in MISSING_MARKERS) else v)
        out[col] = s

    for col in NUMERIC_SOURCE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "Agent" in out.columns:
        # Agent IDs may be read as numbers (9 / 9.0) -> normalise to text IDs.
        out["Agent"] = out["Agent"].map(_id_to_text).fillna("None")
    if "Company" in out.columns:
        out["Company"] = out["Company"].map(_id_to_text)
    if "Meal" in out.columns:
        out["Meal"] = out["Meal"].replace({"Undefined": "SC"})
    for col in ("MarketSegment", "DistributionChannel"):
        if col in out.columns:
            out[col] = out[col].replace({"Undefined": np.nan})
    return out


def _id_to_text(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else str(v)
    return str(v)


def clean_training_data(df: pd.DataFrame, adr_max: float = 1000,
                        drop_zero_guests: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize and drop invalid rows. Returns (clean_df, cleaning_log)."""
    out = standardize(df)
    log = [{"step": "raw rows", "rows_removed": 0, "rows_remaining": len(out)}]

    def _drop(mask: pd.Series, name: str):
        nonlocal out
        n = int(mask.sum())
        out = out.loc[~mask].copy()
        log.append({"step": name, "rows_removed": n, "rows_remaining": len(out)})

    _drop(out["ADR"] < 0, "drop negative ADR")
    _drop(out["ADR"] > adr_max, f"drop ADR > {adr_max} (data-entry error)")
    if drop_zero_guests:
        guests = out["Adults"].fillna(0) + out["Children"].fillna(0) + out["Babies"].fillna(0)
        _drop(guests == 0, "drop bookings with zero guests")

    out = out.reset_index(drop=True)
    return out, pd.DataFrame(log)
