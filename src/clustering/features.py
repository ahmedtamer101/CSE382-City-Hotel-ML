"""Feature policy and fixed feature engineering for booking clustering.

This module deliberately contains no fitted transformations and no clustering
algorithm. It validates the original H2 schema, enforces the clustering
allowlist, and constructs a compact booking-characteristics representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd


EXPECTED_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "AssignedRoomType",
    "BookingChanges",
    "DepositType",
    "Agent",
    "Company",
    "DaysInWaitingList",
    "CustomerType",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ReservationStatus",
    "ReservationStatusDate",
)

# Only these original H2 columns may contribute to the fitting representation.
CLUSTERING_INPUT_ALLOWLIST: Final[tuple[str, ...]] = (
    "LeadTime",
    "ArrivalDateMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "DepositType",
    "CustomerType",
)

LEAKAGE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "IsCanceled",
        "ReservationStatus",
        "ReservationStatusDate",
        "ADR",
        "AssignedRoomType",
        "BookingChanges",
        "DaysInWaitingList",
    }
)

EXCLUDED_FEATURE_POLICY: Final = MappingProxyType(
    {
        "IsCanceled": "current-booking outcome",
        "ReservationStatus": "final reservation outcome",
        "ReservationStatusDate": "date of a final status event",
        "ADR": "regression target and uncertain booking-time availability",
        "AssignedRoomType": "later operational assignment rather than request",
        "BookingChanges": "accumulates after the booking is created",
        "DaysInWaitingList": "final waiting duration is not initially known",
        "ArrivalDateYear": "may create collection-period clusters",
        "ArrivalDateWeekNumber": "redundant with cyclic arrival season",
        "ArrivalDateDayOfMonth": "weak calendar meaning for this objective",
        "Country": "high-cardinality geography; reserved for profiling",
        "Agent": "high-cardinality anonymous identifier",
        "Company": "high-cardinality, mostly missing anonymous identifier",
        "RequiredCarParkingSpaces": "availability timing needs team confirmation",
        "TotalOfSpecialRequests": "availability timing needs team confirmation",
    }
)

SOURCE_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
)

LOG_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "LeadTime",
    "TotalNights",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
)

LINEAR_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "WeekendShare",
    "PartySize",
)

CYCLIC_FEATURES: Final[tuple[str, ...]] = ("ArrivalDateMonth",)

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "ReservedRoomType",
    "DepositType",
    "CustomerType",
    "HasChildren",
    "HasBabies",
)

MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *LOG_NUMERIC_FEATURES,
    *LINEAR_NUMERIC_FEATURES,
    *CYCLIC_FEATURES,
    *CATEGORICAL_FEATURES,
)

VALID_MONTHS: Final[frozenset[str]] = frozenset(
    {
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
)


@dataclass(frozen=True)
class ClusteringFeatureBundle:
    """Model inputs plus untouched source data and non-destructive flags."""

    model_features: pd.DataFrame
    source_data: pd.DataFrame
    edge_case_flags: pd.DataFrame


def validate_source_schema(data: pd.DataFrame) -> None:
    """Require exactly the 31 named H2 source columns."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Clustering source data must be a pandas DataFrame.")
    if data.columns.has_duplicates:
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate source column names: {duplicates}")

    actual = set(data.columns)
    expected = set(EXPECTED_SOURCE_COLUMNS)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError("H2 source schema mismatch: " + "; ".join(details))


def _numeric_source(data: pd.DataFrame) -> pd.DataFrame:
    """Return validated numeric inputs without filling missing observations."""
    numeric = data.loc[:, SOURCE_NUMERIC_FEATURES].copy()
    for column in SOURCE_NUMERIC_FEATURES:
        original = numeric[column]
        converted = pd.to_numeric(original, errors="coerce")
        introduced_missing = original.notna() & converted.isna()
        if introduced_missing.any():
            bad_values = original.loc[introduced_missing].astype(str).unique()[:5]
            raise ValueError(
                f"{column} contains non-numeric values: {bad_values.tolist()}"
            )
        numeric[column] = converted

    finite_values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(finite_values).any():
        raise ValueError("Selected numerical source columns contain infinity.")

    negative_counts = {
        column: int(numeric[column].lt(0).sum())
        for column in SOURCE_NUMERIC_FEATURES
        if numeric[column].lt(0).any()
    }
    if negative_counts:
        raise ValueError(
            "Selected count features must be nonnegative; "
            f"negative values found in {negative_counts}."
        )
    return numeric


def _clean_categorical(series: pd.Series) -> pd.Series:
    """Trim category labels while preserving missing values for imputation."""
    cleaned = series.astype("object").copy()
    present = cleaned.notna()
    cleaned.loc[present] = cleaned.loc[present].astype(str).str.strip()
    cleaned = cleaned.mask(cleaned.eq(""), np.nan)
    return cleaned


def _presence_label(series: pd.Series) -> pd.Series:
    """Convert a nonnegative count into yes/no while preserving missingness."""
    result = pd.Series(np.nan, index=series.index, dtype="object")
    present = series.notna()
    result.loc[present] = np.where(series.loc[present].gt(0), "yes", "no")
    return result


def build_booking_characteristics(data: pd.DataFrame) -> ClusteringFeatureBundle:
    """Build fixed, leakage-safe features without mutating ``data``.

    Zero-person and zero-night records are retained and flagged for later EDA.
    Negative counts, invalid months, invalid repeat-guest codes, and schema
    mismatches fail fast. Missing values remain for the fitted preprocessing
    pipeline to handle.
    """
    validate_source_schema(data)
    source_data = data.copy(deep=True)
    numeric = _numeric_source(data)

    arrival_month = _clean_categorical(data["ArrivalDateMonth"])
    invalid_months = sorted(
        set(arrival_month.dropna().unique()).difference(VALID_MONTHS)
    )
    if invalid_months:
        raise ValueError(f"Unknown ArrivalDateMonth values: {invalid_months}")

    repeat_guest = pd.to_numeric(data["IsRepeatedGuest"], errors="coerce")
    nonnumeric_repeat = data["IsRepeatedGuest"].notna() & repeat_guest.isna()
    if nonnumeric_repeat.any():
        raise ValueError("IsRepeatedGuest must contain only 0, 1, or missing.")
    invalid_repeat = sorted(set(repeat_guest.dropna().unique()).difference({0, 1}))
    if invalid_repeat:
        raise ValueError(
            f"IsRepeatedGuest must contain only 0, 1, or missing; found {invalid_repeat}."
        )

    total_nights = numeric[["StaysInWeekendNights", "StaysInWeekNights"]].sum(
        axis=1, min_count=2
    )
    weekend_share = numeric["StaysInWeekendNights"].div(total_nights)
    weekend_share = weekend_share.mask(total_nights.eq(0), 0.0)
    party_size = numeric[["Adults", "Children", "Babies"]].sum(
        axis=1, min_count=3
    )

    model_features = pd.DataFrame(index=data.index)
    model_features["LeadTime"] = numeric["LeadTime"]
    model_features["TotalNights"] = total_nights
    model_features["PreviousCancellations"] = numeric["PreviousCancellations"]
    model_features["PreviousBookingsNotCanceled"] = numeric[
        "PreviousBookingsNotCanceled"
    ]
    model_features["WeekendShare"] = weekend_share
    model_features["PartySize"] = party_size
    model_features["ArrivalDateMonth"] = arrival_month

    for column in (
        "Meal",
        "MarketSegment",
        "DistributionChannel",
        "ReservedRoomType",
        "DepositType",
        "CustomerType",
    ):
        model_features[column] = _clean_categorical(data[column])

    model_features["IsRepeatedGuest"] = repeat_guest.map({0: "no", 1: "yes"})
    model_features["HasChildren"] = _presence_label(numeric["Children"])
    model_features["HasBabies"] = _presence_label(numeric["Babies"])
    model_features = model_features.loc[:, MODEL_FEATURE_COLUMNS]

    edge_case_flags = pd.DataFrame(
        {
            "ZeroPersonBooking": party_size.eq(0),
            "ZeroNightStay": total_nights.eq(0),
        },
        index=data.index,
    )

    if set(model_features).intersection(LEAKAGE_COLUMNS):
        raise AssertionError("Leakage columns entered the clustering feature frame.")

    return ClusteringFeatureBundle(
        model_features=model_features,
        source_data=source_data,
        edge_case_flags=edge_case_flags,
    )
