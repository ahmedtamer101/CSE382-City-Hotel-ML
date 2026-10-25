"""Tests for leakage-safe clustering feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.clustering.features import (
    CLUSTERING_INPUT_ALLOWLIST,
    EXPECTED_SOURCE_COLUMNS,
    LEAKAGE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    build_booking_characteristics,
    validate_source_schema,
)


def make_h2_fixture() -> pd.DataFrame:
    rows = [
        {
            "IsCanceled": 0,
            "LeadTime": 10,
            "ArrivalDateYear": 2016,
            "ArrivalDateMonth": "January",
            "ArrivalDateWeekNumber": 1,
            "ArrivalDateDayOfMonth": 2,
            "StaysInWeekendNights": 1,
            "StaysInWeekNights": 2,
            "Adults": 2,
            "Children": 1.0,
            "Babies": 0,
            "Meal": "BB",
            "Country": "PRT",
            "MarketSegment": "Direct",
            "DistributionChannel": "Direct",
            "IsRepeatedGuest": 0,
            "PreviousCancellations": 0,
            "PreviousBookingsNotCanceled": 0,
            "ReservedRoomType": "A",
            "AssignedRoomType": "A",
            "BookingChanges": 0,
            "DepositType": "No Deposit",
            "Agent": np.nan,
            "Company": np.nan,
            "DaysInWaitingList": 0,
            "CustomerType": "Transient",
            "ADR": 100.0,
            "RequiredCarParkingSpaces": 0,
            "TotalOfSpecialRequests": 1,
            "ReservationStatus": "Check-Out",
            "ReservationStatusDate": "2016-01-05",
        },
        {
            "IsCanceled": 1,
            "LeadTime": 100,
            "ArrivalDateYear": 2017,
            "ArrivalDateMonth": "December",
            "ArrivalDateWeekNumber": 50,
            "ArrivalDateDayOfMonth": 15,
            "StaysInWeekendNights": 0,
            "StaysInWeekNights": 0,
            "Adults": 0,
            "Children": 0.0,
            "Babies": 0,
            "Meal": "SC",
            "Country": np.nan,
            "MarketSegment": "Online TA",
            "DistributionChannel": "TA/TO",
            "IsRepeatedGuest": 1,
            "PreviousCancellations": 2,
            "PreviousBookingsNotCanceled": 3,
            "ReservedRoomType": "D",
            "AssignedRoomType": "E",
            "BookingChanges": 4,
            "DepositType": "Non Refund",
            "Agent": 9,
            "Company": np.nan,
            "DaysInWaitingList": 5,
            "CustomerType": "Transient-Party",
            "ADR": 250.0,
            "RequiredCarParkingSpaces": 1,
            "TotalOfSpecialRequests": 2,
            "ReservationStatus": "Canceled",
            "ReservationStatusDate": "2016-12-01",
        },
    ]
    return pd.DataFrame(rows, columns=EXPECTED_SOURCE_COLUMNS)


def test_allowlist_and_model_features_exclude_leakage() -> None:
    data = make_h2_fixture()
    bundle = build_booking_characteristics(data)

    assert tuple(bundle.model_features.columns) == MODEL_FEATURE_COLUMNS
    assert set(CLUSTERING_INPUT_ALLOWLIST).isdisjoint(LEAKAGE_COLUMNS)
    assert set(bundle.model_features.columns).isdisjoint(LEAKAGE_COLUMNS)


def test_source_schema_requires_all_and_only_h2_columns() -> None:
    data = make_h2_fixture().drop(columns="LeadTime")

    with pytest.raises(ValueError, match="LeadTime"):
        validate_source_schema(data)


def test_derived_stay_party_and_zero_edge_cases() -> None:
    bundle = build_booking_characteristics(make_h2_fixture())
    first = bundle.model_features.iloc[0]
    second = bundle.model_features.iloc[1]

    assert first["TotalNights"] == 3
    assert first["WeekendShare"] == pytest.approx(1 / 3)
    assert first["PartySize"] == 3
    assert first["HasChildren"] == "yes"
    assert first["HasBabies"] == "no"
    assert second["TotalNights"] == 0
    assert second["WeekendShare"] == 0
    assert second["PartySize"] == 0
    assert bool(bundle.edge_case_flags.iloc[1]["ZeroNightStay"])
    assert bool(bundle.edge_case_flags.iloc[1]["ZeroPersonBooking"])


def test_missing_child_count_is_preserved_for_fitted_imputation() -> None:
    data = make_h2_fixture()
    data.loc[0, "Children"] = np.nan

    bundle = build_booking_characteristics(data)

    assert pd.isna(bundle.model_features.loc[0, "PartySize"])
    assert pd.isna(bundle.model_features.loc[0, "HasChildren"])


@pytest.mark.parametrize("column", ["LeadTime", "Adults", "PreviousCancellations"])
def test_negative_selected_inputs_are_rejected(column: str) -> None:
    data = make_h2_fixture()
    data.loc[0, column] = -1

    with pytest.raises(ValueError, match="nonnegative"):
        build_booking_characteristics(data)


def test_invalid_month_is_rejected() -> None:
    data = make_h2_fixture()
    data.loc[0, "ArrivalDateMonth"] = "Smarch"

    with pytest.raises(ValueError, match="ArrivalDateMonth"):
        build_booking_characteristics(data)


def test_source_dataframe_is_not_mutated_and_is_preserved() -> None:
    data = make_h2_fixture()
    original = data.copy(deep=True)

    bundle = build_booking_characteristics(data)

    assert_frame_equal(data, original)
    assert_frame_equal(bundle.source_data, original)
    assert bundle.source_data is not data
