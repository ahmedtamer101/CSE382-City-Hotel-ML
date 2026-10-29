import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features import FEATURE_SETS, add_features, feature_columns
from src.preprocessing import (EXPECTED_COLUMNS, assign_groups, build_preprocessor,
                               clean_training_data, make_splits, prepare_inference_frame,
                               standardize, validate_schema)

LEAKY_ALWAYS = {"ReservationStatus", "ReservationStatusDate", "AssignedRoomType", "BookingChanges"}


def _prepared(df):
    clean, _ = clean_training_data(df)
    data = add_features(clean)
    data["GroupId"] = assign_groups(data, EXPECTED_COLUMNS)
    data["Split"] = make_splits(data, data["GroupId"], seed=42)
    return data


def test_schema_check_rejects_wrong_file(fake_h2):
    validate_schema(fake_h2)
    with pytest.raises(ValueError):
        validate_schema(fake_h2.drop(columns=["ADR"]))
    with pytest.raises(ValueError):
        validate_schema(fake_h2.assign(Hotel="City Hotel"))


def test_standardize_fixes_padding_and_hidden_missing(fake_h2):
    s = standardize(fake_h2)
    assert not s["Meal"].str.contains(" $", regex=True).any()
    assert "NULL" not in set(s["Agent"]) and "None" in set(s["Agent"])
    assert s["Company"].isna().mean() > 0.8
    assert "Undefined" not in set(s["MarketSegment"].dropna())


def test_cleaning_drops_invalid_rows(fake_h2):
    clean, log = clean_training_data(fake_h2)
    assert clean["ADR"].max() <= 1000
    guests = clean["Adults"] + clean["Children"].fillna(0) + clean["Babies"]
    assert (guests > 0).all()
    assert log["rows_remaining"].iloc[-1] == len(clean)


def test_no_target_or_leaky_column_in_any_feature_set():
    for task, fs in FEATURE_SETS.items():
        cols = set(feature_columns(task))
        assert not (cols & LEAKY_ALWAYS), task
        assert "IsCanceled" not in cols, task
        if fs["target"]:
            assert fs["target"] not in cols, task
    assert "ADR" not in feature_columns("regression")
    assert "Country" not in feature_columns("classification")
    assert "RequiredCarParkingSpaces" not in feature_columns("classification")
    assert "DepositType" not in feature_columns("clustering")


def test_split_is_disjoint_grouped_and_stratified(fake_h2):
    data = _prepared(fake_h2)
    assert set(data["Split"]) == {"train", "val", "test"}
    assert data.groupby("GroupId")["Split"].nunique().max() == 1  # duplicates never cross splits
    shares = data["Split"].value_counts(normalize=True)
    assert 0.5 < shares["train"] < 0.7 and 0.1 < shares["test"] < 0.3
    rates = data.groupby("Split")["IsCanceled"].mean()
    assert rates.max() - rates.min() < 0.1


def test_split_is_reproducible(fake_h2):
    a = _prepared(fake_h2)["Split"]
    b = _prepared(fake_h2)["Split"]
    assert (a == b).all()


@pytest.mark.parametrize("task", ["regression", "classification", "clustering"])
def test_preprocessor_fits_on_train_and_handles_unseen(fake_h2, task):
    data = _prepared(fake_h2)
    train = data[data["Split"] == "train"]
    prep = build_preprocessor(task).fit(train[feature_columns(task)])
    Xt = prep.transform(data[data["Split"] == "test"][feature_columns(task)])
    assert not np.isnan(Xt).any()
    assert Xt.shape[1] == len(prep.get_feature_names_out())

    new = prepare_inference_frame({
        "LeadTime": 30, "ArrivalDateYear": 2017, "ArrivalDateMonth": "May",
        "ArrivalDateWeekNumber": 20, "ArrivalDateDayOfMonth": 14,
        "StaysInWeekendNights": 1, "StaysInWeekNights": 2, "Adults": 2,
        "Meal": "BB", "Country": "NEVER_SEEN", "MarketSegment": "Online TA",
        "Agent": 9, "ADR": 95.0, "CustomerType": "Transient",
    }, task)  # several fields deliberately missing
    out = prep.transform(new)
    assert out.shape == (1, Xt.shape[1]) and not np.isnan(out).any()


def test_scaler_statistics_come_from_train_only(fake_h2):
    data = _prepared(fake_h2)
    train = data[data["Split"] == "train"]
    prep = build_preprocessor("regression").fit(train[feature_columns("regression")])
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    idx = FEATURE_SETS["regression"]["numeric"].index("Adults")
    assert np.isclose(scaler.mean_[idx], train["Adults"].mean())
