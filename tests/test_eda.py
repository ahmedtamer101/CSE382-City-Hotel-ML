
import pandas as pd
import pytest

from src.data.eda import (
    EXPECTED_COLUMNS,
    validate_schema,
    audit,
)


def make_sample_dataframe():
    """Create a small synthetic H2 dataset for testing."""

    df = pd.DataFrame(
        {column: [0, 0, 0] for column in EXPECTED_COLUMNS}
    )

    df["Adults"] = [2, 2, 0]
    df["Children"] = [0, 0, 0]
    df["Babies"] = [0, 0, 0]

    df["LeadTime"] = [10, 10, 20]
    df["ADR"] = [100, 100, 200]

    df["StaysInWeekNights"] = [2, 2, 0]
    df["StaysInWeekendNights"] = [0, 0, 0]

    df["IsCanceled"] = [0, 0, 1]

    return df


def test_expected_column_count():
    assert len(EXPECTED_COLUMNS) == 31
    assert len(set(EXPECTED_COLUMNS)) == 31


def test_schema_rejects_missing_column():
    df = make_sample_dataframe()
    df = df.drop(columns=["ADR"])

    with pytest.raises(ValueError, match="Invalid H2 dataset schema"):
        validate_schema(df)


def test_schema_rejects_extra_column():
    df = make_sample_dataframe()
    df["UnexpectedColumn"] = 123

    with pytest.raises(ValueError, match="Invalid H2 dataset schema"):
        validate_schema(df)


def test_schema_rejects_wrong_column_order():
    df = make_sample_dataframe()
    df = df[EXPECTED_COLUMNS[::-1]]

    with pytest.raises(ValueError, match="Invalid H2 dataset schema"):
        validate_schema(df)


def test_missing_value_detection(tmp_path):
    df = make_sample_dataframe()
    df.loc[0, "Country"] = None

    audit(df, tmp_path)

    profile = pd.read_csv(tmp_path / "column_profile.csv")

    country = profile.loc[profile["feature"] == "Country"]

    assert int(country["actual_missing"].iloc[0]) == 1


def test_duplicate_counting(tmp_path):
    df = make_sample_dataframe()

    audit(df, tmp_path)

    checks = pd.read_csv(tmp_path / "quality_checks.csv")

    duplicates = checks.loc[
        checks["check"] == "identical_rows_after_first",
        "count",
    ].iloc[0]

    assert int(duplicates) == 1


def test_eda_quality_calculations(tmp_path):
    df = make_sample_dataframe()

    audit(df, tmp_path)

    checks = pd.read_csv(tmp_path / "quality_checks.csv")

    values = dict(zip(checks["check"], checks["count"]))

    assert int(values["zero_total_guests"]) == 1
    assert int(values["zero_total_nights"]) == 1
    assert int(values["negative_lead_time"]) == 0
    assert int(values["negative_adr"]) == 0
    assert int(values["adr_over_1000"]) == 0
