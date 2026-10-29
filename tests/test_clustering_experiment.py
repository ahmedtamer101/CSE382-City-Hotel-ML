"""Tests for the untuned K-Means baseline experiment."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.clustering.experiment import (
    CLUSTER_SIZE_COLUMNS,
    KMeansBaselineConfig,
    build_kmeans_baseline,
    load_h2_dataset,
    main,
    run_kmeans_baseline,
    write_baseline_outputs,
)
from src.clustering.features import LEAKAGE_COLUMNS
from tests.test_clustering_features import make_h2_fixture


def make_experiment_fixture(rows: int = 12) -> pd.DataFrame:
    """Create distinct valid bookings from the existing tiny H2 fixture."""
    template = make_h2_fixture().iloc[0].copy()
    months = ["January", "April", "August", "December"]
    meals = ["BB", "HB", "SC"]
    segments = ["Direct", "Online TA", "Groups"]
    records = []
    for index in range(rows):
        row = template.copy()
        row["LeadTime"] = index * 17
        row["ArrivalDateMonth"] = months[index % len(months)]
        row["StaysInWeekendNights"] = index % 3
        row["StaysInWeekNights"] = 1 + index % 5
        row["Adults"] = 1 + index % 3
        row["Children"] = float(index % 2)
        row["Babies"] = int(index % 7 == 0)
        row["Meal"] = meals[index % len(meals)]
        row["MarketSegment"] = segments[index % len(segments)]
        row["DistributionChannel"] = (
            "Direct" if index % 3 == 0 else "TA/TO"
        )
        row["IsRepeatedGuest"] = index % 2
        row["PreviousCancellations"] = index % 4
        row["PreviousBookingsNotCanceled"] = index % 6
        row["ReservedRoomType"] = ["A", "D", "E"][index % 3]
        row["DepositType"] = (
            "No Deposit" if index % 2 == 0 else "Non Refund"
        )
        row["CustomerType"] = (
            "Transient" if index % 2 == 0 else "Transient-Party"
        )
        row["IsCanceled"] = index % 2
        row["ADR"] = 50.0 + index * 100
        row["ReservationStatus"] = (
            "Canceled" if index % 2 else "Check-Out"
        )
        records.append(row)
    return pd.DataFrame(records, columns=make_h2_fixture().columns)


def test_default_kmeans_configuration_is_exact() -> None:
    model = build_kmeans_baseline()
    parameters = model.get_params()

    assert parameters["n_clusters"] == 3
    assert parameters["init"] == "k-means++"
    assert parameters["n_init"] == 10
    assert parameters["random_state"] == 42
    assert parameters["max_iter"] == 300


def test_baseline_assigns_one_valid_label_per_row() -> None:
    data = make_experiment_fixture()

    result = run_kmeans_baseline(data)

    assert len(result.labels) == len(data)
    assert set(result.labels).issubset({0, 1, 2})
    assert result.cluster_sizes["count"].sum() == len(data)
    assert result.cluster_sizes["percentage"].sum() == pytest.approx(100.0)
    assert tuple(result.cluster_sizes.columns) == CLUSTER_SIZE_COLUMNS


def test_fixed_seed_reproduces_labels_and_inertia() -> None:
    data = make_experiment_fixture()

    first = run_kmeans_baseline(data)
    second = run_kmeans_baseline(data)

    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.metrics["inertia"] == pytest.approx(
        second.metrics["inertia"], rel=1e-12, abs=1e-12
    )


def test_outcome_columns_cannot_affect_fitted_input_or_labels() -> None:
    original = make_experiment_fixture()
    altered = original.copy(deep=True)
    for column in LEAKAGE_COLUMNS:
        if pd.api.types.is_numeric_dtype(altered[column]):
            altered[column] = altered[column].fillna(0) + 1_000_000
        else:
            altered[column] = "LEAKAGE_SENTINEL"

    original_result = run_kmeans_baseline(original)
    altered_result = run_kmeans_baseline(altered)

    np.testing.assert_array_equal(original_result.labels, altered_result.labels)
    assert original_result.metrics["inertia"] == pytest.approx(
        altered_result.metrics["inertia"], rel=1e-12, abs=1e-12
    )
    assert not any(
        excluded in name
        for excluded in LEAKAGE_COLUMNS
        for name in original_result.feature_names
    )


@pytest.mark.parametrize(
    "config, message",
    [
        (KMeansBaselineConfig(n_clusters=1), "n_clusters"),
        (KMeansBaselineConfig(n_init=0), "n_init"),
        (KMeansBaselineConfig(max_iter=0), "max_iter"),
    ],
)
def test_invalid_configuration_fails_clearly(
    config: KMeansBaselineConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_kmeans_baseline(config)


def test_cluster_count_cannot_exceed_observations() -> None:
    data = make_experiment_fixture(rows=3)

    with pytest.raises(ValueError, match="observations"):
        run_kmeans_baseline(data, KMeansBaselineConfig(n_clusters=4))


def test_phase_two_preprocessing_and_output_structure_are_compatible() -> None:
    data = make_experiment_fixture()

    result = run_kmeans_baseline(data)

    assert result.metrics["source_feature_count"] == 31
    assert result.metrics["engineered_feature_count"] == 16
    assert result.metrics["transformed_feature_count"] == len(
        result.feature_names
    )
    assert result.metadata["matrix"]["is_sparse"] is True
    assert result.metadata["row_policy"]["all_input_rows_preserved"] is True
    assert result.metadata["model_serialized"] is False


def test_small_output_files_have_expected_structure(tmp_path) -> None:
    result = run_kmeans_baseline(make_experiment_fixture())

    paths = write_baseline_outputs(result, tmp_path)

    assert set(paths) == {"metrics", "cluster_sizes", "metadata"}
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    sizes = pd.read_csv(paths["cluster_sizes"])
    assert metrics["experiment"] == "untuned_kmeans_baseline"
    assert "inertia" in metrics
    assert "package_versions" in metadata
    assert metadata["reproduction_command"].startswith(
        "python -m src.clustering.experiment"
    )
    assert tuple(sizes.columns) == CLUSTER_SIZE_COLUMNS
    assert sizes["count"].sum() == len(result.labels)


def test_loader_rejects_invalid_source_schema(tmp_path) -> None:
    invalid_path = tmp_path / "invalid.csv"
    make_experiment_fixture().drop(columns="LeadTime").to_csv(
        invalid_path, index=False
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        load_h2_dataset(invalid_path, expected_rows=None)


def test_cli_reports_missing_dataset_without_creating_outputs(
    tmp_path, capsys
) -> None:
    output_directory = tmp_path / "outputs"

    exit_code = main(
        [
            "--dataset",
            str(tmp_path / "missing.csv"),
            "--output-dir",
            str(output_directory),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "not found" in captured.err
    assert not output_directory.exists()
