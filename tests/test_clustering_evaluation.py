"""Focused tests for bounded metrics and aligned booking profiles."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import sparse
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.clustering.evaluation import (
    build_cluster_profiles,
    compute_internal_metrics,
    evaluate_baseline,
    stratified_sample_indices,
    validate_labels,
    write_evaluation_outputs,
)
from src.clustering.experiment import run_kmeans_baseline
from src.clustering.features import LEAKAGE_COLUMNS
from tests.test_clustering_experiment import make_experiment_fixture


def test_metrics_match_sklearn_on_tiny_matrix() -> None:
    matrix = sparse.csr_matrix([[0., 0.], [0., 1.], [8., 8.], [9., 8.]])
    labels = np.array([0, 0, 1, 1])
    metrics, metadata = compute_internal_metrics(matrix, labels, max_samples=4)
    assert metrics["silhouette_score"] == pytest.approx(silhouette_score(matrix, labels))
    assert metrics["davies_bouldin_index"] == pytest.approx(
        davies_bouldin_score(matrix.toarray(), labels)
    )
    assert metrics["davies_bouldin_scope"] == "full_data"
    assert metadata["sample_cluster_counts"] == {"0": 2, "1": 2}


def test_sample_is_stratified_and_reproducible() -> None:
    labels = np.array([0] * 50 + [1] * 30 + [2] * 20)
    first = stratified_sample_indices(labels, max_samples=20, seed=42)
    second = stratified_sample_indices(labels, max_samples=20, seed=42)
    assert_array_equal(first, second)
    assert_array_equal(np.bincount(labels[first]), [10, 6, 4])
    assert len(np.unique(first)) == 20


@pytest.mark.parametrize("labels, rows", [
    (np.array([0, 1]), 3),
    (np.array([0, 0, 0]), 3),
    (np.array([0, 1, 2]), 3),
    (np.array([0, 2, 2]), 3),
])
def test_invalid_labels_fail(labels, rows) -> None:
    with pytest.raises(ValueError):
        validate_labels(labels, rows)


def test_tiny_sample_and_nonfinite_matrix_fail() -> None:
    matrix = sparse.csr_matrix([[0., 0.], [0., 1.], [8., 8.], [9., 8.]])
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="Sample size"):
        compute_internal_metrics(matrix, labels, max_samples=2)
    invalid = matrix.copy()
    invalid.data[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compute_internal_metrics(invalid, labels, max_samples=4)


def test_dense_budget_selects_sampled_dbi() -> None:
    matrix = sparse.csr_matrix([[0., 0.], [0., 1.], [8., 8.], [9., 8.]])
    labels = np.array([0, 0, 1, 1])
    metrics, metadata = compute_internal_metrics(
        matrix, labels, max_samples=4, dense_budget_bytes=48
    )
    assert metrics["davies_bouldin_scope"] == "stratified_sample"
    assert metadata["dbi_rows"] == 3


def test_metric_memory_limits_fail_clearly() -> None:
    matrix = sparse.csr_matrix([[0., 0.], [0., 1.], [8., 8.], [9., 8.]])
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="between 1 and 3000"):
        compute_internal_metrics(matrix, labels, max_samples=3001)
    with pytest.raises(ValueError, match="Dense budget"):
        compute_internal_metrics(matrix, labels, max_samples=4, dense_budget_bytes=1)


def test_profiles_use_original_units_and_positional_alignment() -> None:
    source = make_experiment_fixture(12)
    source.index = [7, 7, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0]
    original = source.copy(deep=True)
    baseline = run_kmeans_baseline(source)
    profiles, categories = build_cluster_profiles(baseline)
    assert_frame_equal(source, original)
    assert profiles["count"].sum() == 12
    assert profiles["percentage"].sum() == pytest.approx(100.0)
    assert set(categories["feature"]) >= {"MarketSegment", "DepositType", "Meal"}
    features = baseline.feature_bundle.model_features.reset_index(drop=True)
    for row in profiles.itertuples(index=False):
        selected = features.iloc[np.flatnonzero(baseline.labels == row.cluster_id)]
        assert row.LeadTime_median == pytest.approx(selected["LeadTime"].median())
        assert row.TotalNights_q25 == pytest.approx(selected["TotalNights"].quantile(.25))
        assert row.PartySize_median == pytest.approx(selected["PartySize"].median())
        expected_top = selected["MarketSegment"].value_counts().index[0]
        assert row.MarketSegment_top1 == expected_top
    assert not any(
        column in name for column in LEAKAGE_COLUMNS for name in baseline.feature_names
    )


def test_evaluation_keeps_baseline_counts_and_input_unchanged() -> None:
    source = make_experiment_fixture(12)
    original = source.copy(deep=True)
    baseline = run_kmeans_baseline(source)
    evaluated = evaluate_baseline(baseline, max_samples=12)
    assert_frame_equal(source, original)
    assert evaluated.metrics["cluster_count"] == 3
    assert [item["count"] for item in evaluated.metrics["cluster_sizes"]] == list(
        baseline.cluster_sizes["count"]
    )
    assert evaluated.metadata["outcome_fields_used"] is False


def test_evaluation_writes_only_aggregate_tables(tmp_path) -> None:
    evaluated = evaluate_baseline(
        run_kmeans_baseline(make_experiment_fixture(12)), max_samples=12
    )
    paths = write_evaluation_outputs(evaluated, tmp_path)
    assert set(paths) == {"metrics", "profiles", "categories", "metadata"}
    assert {path.name for path in paths.values()} == {
        "evaluation_metrics.json", "cluster_profiles.csv",
        "cluster_category_profiles.csv", "evaluation_metadata.json",
    }
    assert json.loads(paths["metrics"].read_text())["cluster_count"] == 3
    assert len(pd.read_csv(paths["profiles"])) == 3
    assert "lift" in pd.read_csv(paths["categories"]).columns
