"""Checks for report figures built from fixed labels and booking-scale values."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from src.clustering.evaluation import stratified_sample_indices
from src.clustering.experiment import run_kmeans_baseline
from src.clustering.features import LEAKAGE_COLUMNS
from src.clustering.visualization import (
    FIGURE_FILES,
    fit_pca_projection,
    plot_total_nights,
    prepare_plot_data,
    render_visualizations,
)
from tests.test_clustering_experiment import make_experiment_fixture


@pytest.fixture
def fixture_baseline():
    source = make_experiment_fixture(24)
    source.index = [i // 2 for i in range(len(source))]
    source_copy = source.copy(deep=True)
    baseline = run_kmeans_baseline(source, dataset_sha256="test-only")
    return source, source_copy, baseline


def test_plot_data_counts_percentages_alignment_and_no_mutation(fixture_baseline) -> None:
    source, original, baseline = fixture_baseline
    frame = prepare_plot_data(baseline)
    assert_frame_equal(source, original)
    assert len(frame) == len(source)
    assert_array_equal(frame.cluster_id.to_numpy(), baseline.labels)
    assert_array_equal(frame.LeadTime.to_numpy(), source.LeadTime.to_numpy())
    assert_array_equal(
        frame.TotalNights.to_numpy(),
        (source.StaysInWeekendNights + source.StaysInWeekNights).to_numpy(),
    )
    counts = frame.cluster_id.value_counts().sort_index().to_numpy()
    assert_array_equal(counts, baseline.cluster_sizes["count"])
    assert_allclose(counts / len(frame) * 100, baseline.cluster_sizes["percentage"])
    assert not set(frame.columns) & LEAKAGE_COLUMNS
    assert not any(column in name for column in LEAKAGE_COLUMNS for name in baseline.feature_names)


def test_plot_data_rejects_tampered_labels(fixture_baseline) -> None:
    _, _, baseline = fixture_baseline
    baseline.labels = baseline.labels[::-1].copy()
    if np.array_equal(np.bincount(baseline.labels), baseline.cluster_sizes["count"]):
        baseline.labels[0] = (baseline.labels[0] + 1) % 3
    with pytest.raises(ValueError, match="Cluster size table"):
        prepare_plot_data(baseline)


def test_pca_uses_full_fitted_matrix_and_has_valid_variance(fixture_baseline) -> None:
    _, _, baseline = fixture_baseline
    original_labels = baseline.labels.copy()
    first, metadata = fit_pca_projection(baseline)
    second, repeated = fit_pca_projection(baseline)
    assert_array_equal(first, second)
    assert metadata == repeated
    assert first.shape == (24, 2)
    assert metadata["fit_rows"] == 24
    assert metadata["input_features"] == baseline.transformed_matrix.shape[1]
    assert metadata["method"] == "centered_PCA"
    assert len(metadata["explained_variance_ratio"]) == 2
    assert all(0 <= value <= 1 for value in metadata["explained_variance_ratio"])
    assert 0 <= metadata["explained_variance_ratio_sum"] <= 1 + 1e-12
    assert_array_equal(original_labels, baseline.labels)
    with pytest.raises(ValueError, match="dense feature array"):
        fit_pca_projection(baseline, dense_budget_bytes=1)


def test_scatter_sample_is_fixed_and_stratified(fixture_baseline) -> None:
    _, _, baseline = fixture_baseline
    first = stratified_sample_indices(baseline.labels, max_samples=12, seed=42)
    second = stratified_sample_indices(baseline.labels, max_samples=12, seed=42)
    assert_array_equal(first, second)
    assert len(first) == 12
    assert len(np.unique(first)) == 12
    assert set(baseline.labels[first]) == {0, 1, 2}


def test_nights_plot_reports_long_tail_and_real_maximum(fixture_baseline, tmp_path, monkeypatch) -> None:
    _, _, baseline = fixture_baseline
    frame = prepare_plot_data(baseline)
    frame.loc[0, "TotalNights"] = 57
    captured = {}
    from src.clustering import visualization
    original_save = visualization._save_figure

    def inspect(figure, path):
        captured["caption"] = figure.texts[0].get_text()
        captured["tick_labels"] = [item.get_text() for item in figure.axes[0].get_xticklabels()]
        original_save(figure, path)

    monkeypatch.setattr(visualization, "_save_figure", inspect)
    destination = tmp_path / "nights.png"
    plot_total_nights(frame, destination)
    assert destination.is_file()
    assert ">10" in captured["tick_labels"]
    assert "57 nights" in captured["caption"]


def test_all_figures_and_metadata_are_deterministic(fixture_baseline, tmp_path) -> None:
    source, original, baseline = fixture_baseline
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = render_visualizations(baseline, first_dir, sample_size=12)
    second = render_visualizations(baseline, second_dir, sample_size=12)
    assert first == second
    assert_frame_equal(source, original)
    assert first["outcome_fields_used"] is False
    assert first["scatter_sample"]["rows"] == 12
    assert first["pca"]["fit_rows"] == len(source)
    assert first["cluster_counts"] == {
        str(row.cluster_id): int(row.count)
        for row in baseline.cluster_sizes.itertuples(index=False)
    }
    assert {item["file"] for item in first["figures"]} == {
        f"figures/{filename}" for filename in FIGURE_FILES.values()
    }
    for name in FIGURE_FILES.values():
        left = first_dir / "figures" / name
        right = second_dir / "figures" / name
        assert left.is_file() and left.stat().st_size > 1000
        assert hashlib.sha256(left.read_bytes()).digest() == hashlib.sha256(right.read_bytes()).digest()
    assert json.loads((first_dir / "visualization_metadata.json").read_text()) == first
