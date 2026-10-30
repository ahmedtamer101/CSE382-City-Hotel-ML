"""Report figures for the verified, fixed City Hotel K-Means baseline.

Figures describe original-scale booking attributes. PCA alone uses the exact
48-column matrix fitted by K-Means; the existing labels are never refitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from src.clustering.evaluation import (
    DEFAULT_DENSE_BUDGET_BYTES,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    evaluate_baseline,
    stratified_sample_indices,
    validate_labels,
    verify_committed_baseline,
)
from src.clustering.experiment import (
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_DIRECTORY,
    KMeansBaselineResult,
    compute_sha256,
    load_h2_dataset,
    run_kmeans_baseline,
)
from src.clustering.features import LEAKAGE_COLUMNS, MODEL_FEATURE_COLUMNS


FIGURE_FILES: Final[dict[str, str]] = {
    "sizes": "cluster_size_distribution.png",
    "lead_time": "lead_time_by_cluster.png",
    "total_nights": "total_nights_by_cluster.png",
    "market_segment": "market_segment_by_cluster.png",
    "deposit_type": "deposit_type_by_cluster.png",
    "pca": "pca_cluster_projection.png",
}
PROFILE_INPUTS: Final[tuple[str, ...]] = (
    "LeadTime", "TotalNights", "MarketSegment", "DepositType",
)
COLORS: Final[tuple[str, ...]] = ("#2364AA", "#F18F01", "#6B4E9B")


def prepare_plot_data(baseline: KMeansBaselineResult) -> pd.DataFrame:
    """Join untouched booking characteristics to labels by row position."""
    bundle = baseline.feature_bundle
    features = bundle.model_features
    labels = validate_labels(baseline.labels, len(features))
    if len(bundle.source_data) != len(features):
        raise ValueError("Source rows and fitted features are misaligned.")
    if tuple(features.columns) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Unexpected clustering feature contract.")
    if set(features.columns) & LEAKAGE_COLUMNS:
        raise ValueError("Outcome field entered the fitted feature representation.")
    if baseline.transformed_matrix.shape != (len(labels), len(baseline.feature_names)):
        raise ValueError("Fitted matrix and labels are misaligned.")
    counts = np.bincount(labels)
    expected = baseline.cluster_sizes.sort_values("cluster_id")
    if not np.array_equal(expected["cluster_id"].to_numpy(), np.arange(len(counts))):
        raise ValueError("Cluster size table has unexpected IDs.")
    if not np.array_equal(expected["count"].to_numpy(), counts):
        raise ValueError("Cluster size table does not match fitted labels.")
    frame = features.loc[:, PROFILE_INPUTS].reset_index(drop=True).copy(deep=True)
    for column in ("LeadTime", "TotalNights"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError(f"{column} must be finite and nonnegative for plotting.")
        frame[column] = numeric
    frame["cluster_id"] = labels.copy()
    return frame


def fit_pca_projection(
    baseline: KMeansBaselineResult,
    *,
    dense_budget_bytes: int = DEFAULT_DENSE_BUDGET_BYTES,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit centered, two-component PCA to every fitted K-Means input row."""
    matrix = baseline.transformed_matrix
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("PCA needs at least three rows and two fitted features.")
    if matrix.shape[1] != len(baseline.feature_names):
        raise ValueError("PCA feature names do not match the fitted matrix.")
    if dense_budget_bytes < 1:
        raise ValueError("Dense matrix budget must be positive.")
    estimate = int(matrix.shape[0] * matrix.shape[1] * np.dtype(matrix.dtype).itemsize)
    if estimate > dense_budget_bytes:
        raise ValueError(
            f"Centered PCA needs a dense feature array of {estimate} bytes, "
            f"above the {dense_budget_bytes}-byte limit."
        )
    if sparse.issparse(matrix):
        if not np.isfinite(matrix.data).all():
            raise ValueError("PCA input contains NaN or infinity.")
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    if not np.isfinite(dense).all():
        raise ValueError("PCA input contains NaN or infinity.")
    model = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    coordinates = model.fit_transform(dense)
    variance = [float(value) for value in model.explained_variance_ratio_]
    metadata = {
        "method": "centered_PCA",
        "solver": "randomized",
        "random_state": seed,
        "fit_rows": int(matrix.shape[0]),
        "input_features": int(matrix.shape[1]),
        "dense_input_bytes_estimate": estimate,
        "dense_budget_bytes": dense_budget_bytes,
        "explained_variance_ratio": variance,
        "explained_variance_ratio_sum": float(sum(variance)),
        "cluster_labels_refitted": False,
    }
    return coordinates, metadata


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _cluster_counts(frame: pd.DataFrame) -> pd.Series:
    return frame["cluster_id"].value_counts().sort_index()


def plot_cluster_sizes(frame: pd.DataFrame, path: Path) -> None:
    counts = _cluster_counts(frame)
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(counts))
    bars = axis.bar(positions, counts.to_numpy(), color=COLORS[:len(counts)])
    axis.set_xticks(positions, [f"Cluster {i}" for i in counts.index])
    axis.set_ylabel("Bookings (count)")
    axis.set_title("City Hotel bookings by K-Means cluster")
    axis.set_ylim(0, counts.max() * 1.16)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    for bar, count in zip(bars, counts, strict=True):
        axis.annotate(
            f"{count:,}\n({count / len(frame):.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
            fontsize=10,
        )
    _save_figure(figure, path)


def plot_lead_time(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for cluster in sorted(frame["cluster_id"].unique()):
        values = np.sort(frame.loc[frame.cluster_id == cluster, "LeadTime"].to_numpy())
        axis.step(
            values, np.arange(1, len(values) + 1) / len(values) * 100,
            where="post", label=f"Cluster {cluster} (n={len(values):,})",
            color=COLORS[int(cluster)], linewidth=2,
        )
    axis.set(xlabel="Lead time (days before arrival)", ylabel="Bookings at or below value (%)",
             title="Lead time distribution by cluster", xlim=(0, frame.LeadTime.max()), ylim=(0, 101))
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", frameon=True)
    _save_figure(figure, path)


def plot_total_nights(frame: pd.DataFrame, path: Path, *, display_max: int = 10) -> None:
    """Show 0 through display_max nights plus an explicit long-stay tail."""
    if display_max < 1:
        raise ValueError("display_max must be positive.")
    groups = sorted(frame["cluster_id"].unique())
    labels = [str(i) for i in range(display_max + 1)] + [f">{display_max}"]
    positions = np.arange(len(labels))
    width = 0.78 / len(groups)
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    for index, cluster in enumerate(groups):
        values = frame.loc[frame.cluster_id == cluster, "TotalNights"]
        percentages = [float(values.eq(i).mean() * 100) for i in range(display_max + 1)]
        percentages.append(float(values.gt(display_max).mean() * 100))
        axis.bar(positions + (index - (len(groups) - 1) / 2) * width,
                 percentages, width=width, color=COLORS[int(cluster)],
                 label=f"Cluster {cluster}")
    axis.set_xticks(positions, labels)
    axis.set(xlabel="Total nights booked", ylabel="Within-cluster bookings (%)",
             title="Length of stay by cluster")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend()
    figure.text(0.01, 0.01,
                f">{display_max} includes {int(frame.TotalNights.gt(display_max).sum()):,} "
                f"bookings; observed maximum = {frame.TotalNights.max():g} nights.",
                fontsize=9)
    _save_figure(figure, path)


def plot_category_percentages(frame: pd.DataFrame, column: str, path: Path) -> None:
    if column not in {"MarketSegment", "DepositType"}:
        raise ValueError("Only approved booking categories may be plotted.")
    categories = frame[column].fillna("__MISSING__")
    order = categories.value_counts().index.tolist()
    groups = sorted(frame.cluster_id.unique())
    positions = np.arange(len(order))
    width = 0.78 / len(groups)
    figure, axis = plt.subplots(figsize=(10, 6 if len(order) > 4 else 5))
    for index, cluster in enumerate(groups):
        selected = categories.loc[frame.cluster_id.eq(cluster)]
        percentages = selected.value_counts(normalize=True).reindex(order, fill_value=0) * 100
        axis.bar(positions + (index - (len(groups) - 1) / 2) * width,
                 percentages, width=width, color=COLORS[int(cluster)],
                 label=f"Cluster {cluster}")
    axis.set_xticks(positions, order, rotation=30 if len(order) > 4 else 0, ha="right" if len(order) > 4 else "center")
    axis.set(xlabel=column, ylabel="Within-cluster bookings (%)",
             title=f"{column} composition by cluster")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend()
    _save_figure(figure, path)


def plot_pca(
    coordinates: np.ndarray, labels: np.ndarray, variance: Sequence[float],
    indices: np.ndarray, path: Path,
) -> None:
    values = validate_labels(labels, len(coordinates))
    if coordinates.shape != (len(values), 2) or len(variance) != 2:
        raise ValueError("PCA plot requires two scores and two variance ratios.")
    if not np.isfinite(coordinates).all():
        raise ValueError("PCA scores contain NaN or infinity.")
    figure, axis = plt.subplots(figsize=(9, 6.5))
    for cluster in np.unique(values):
        selected = indices[values[indices] == cluster]
        axis.scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            s=11, alpha=0.42, color=COLORS[int(cluster)], rasterized=True,
            label=f"Cluster {cluster} (n={len(selected):,} shown)",
        )
    axis.set(
        xlabel=f"PC1 score ({variance[0]:.1%} variance)",
        ylabel=f"PC2 score ({variance[1]:.1%} variance)",
        title="PCA view of fitted K-Means clusters",
    )
    axis.grid(alpha=0.2)
    axis.legend(markerscale=2, frameon=True)
    figure.text(0.01, 0.01,
                "PCA fitted on all bookings; plotted rows are a fixed proportional sample. "
                "Distances in this view omit other fitted dimensions.", fontsize=9)
    _save_figure(figure, path)


def render_visualizations(
    baseline: KMeansBaselineResult, output_dir: Path, *,
    sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SAMPLE_SEED,
    dense_budget_bytes: int = DEFAULT_DENSE_BUDGET_BYTES,
) -> dict[str, Any]:
    """Write six report figures and deterministic, aggregate provenance metadata."""
    frame = prepare_plot_data(baseline)
    if not 1 <= sample_size <= DEFAULT_SAMPLE_SIZE:
        raise ValueError("Scatter sample size must be between 1 and 3000.")
    indices = stratified_sample_indices(baseline.labels, max_samples=sample_size, seed=seed)
    coordinates, pca_metadata = fit_pca_projection(
        baseline, dense_budget_bytes=dense_budget_bytes, seed=seed
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_cluster_sizes(frame, figure_dir / FIGURE_FILES["sizes"])
    plot_lead_time(frame, figure_dir / FIGURE_FILES["lead_time"])
    plot_total_nights(frame, figure_dir / FIGURE_FILES["total_nights"])
    plot_category_percentages(frame, "MarketSegment", figure_dir / FIGURE_FILES["market_segment"])
    plot_category_percentages(frame, "DepositType", figure_dir / FIGURE_FILES["deposit_type"])
    plot_pca(coordinates, baseline.labels, pca_metadata["explained_variance_ratio"],
             indices, figure_dir / FIGURE_FILES["pca"])
    descriptions = {
        "sizes": ("verified fixed K-Means labels", "all_rows"),
        "lead_time": ("LeadTime in original days; verified labels", "all_rows"),
        "total_nights": ("TotalNights in original nights; verified labels; >10 tail", "all_rows"),
        "market_segment": ("original MarketSegment; verified labels", "all_rows"),
        "deposit_type": ("original DepositType; verified labels", "all_rows"),
        "pca": ("exact fitted 48-feature matrix; verified labels", "stratified_sample"),
    }
    metadata = {
        "dataset_sha256": baseline.metadata["dataset_sha256"],
        "baseline_parameters": baseline.metrics["kmeans_parameters"],
        "baseline_inertia": baseline.metrics["inertia"],
        "dataset_rows": len(frame),
        "cluster_counts": {str(i): int(n) for i, n in _cluster_counts(frame).items()},
        "label_source": "reproduced fixed Phase 3 K-Means baseline; no PCA refit of labels",
        "outcome_fields_used": False,
        "scatter_sample": {
            "method": "proportional_stratified_without_replacement",
            "seed": seed,
            "rows": len(indices),
            "cluster_counts": {str(i): int(n) for i, n in
                               enumerate(np.bincount(baseline.labels[indices]))},
        },
        "pca": pca_metadata,
        "figures": [
            {"file": f"figures/{FIGURE_FILES[key]}", "inputs": inputs, "scope": scope}
            for key, (inputs, scope) in descriptions.items()
        ],
    }
    (output_dir / "visualization_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _verify_evaluation(baseline: KMeansBaselineResult, output_dir: Path) -> None:
    """Check that the recomputed fixed evaluation matches committed Phase 4."""
    with (output_dir / "evaluation_metrics.json").open(encoding="utf-8") as stream:
        expected = json.load(stream)
    with (output_dir / "evaluation_metadata.json").open(encoding="utf-8") as stream:
        expected_metadata = json.load(stream)
    if expected_metadata["dataset_sha256"] != baseline.metadata["dataset_sha256"]:
        raise ValueError("Evaluation dataset checksum differs from the baseline.")
    actual = evaluate_baseline(baseline)
    for metric in ("silhouette_score", "davies_bouldin_index"):
        if not np.isclose(actual.metrics[metric], expected[metric], rtol=1e-9, atol=1e-9):
            raise ValueError(f"Committed {metric} does not match reproduced evaluation.")
    if actual.metrics["cluster_sizes"] != expected["cluster_sizes"]:
        raise ValueError("Committed evaluation cluster sizes do not match the fitted labels.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render verified City Hotel clustering figures.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    args = parser.parse_args(argv)
    try:
        data = load_h2_dataset(args.dataset)
        sha = compute_sha256(args.dataset)
        baseline = run_kmeans_baseline(data, dataset_sha256=sha,
                                       dataset_path=args.dataset.as_posix())
        verify_committed_baseline(
            baseline, args.output_dir / "baseline_metrics.json",
            args.output_dir / "baseline_cluster_sizes.csv",
            args.output_dir / "baseline_metadata.json", sha,
        )
        _verify_evaluation(baseline, args.output_dir)
        metadata = render_visualizations(
            baseline, args.output_dir, sample_size=args.sample_size, seed=args.seed
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"clustering visualization failed: {error}", file=sys.stderr)
        return 2
    print(f"Created {len(metadata['figures'])} figures for {metadata['dataset_rows']:,} "
          f"verified bookings. PCA variance: "
          f"{metadata['pca']['explained_variance_ratio_sum']:.2%} in two components.")
    print(args.output_dir / "visualization_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
