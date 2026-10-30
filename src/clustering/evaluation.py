"""Bounded internal evaluation and original-scale profiles for the fixed baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.clustering.experiment import (
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_DIRECTORY,
    KMeansBaselineResult,
    compute_sha256,
    load_h2_dataset,
    run_kmeans_baseline,
)
from src.clustering.features import LEAKAGE_COLUMNS, MODEL_FEATURE_COLUMNS


DEFAULT_SAMPLE_SIZE: Final[int] = 3_000
DEFAULT_SAMPLE_SEED: Final[int] = 42
DEFAULT_DENSE_BUDGET_BYTES: Final[int] = 64 * 1024 * 1024
CATEGORY_COLUMNS: Final[tuple[str, ...]] = (
    "MarketSegment", "DistributionChannel", "Meal", "DepositType",
    "IsRepeatedGuest", "ReservedRoomType", "CustomerType",
)
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "LeadTime", "TotalNights", "PartySize", "WeekendShare",
    "PreviousCancellations", "PreviousBookingsNotCanceled",
)
PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "cluster_id", "count", "percentage",
    *(name for column in NUMERIC_COLUMNS for name in
      (f"{column}_median", f"{column}_q25", f"{column}_q75")),
    "zero_person_count", "zero_night_count",
    *(name for column in CATEGORY_COLUMNS for name in
      (f"{column}_top1", f"{column}_top1_percentage",
       f"{column}_top2", f"{column}_top2_percentage")),
)
CATEGORY_PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "cluster_id", "feature", "category", "count", "cluster_percentage",
    "overall_percentage", "lift",
)


@dataclass
class EvaluationResult:
    """Small, serializable summaries of an existing fitted baseline."""

    metrics: dict[str, Any]
    metadata: dict[str, Any]
    profiles: pd.DataFrame
    category_profiles: pd.DataFrame


def validate_labels(labels: np.ndarray, n_rows: int) -> np.ndarray:
    """Require one contiguous, nonempty, zero-based cluster ID per row."""
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError("Cluster labels must have exactly one value per row.")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Cluster labels must be integers.")
    unique = np.unique(values)
    if len(unique) < 2 or len(unique) >= n_rows:
        raise ValueError("Internal metrics require 2 through n_rows - 1 clusters.")
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("Cluster IDs must be contiguous and start at zero.")
    return values


def stratified_sample_indices(
    labels: np.ndarray, *, max_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> np.ndarray:
    """Select a deterministic, roughly proportional sample without replacement."""
    values = validate_labels(labels, len(labels))
    n_rows = len(values)
    n_clusters = len(np.unique(values))
    if max_samples < n_clusters + 1:
        raise ValueError("Sample size must exceed the number of clusters.")
    sample_size = min(max_samples, n_rows)
    counts = np.bincount(values)
    ideal = sample_size * counts / n_rows
    quota = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while quota.sum() < sample_size:
        available = np.flatnonzero(quota < counts)
        selected = available[np.argmax((ideal - quota)[available])]
        quota[selected] += 1
    while quota.sum() > sample_size:
        available = np.flatnonzero(quota > 1)
        selected = available[np.argmax((quota - ideal)[available])]
        quota[selected] -= 1
    rng = np.random.default_rng(seed)
    chosen = [
        rng.choice(np.flatnonzero(values == cluster), size=int(amount), replace=False)
        for cluster, amount in enumerate(quota)
    ]
    return np.sort(np.concatenate(chosen))


def compute_internal_metrics(
    matrix: Any,
    labels: np.ndarray,
    *,
    max_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
    dense_budget_bytes: int = DEFAULT_DENSE_BUDGET_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute sampled silhouette and full or sampled Davies-Bouldin.

    Only the bounded silhouette sample is passed to pairwise distance code.
    Davies-Bouldin receives a dense full matrix only when its estimated array
    size is within the configured budget; otherwise it uses the same sample.
    """
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError("Feature matrix must be two-dimensional.")
    n_rows, n_features = matrix.shape
    values = validate_labels(labels, n_rows)
    if max_samples < 1 or max_samples > DEFAULT_SAMPLE_SIZE or dense_budget_bytes < 1:
        raise ValueError("Sample size must be between 1 and 3000, and dense budget positive.")
    if sparse.issparse(matrix):
        finite = np.isfinite(matrix.data).all()
    else:
        finite = np.isfinite(matrix).all()
    if not finite:
        raise ValueError("Feature matrix contains NaN or infinity.")

    indices = stratified_sample_indices(values, max_samples=max_samples, seed=seed)
    sampled_matrix = matrix[indices]
    sampled_labels = values[indices]
    sampled_counts = np.bincount(sampled_labels, minlength=len(np.unique(values)))
    if np.count_nonzero(sampled_counts) < 2 or len(indices) <= np.count_nonzero(sampled_counts):
        raise ValueError("Silhouette sample needs at least two represented clusters and an extra row.")

    silhouette_start = time.perf_counter()
    silhouette = float(silhouette_score(sampled_matrix, sampled_labels))
    silhouette_seconds = time.perf_counter() - silhouette_start

    dense_estimate = int(n_rows * n_features * np.dtype(matrix.dtype).itemsize)
    full_dbi = dense_estimate <= dense_budget_bytes
    if full_dbi:
        dbi_matrix, dbi_labels = matrix, values
    else:
        max_dense_rows = dense_budget_bytes // (n_features * np.dtype(matrix.dtype).itemsize)
        if max_dense_rows <= len(np.unique(values)):
            raise ValueError("Dense budget is too small for a valid Davies-Bouldin sample.")
        dbi_indices = stratified_sample_indices(
            values, max_samples=min(len(indices), max_dense_rows), seed=seed
        )
        dbi_matrix, dbi_labels = matrix[dbi_indices], values[dbi_indices]
    if sparse.issparse(dbi_matrix):
        dbi_matrix = dbi_matrix.toarray()
    dbi_start = time.perf_counter()
    dbi = float(davies_bouldin_score(dbi_matrix, dbi_labels))
    dbi_seconds = time.perf_counter() - dbi_start
    if not np.isfinite([silhouette, dbi]).all():
        raise ValueError("Internal evaluation produced a nonfinite metric.")

    metrics = {
        "silhouette_score": silhouette,
        "silhouette_scope": "stratified_sample",
        "silhouette_duration_seconds": float(silhouette_seconds),
        "davies_bouldin_index": dbi,
        "davies_bouldin_scope": "full_data" if full_dbi else "stratified_sample",
        "davies_bouldin_duration_seconds": float(dbi_seconds),
    }
    metadata = {
        "sample_method": "proportional_stratified_without_replacement",
        "sample_seed": seed,
        "sample_max_rows": max_samples,
        "sample_rows": len(indices),
        "sample_cluster_counts": {
            str(i): int(count) for i, count in enumerate(sampled_counts)
        },
        "silhouette_pairwise_distance_bytes_estimate": int(len(indices) ** 2 * 8),
        "full_dense_feature_bytes_estimate": dense_estimate,
        "dense_budget_bytes": dense_budget_bytes,
        "dbi_rows": len(dbi_labels),
    }
    return metrics, metadata


def build_cluster_profiles(
    baseline: KMeansBaselineResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize aligned source rows in original units without mutation."""
    bundle = baseline.feature_bundle
    features = bundle.model_features
    source = bundle.source_data
    flags = bundle.edge_case_flags
    labels = validate_labels(baseline.labels, len(source))
    if not (len(features) == len(flags) == len(labels)):
        raise ValueError("Source, features, flags, and labels are misaligned.")
    if tuple(features.columns) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Unexpected model feature contract for profiling.")
    if set(features.columns) & LEAKAGE_COLUMNS:
        raise ValueError("Leakage field found in model features.")
    # Positional assignment is deliberate: duplicate source index labels are legal.
    frame = features.reset_index(drop=True).copy()
    frame["cluster_id"] = labels
    flag_frame = flags.reset_index(drop=True)
    n_rows = len(frame)
    category_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for cluster in np.unique(labels):
        part = frame.loc[frame["cluster_id"] == cluster]
        count = len(part)
        row: dict[str, Any] = {
            "cluster_id": int(cluster),
            "count": count,
            "percentage": count / n_rows * 100.0,
        }
        for column in NUMERIC_COLUMNS:
            numbers = pd.to_numeric(part[column], errors="coerce")
            row[f"{column}_median"] = float(numbers.median())
            row[f"{column}_q25"] = float(numbers.quantile(0.25))
            row[f"{column}_q75"] = float(numbers.quantile(0.75))
        mask = frame["cluster_id"].eq(cluster).to_numpy()
        row["zero_person_count"] = int(flag_frame.loc[mask, "ZeroPersonBooking"].sum())
        row["zero_night_count"] = int(flag_frame.loc[mask, "ZeroNightStay"].sum())
        for column in CATEGORY_COLUMNS:
            overall = frame[column].fillna("__MISSING__").value_counts()
            selected = part[column].fillna("__MISSING__").value_counts()
            for rank in (1, 2):
                if len(selected) >= rank:
                    category = selected.index[rank - 1]
                    row[f"{column}_top{rank}"] = str(category)
                    row[f"{column}_top{rank}_percentage"] = float(selected.iloc[rank - 1] / count * 100)
                else:
                    row[f"{column}_top{rank}"] = ""
                    row[f"{column}_top{rank}_percentage"] = np.nan
            for category, category_count in selected.items():
                overall_percentage = float(overall[category] / n_rows * 100)
                cluster_percentage = float(category_count / count * 100)
                category_rows.append({
                    "cluster_id": int(cluster), "feature": column,
                    "category": str(category), "count": int(category_count),
                    "cluster_percentage": cluster_percentage,
                    "overall_percentage": overall_percentage,
                    "lift": cluster_percentage / overall_percentage,
                })
        profile_rows.append(row)
    profiles = pd.DataFrame(profile_rows, columns=PROFILE_COLUMNS)
    category_profiles = pd.DataFrame(category_rows, columns=CATEGORY_PROFILE_COLUMNS)
    return profiles, category_profiles


def verify_committed_baseline(
    result: KMeansBaselineResult,
    metrics_path: Path,
    sizes_path: Path,
    metadata_path: Path,
    dataset_sha256: str,
) -> None:
    """Stop if the reproduced fixed baseline differs from committed results."""
    with metrics_path.open(encoding="utf-8") as stream:
        expected_metrics = json.load(stream)
    with metadata_path.open(encoding="utf-8") as stream:
        expected_metadata = json.load(stream)
    expected_sizes = pd.read_csv(sizes_path)
    actual_metrics = result.metrics
    for key in ("dataset_rows", "source_feature_count", "engineered_feature_count",
                "transformed_feature_count", "n_iterations", "kmeans_parameters"):
        if actual_metrics[key] != expected_metrics[key]:
            raise ValueError(f"Baseline mismatch in {key}.")
    if not np.isclose(actual_metrics["inertia"], expected_metrics["inertia"], rtol=1e-9, atol=1e-6):
        raise ValueError("Baseline inertia does not match the committed result.")
    if expected_metadata["dataset_sha256"] != dataset_sha256:
        raise ValueError("Dataset SHA-256 does not match the committed baseline.")
    if not np.array_equal(result.cluster_sizes["cluster_id"], expected_sizes["cluster_id"]):
        raise ValueError("Baseline cluster IDs do not match.")
    if not np.array_equal(result.cluster_sizes["count"], expected_sizes["count"]):
        raise ValueError("Baseline cluster sizes do not match.")


def evaluate_baseline(
    baseline: KMeansBaselineResult, *, max_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
    dense_budget_bytes: int = DEFAULT_DENSE_BUDGET_BYTES,
) -> EvaluationResult:
    """Evaluate the exact in-memory fitted matrix and aligned booking records."""
    metrics, sample_metadata = compute_internal_metrics(
        baseline.transformed_matrix, baseline.labels, max_samples=max_samples,
        seed=seed, dense_budget_bytes=dense_budget_bytes,
    )
    profiles, category_profiles = build_cluster_profiles(baseline)
    metrics.update({
        "dataset_rows": baseline.metrics["dataset_rows"],
        "cluster_count": len(profiles),
        "cluster_sizes": [
            {"cluster_id": int(row.cluster_id), "count": int(row.count),
             "percentage": float(row.percentage)}
            for row in profiles[["cluster_id", "count", "percentage"]].itertuples(index=False)
        ],
    })
    metadata = {
        "baseline_parameters": baseline.metrics["kmeans_parameters"],
        "baseline_inertia": baseline.metrics["inertia"],
        "baseline_transformed_features": baseline.metrics["transformed_feature_count"],
        "dataset_sha256": baseline.metadata["dataset_sha256"],
        "package_versions": baseline.metadata["package_versions"],
        "distance_space": "exact fitted Phase 3 transformed feature matrix",
        "row_alignment": "original positional order; no rows removed",
        "outcome_fields_used": False,
        **sample_metadata,
    }
    return EvaluationResult(metrics, metadata, profiles, category_profiles)


def write_evaluation_outputs(result: EvaluationResult, output_dir: Path) -> dict[str, Path]:
    """Persist only small aggregate outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "evaluation_metrics.json",
        "profiles": output_dir / "cluster_profiles.csv",
        "categories": output_dir / "cluster_category_profiles.csv",
        "metadata": output_dir / "evaluation_metadata.json",
    }
    paths["metrics"].write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metadata"].write_text(json.dumps(result.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result.profiles.to_csv(paths["profiles"], index=False, float_format="%.6f")
    result.category_profiles.to_csv(paths["categories"], index=False, float_format="%.6f")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """Reproduce, verify, evaluate, and summarize the committed baseline."""
    parser = argparse.ArgumentParser(description="Evaluate the fixed Phase 3 K-Means baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    args = parser.parse_args(argv)
    try:
        data = load_h2_dataset(args.dataset)
        sha = compute_sha256(args.dataset)
        baseline = run_kmeans_baseline(data, dataset_sha256=sha, dataset_path=args.dataset.as_posix())
        verify_committed_baseline(
            baseline, args.output_dir / "baseline_metrics.json",
            args.output_dir / "baseline_cluster_sizes.csv",
            args.output_dir / "baseline_metadata.json", sha,
        )
        evaluation = evaluate_baseline(baseline, max_samples=args.sample_size, seed=args.seed)
        paths = write_evaluation_outputs(evaluation, args.output_dir)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"clustering evaluation failed: {error}", file=sys.stderr)
        return 2
    print(f"Baseline verified; silhouette={evaluation.metrics['silhouette_score']:.6f} "
          f"({evaluation.metadata['sample_rows']} sampled), "
          f"Davies-Bouldin={evaluation.metrics['davies_bouldin_index']:.6f} "
          f"({evaluation.metrics['davies_bouldin_scope']}).")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
