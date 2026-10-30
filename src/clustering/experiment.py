"""Reproducible, untuned K-Means baseline for City Hotel bookings.

The module intentionally performs no cluster-count search, hyperparameter
optimization, internal metric calculation, visualization, or model
serialization. It reuses the clustering-owned Phase 2 feature contract and
preprocessing pipeline, then records a small set of factual baseline outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from src.clustering.features import (
    ClusteringFeatureBundle,
    EXPECTED_SOURCE_COLUMNS,
    LEAKAGE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    build_booking_characteristics,
)
from src.clustering.pipeline import build_clustering_pipeline


EXPECTED_H2_ROWS: Final[int] = 79_330
DEFAULT_DATASET_PATH: Final[Path] = Path("data/H2.csv")
DEFAULT_OUTPUT_DIRECTORY: Final[Path] = Path("outputs/clustering")
REPRODUCTION_COMMAND: Final[str] = (
    "python -m src.clustering.experiment --dataset data/H2.csv "
    "--output-dir outputs/clustering"
)
CLUSTER_SIZE_COLUMNS: Final[tuple[str, ...]] = (
    "cluster_id",
    "count",
    "percentage",
)


@dataclass(frozen=True)
class KMeansBaselineConfig:
    """Fixed Milestone 1 baseline defaults with explicit validation."""

    n_clusters: int = 3
    init: str = "k-means++"
    n_init: int = 10
    random_state: int = 42
    max_iter: int = 300

    def validate(self) -> None:
        """Reject invalid values before sklearn starts fitting."""
        integer_fields = {
            "n_clusters": self.n_clusters,
            "n_init": self.n_init,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer; received {value!r}.")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2 for clustering.")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if self.init not in {"k-means++", "random"}:
            raise ValueError("init must be either 'k-means++' or 'random'.")

    def as_dict(self) -> dict[str, int | str]:
        """Return the explicitly controlled sklearn parameters."""
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "n_init": self.n_init,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
        }


@dataclass
class KMeansBaselineResult:
    """In-memory baseline result plus small serializable summaries."""

    labels: np.ndarray
    cluster_sizes: pd.DataFrame
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    feature_names: tuple[str, ...]
    model: KMeans = field(repr=False)
    transformed_matrix: Any = field(repr=False)
    feature_bundle: ClusteringFeatureBundle = field(repr=False)


def build_kmeans_baseline(
    config: KMeansBaselineConfig | None = None,
) -> KMeans:
    """Construct the exact untuned baseline estimator."""
    config = config or KMeansBaselineConfig()
    config.validate()
    return KMeans(**config.as_dict())


def compute_sha256(path: Path) -> str:
    """Compute a dataset checksum without loading the complete file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_h2_dataset(
    path: Path,
    *,
    expected_rows: int | None = EXPECTED_H2_ROWS,
) -> pd.DataFrame:
    """Load H2 with its documented textual missing markers and exact schema."""
    if not path.is_file():
        raise FileNotFoundError(f"H2 dataset not found: {path}")

    data = pd.read_csv(
        path,
        skipinitialspace=True,
        na_values=["NULL", "NA"],
        low_memory=False,
    )
    if tuple(data.columns) != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            "H2 schema mismatch: expected the exact 31 source columns in "
            "their documented order."
        )
    if expected_rows is not None and len(data) != expected_rows:
        raise ValueError(
            f"H2 row-count mismatch: expected {expected_rows:,}, "
            f"found {len(data):,}."
        )
    return data


def _matrix_statistics(matrix) -> dict[str, int | float | str | bool]:
    """Describe matrix storage without converting a sparse matrix to dense."""
    rows, columns = matrix.shape
    item_size = np.dtype(matrix.dtype).itemsize
    dense_equivalent_bytes = int(rows * columns * item_size)

    if sparse.issparse(matrix):
        csr = matrix.tocsr(copy=False)
        stored_values = csr.data
        stored_bytes = int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
        nonzero_values = int(csr.nnz)
        matrix_format = csr.getformat()
    else:
        array = np.asarray(matrix)
        stored_values = array
        stored_bytes = int(array.nbytes)
        nonzero_values = int(np.count_nonzero(array))
        matrix_format = "dense"

    if not np.isfinite(stored_values).all():
        raise ValueError("Transformed feature matrix contains NaN or infinity.")

    total_cells = rows * columns
    return {
        "format": matrix_format,
        "is_sparse": bool(sparse.issparse(matrix)),
        "rows": int(rows),
        "columns": int(columns),
        "nonzero_values": nonzero_values,
        "density": float(nonzero_values / total_cells) if total_cells else 0.0,
        "stored_bytes": stored_bytes,
        "dense_equivalent_bytes": dense_equivalent_bytes,
    }


def _package_versions() -> dict[str, str]:
    """Return the environment versions relevant to numerical reproducibility."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_kmeans_baseline(
    source_data: pd.DataFrame,
    config: KMeansBaselineConfig | None = None,
    *,
    dataset_sha256: str | None = None,
    dataset_path: str | None = None,
) -> KMeansBaselineResult:
    """Fit preprocessing and the fixed K-Means baseline on every input row."""
    config = config or KMeansBaselineConfig()
    config.validate()
    if len(source_data) < config.n_clusters:
        raise ValueError(
            "n_clusters cannot exceed the number of available observations: "
            f"n_clusters={config.n_clusters}, observations={len(source_data)}."
        )

    total_start = time.perf_counter()
    bundle = build_booking_characteristics(source_data)
    model_features = bundle.model_features
    if tuple(model_features.columns) != MODEL_FEATURE_COLUMNS:
        raise AssertionError("Unexpected clustering feature contract.")
    if set(model_features.columns).intersection(LEAKAGE_COLUMNS):
        raise AssertionError("Leakage columns entered the K-Means input.")

    preprocessing = build_clustering_pipeline()
    preprocessing_start = time.perf_counter()
    transformed = preprocessing.fit_transform(model_features)
    preprocessing_duration = time.perf_counter() - preprocessing_start
    feature_names = tuple(
        str(name)
        for name in preprocessing.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )
    matrix_statistics = _matrix_statistics(transformed)

    model = build_kmeans_baseline(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit_start = time.perf_counter()
        model.fit(transformed)
        fit_duration = time.perf_counter() - fit_start

    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    labels = np.asarray(model.labels_, dtype=np.int64).copy()
    if labels.shape != (len(source_data),):
        raise AssertionError("K-Means returned an unexpected number of labels.")
    if labels.size and (labels.min() < 0 or labels.max() >= config.n_clusters):
        raise AssertionError("K-Means returned a label outside the configured range.")

    counts = np.bincount(labels, minlength=config.n_clusters)
    cluster_sizes = pd.DataFrame(
        {
            "cluster_id": np.arange(config.n_clusters, dtype=int),
            "count": counts.astype(int),
            "percentage": counts.astype(float) / len(source_data) * 100.0,
        },
        columns=CLUSTER_SIZE_COLUMNS,
    )

    total_duration = time.perf_counter() - total_start
    metrics: dict[str, Any] = {
        "experiment": "untuned_kmeans_baseline",
        "dataset_rows": int(len(source_data)),
        "source_feature_count": int(source_data.shape[1]),
        "engineered_feature_count": int(model_features.shape[1]),
        "transformed_feature_count": int(transformed.shape[1]),
        "kmeans_parameters": config.as_dict(),
        "random_seed": config.random_state,
        "preprocessing_duration_seconds": float(preprocessing_duration),
        "fit_duration_seconds": float(fit_duration),
        "total_duration_seconds": float(total_duration),
        "n_iterations": int(model.n_iter_),
        "inertia": float(model.inertia_),
        "convergence_warnings": convergence_warnings,
    }
    metadata: dict[str, Any] = {
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_sha256,
        "reproduction_command": REPRODUCTION_COMMAND,
        "package_versions": _package_versions(),
        "transformed_feature_names": list(feature_names),
        "matrix": matrix_statistics,
        "row_policy": {
            "all_input_rows_preserved": True,
            "duplicates_removed": False,
            "unusual_bookings_removed": False,
        },
        "leakage_excluded_columns": sorted(LEAKAGE_COLUMNS),
        "model_serialized": False,
    }
    return KMeansBaselineResult(
        labels=labels,
        cluster_sizes=cluster_sizes,
        metrics=metrics,
        metadata=metadata,
        feature_names=feature_names,
        model=model,
        transformed_matrix=transformed,
        feature_bundle=bundle,
    )


def write_baseline_outputs(
    result: KMeansBaselineResult,
    output_directory: Path,
) -> dict[str, Path]:
    """Write only small tables and metadata; never serialize the fitted model."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_directory / "baseline_metrics.json",
        "cluster_sizes": output_directory / "baseline_cluster_sizes.csv",
        "metadata": output_directory / "baseline_metadata.json",
    }
    paths["metrics"].write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result.cluster_sizes.to_csv(
        paths["cluster_sizes"],
        index=False,
        float_format="%.6f",
    )
    paths["metadata"].write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone clustering-baseline CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed, untuned K-Means baseline using the committed "
            "clustering feature and preprocessing contract."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--init", choices=("k-means++", "random"), default="k-means++")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment, write summaries, and report failures clearly."""
    args = build_parser().parse_args(argv)
    config = KMeansBaselineConfig(
        n_clusters=args.n_clusters,
        init=args.init,
        n_init=args.n_init,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    try:
        data = load_h2_dataset(args.dataset)
        result = run_kmeans_baseline(
            data,
            config,
            dataset_sha256=compute_sha256(args.dataset),
            dataset_path=args.dataset.as_posix(),
        )
        paths = write_baseline_outputs(result, args.output_dir)
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
        print(f"clustering baseline failed: {error}", file=sys.stderr)
        return 2

    print(
        "K-Means baseline complete: "
        f"rows={result.metrics['dataset_rows']}, "
        f"features={result.metrics['transformed_feature_count']}, "
        f"inertia={result.metrics['inertia']:.6f}, "
        f"iterations={result.metrics['n_iterations']}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
