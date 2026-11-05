"""Fitted preprocessing pipeline for the clustering feature representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.clustering.features import (
    CATEGORICAL_FEATURES,
    CYCLIC_FEATURES,
    LINEAR_NUMERIC_FEATURES,
    LOG_NUMERIC_FEATURES,
)


MONTH_TO_NUMBER: Final = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


@dataclass(frozen=True)
class ClusteringPreprocessingConfig:
    """Configuration for deterministic, model-independent preprocessing."""

    numeric_imputation_strategy: str = "median"
    categorical_missing_value: str = "__MISSING__"
    sparse_output: bool = True


class NonNegativeLog1pTransformer(TransformerMixin, BaseEstimator):
    """Apply log(1 + x) and reject values outside its intended domain."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn API uses X
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("Log1p input must be a two-dimensional matrix.")
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):  # noqa: N803 - sklearn API uses X
        array = np.asarray(X, dtype=float)
        if np.isnan(array).any() or np.isinf(array).any():
            raise ValueError("Log1p input must be finite after imputation.")
        if (array < 0).any():
            raise ValueError("Log1p features must be nonnegative.")
        return np.log1p(array)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        return np.asarray(input_features, dtype=object)


class ArrivalMonthCyclicEncoder(TransformerMixin, BaseEstimator):
    """Encode month adjacency with sine/cosine plus a missingness indicator."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn API uses X
        values = self._one_column(X)
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray(
            [getattr(values, "name", None) or "ArrivalDateMonth"], dtype=object
        )
        return self

    def transform(self, X):  # noqa: N803 - sklearn API uses X
        values = self._one_column(X)
        missing = pd.isna(values)
        normalized = pd.Series(values, dtype="object").where(~missing, None)
        normalized = normalized.map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        month_numbers = normalized.map(MONTH_TO_NUMBER)
        unknown = normalized.notna() & month_numbers.isna()
        if unknown.any():
            bad = sorted(normalized.loc[unknown].astype(str).unique())
            raise ValueError(f"Unknown ArrivalDateMonth values: {bad}")

        angle = 2.0 * np.pi * (month_numbers.fillna(1.0).to_numpy() - 1.0) / 12.0
        encoded = np.column_stack(
            (np.sin(angle), np.cos(angle), missing.astype(float))
        )
        encoded[missing, :2] = 0.0
        return encoded.astype(float, copy=False)

    def get_feature_names_out(self, input_features=None):
        source = (
            str(input_features[0])
            if input_features is not None
            else "ArrivalDateMonth"
        )
        return np.asarray(
            [f"{source}_sin", f"{source}_cos", f"{source}_missing"],
            dtype=object,
        )

    @staticmethod
    def _one_column(X):  # noqa: N803 - sklearn API uses X
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("Month encoder expects exactly one column.")
            return X.iloc[:, 0]
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            return pd.Series(array)
        if array.ndim == 2 and array.shape[1] == 1:
            return pd.Series(array[:, 0])
        raise ValueError("Month encoder expects exactly one column.")


def build_clustering_preprocessor(
    config: ClusteringPreprocessingConfig | None = None,
) -> ColumnTransformer:
    """Build the fitted imputation, encoding, and scaling transformer.

    Feature construction in ``features.py`` is fixed and deterministic. The
    imputers, one-hot vocabulary, and scalers created here learn only when this
    returned transformer is fitted.
    """
    config = config or ClusteringPreprocessingConfig()

    log_numeric = Pipeline(
        steps=(
            (
                "imputer",
                SimpleImputer(strategy=config.numeric_imputation_strategy),
            ),
            ("log1p", NonNegativeLog1pTransformer()),
            ("scaler", StandardScaler()),
        )
    )
    linear_numeric = Pipeline(
        steps=(
            (
                "imputer",
                SimpleImputer(strategy=config.numeric_imputation_strategy),
            ),
            ("scaler", StandardScaler()),
        )
    )
    categorical = Pipeline(
        steps=(
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=config.categorical_missing_value,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=config.sparse_output,
                    dtype=np.float64,
                ),
            ),
        )
    )

    return ColumnTransformer(
        transformers=(
            ("log_numeric", log_numeric, list(LOG_NUMERIC_FEATURES)),
            ("linear_numeric", linear_numeric, list(LINEAR_NUMERIC_FEATURES)),
            ("arrival_month", ArrivalMonthCyclicEncoder(), list(CYCLIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ),
        remainder="drop",
        sparse_threshold=1.0 if config.sparse_output else 0.0,
        verbose_feature_names_out=True,
    )


def build_clustering_pipeline(
    config: ClusteringPreprocessingConfig | None = None,
) -> Pipeline:
    """Return the preprocessing-only pipeline; no clustering model is added."""
    return Pipeline(
        steps=(("preprocessor", build_clustering_preprocessor(config)),)
    )
