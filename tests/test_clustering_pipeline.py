"""Tests for clustering preprocessing without fitting a clustering model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import sparse

from src.clustering.features import LEAKAGE_COLUMNS, build_booking_characteristics
from src.clustering.pipeline import ArrivalMonthCyclicEncoder, build_clustering_pipeline
from tests.test_clustering_features import make_h2_fixture


def _finite_values(matrix) -> np.ndarray:
    return matrix.data if sparse.issparse(matrix) else np.asarray(matrix)


def test_missing_numeric_and_categorical_values_are_imputed() -> None:
    data = make_h2_fixture()
    data.loc[0, "Children"] = np.nan
    data.loc[0, "Meal"] = np.nan
    features = build_booking_characteristics(data).model_features

    transformed = build_clustering_pipeline().fit_transform(features)

    assert np.isfinite(_finite_values(transformed)).all()


def test_unknown_category_is_ignored_without_changing_dimension() -> None:
    train = build_booking_characteristics(make_h2_fixture()).model_features
    pipeline = build_clustering_pipeline().fit(train)
    transformed_train = pipeline.transform(train)

    unknown_data = make_h2_fixture().iloc[[0]].copy()
    unknown_data.loc[:, "Meal"] = "NEW_MEAL"
    unknown = build_booking_characteristics(unknown_data).model_features
    transformed_unknown = pipeline.transform(unknown)

    assert transformed_unknown.shape[1] == transformed_train.shape[1]
    assert np.isfinite(_finite_values(transformed_unknown)).all()


def test_month_cyclic_encoding_wraps_and_marks_missing_values() -> None:
    months = pd.DataFrame(
        {"ArrivalDateMonth": ["January", "December", np.nan]}
    )

    encoded = ArrivalMonthCyclicEncoder().fit_transform(months)

    assert_allclose(encoded[0], [0.0, 1.0, 0.0], atol=1e-12)
    assert_allclose(
        encoded[1],
        [np.sin(11 * np.pi / 6), np.cos(11 * np.pi / 6), 0.0],
        atol=1e-12,
    )
    assert_allclose(encoded[2], [0.0, 0.0, 1.0], atol=1e-12)


def test_transformations_are_deterministic() -> None:
    features = build_booking_characteristics(make_h2_fixture()).model_features

    first = build_clustering_pipeline().fit_transform(features)
    second = build_clustering_pipeline().fit_transform(features)

    assert_allclose(first.toarray(), second.toarray())


def test_output_dimension_matches_reported_feature_names() -> None:
    features = build_booking_characteristics(make_h2_fixture()).model_features
    pipeline = build_clustering_pipeline()

    transformed = pipeline.fit_transform(features)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    assert sparse.issparse(transformed)
    assert transformed.shape == (len(features), len(names))
    # Six numeric outputs, three month outputs, and 17 one-hot columns
    # learned from this deliberately tiny two-row fixture.
    assert len(names) == 26


def test_leakage_values_cannot_change_model_matrix() -> None:
    original = make_h2_fixture()
    altered = original.copy(deep=True)
    for column in LEAKAGE_COLUMNS:
        if pd.api.types.is_numeric_dtype(altered[column]):
            altered[column] = altered[column].fillna(0) + 9999
        else:
            altered[column] = "LEAKAGE_SENTINEL"

    original_features = build_booking_characteristics(original).model_features
    altered_features = build_booking_characteristics(altered).model_features
    pipeline = build_clustering_pipeline()
    original_matrix = pipeline.fit_transform(original_features)
    altered_matrix = pipeline.transform(altered_features)

    assert_allclose(original_matrix.toarray(), altered_matrix.toarray())
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    assert not any(column in name for column in LEAKAGE_COLUMNS for name in names)


def test_zero_person_and_zero_night_rows_transform_to_finite_values() -> None:
    features = build_booking_characteristics(make_h2_fixture()).model_features

    transformed = build_clustering_pipeline().fit_transform(features)

    assert np.isfinite(_finite_values(transformed)).all()
