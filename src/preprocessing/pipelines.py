"""Reusable scikit-learn preprocessors — the main interface for Members 3, 4 and 5.

Usage (fit ONLY on training data, ideally inside a Pipeline so CV folds stay clean):

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from src.preprocessing import build_preprocessor, load_task_split

    X_train, y_train = load_task_split("regression", "train")
    model = Pipeline([("prep", build_preprocessor("regression")),
                      ("model", LinearRegression())])
    model.fit(X_train, y_train)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ..features import add_features, get_feature_set, feature_columns
from .cleaning import standardize


def _make_ohe(min_frequency):
    kwargs = dict(handle_unknown="infrequent_if_exist", min_frequency=min_frequency)
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(sparse=False, **kwargs)


def _to_text(X):
    """Make categorical inputs uniform text so OneHotEncoder never sees mixed types."""
    df = pd.DataFrame(X).astype("object")
    return df.where(df.notna(), "Unknown").astype(str)


def build_preprocessor(task: str, min_frequency: float = 0.005, scale: bool = True) -> ColumnTransformer:
    """Unfitted ColumnTransformer for `task` ('regression' | 'classification' | 'clustering').

    scale=False keeps numeric values unscaled (only useful for tree models if you
    want readable thresholds; scaling never hurts trees, so the default is True).
    """
    fs = get_feature_set(task)
    scaler = [("scale", StandardScaler())] if scale else []

    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))] + scaler)
    skewed = Pipeline([("impute", SimpleImputer(strategy="median")),
                       ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]
                      + scaler)
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline([
        ("to_text", FunctionTransformer(_to_text, feature_names_out="one-to-one")),
        ("onehot", _make_ohe(min_frequency)),
    ])

    transformers = []
    for name, pipe in (("num", numeric), ("skew", skewed), ("bin", binary), ("cat", categorical)):
        cols = fs[{"num": "numeric", "skew": "skewed", "bin": "binary", "cat": "categorical"}[name]]
        if cols:
            transformers.append((name, pipe, list(cols)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def prepare_inference_frame(records: pd.DataFrame | dict | list, task: str) -> pd.DataFrame:
    """Turn raw booking input (same source labels as H2.csv) into the model input columns.

    Applies the same standardisation + feature engineering as training, but never
    drops rows. Missing optional fields are filled with NaN and imputed by the
    saved preprocessor. Used by the application for new-sample predictions.
    """
    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records).copy()
    needed_raw = ["ArrivalDateMonth", "ArrivalDateYear", "ArrivalDateDayOfMonth",
                  "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children",
                  "Babies", "Agent", "Company"]
    for col in needed_raw:
        if col not in df.columns:
            df[col] = np.nan
    df = add_features(standardize(df))
    cols = feature_columns(task)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[cols]
