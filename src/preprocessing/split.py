"""Reproducible, leakage-safe train / validation / test split.

Why grouped? H2 has no booking ID and 32.65 % of rows are exact duplicates
(mostly identical group bookings). They are kept because they are real bookings,
but identical rows are forced into the SAME split; otherwise a model could be
"tested" on a copy of a training row and look better than it really is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ReservationStatusDate is excluded from the key so that bookings identical in
# every predictor but different only in the post-outcome date stay together too.
_GROUP_KEY_EXCLUDE = {"ReservationStatusDate"}


def assign_groups(df: pd.DataFrame, source_columns: list[str]) -> pd.Series:
    key_cols = [c for c in source_columns if c in df.columns and c not in _GROUP_KEY_EXCLUDE]
    hashed = pd.util.hash_pandas_object(df[key_cols].astype(str), index=False)
    codes, _ = pd.factorize(hashed)
    return pd.Series(codes, index=df.index, name="GroupId")


def make_splits(df: pd.DataFrame, groups: pd.Series, stratify_on: str = "IsCanceled",
                test_folds: int = 5, val_folds: int = 4, seed: int = 42) -> pd.Series:
    """Return a Series with values 'train' / 'val' / 'test' (~60/20/20)."""
    y = df[stratify_on].to_numpy()
    idx = np.arange(len(df))

    outer = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(idx, y, groups))

    inner = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
    tr_pos, val_pos = next(inner.split(rest_idx, y[rest_idx], groups.to_numpy()[rest_idx]))

    split = np.empty(len(df), dtype=object)
    split[test_idx] = "test"
    split[rest_idx[tr_pos]] = "train"
    split[rest_idx[val_pos]] = "val"
    return pd.Series(split, index=df.index, name="Split")


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Split")
    out = pd.DataFrame({
        "rows": g.size(),
        "share_%": (g.size() / len(df) * 100).round(2),
        "cancel_rate_%": (g["IsCanceled"].mean() * 100).round(2),
        "ADR_mean": g["ADR"].mean().round(2),
        "unique_groups": g["GroupId"].nunique(),
    })
    return out.reindex(["train", "val", "test"])
