"""Load the official task splits written by `python -m src.preprocessing.run`."""
from __future__ import annotations

import os

import pandas as pd

from ..features import get_feature_set, feature_columns

DEFAULT_PROCESSED = os.path.join("data", "processed")


def load_task_split(task: str, split: str, processed_dir: str = DEFAULT_PROCESSED,
                    with_groups: bool = False):
    """Return (X, y) for task in {'regression','classification'} and split in {'train','val','test'}.

    For 'clustering' returns (X, None) with all cleaned rows (split ignored).
    with_groups=True also returns GroupId — pass it as `groups=` to GroupKFold /
    StratifiedGroupKFold so duplicated bookings never cross CV folds.
    """
    fs = get_feature_set(task)
    if task == "clustering":
        path = os.path.join(processed_dir, "clustering", "clustering_all.csv")
    else:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test'")
        path = os.path.join(processed_dir, task, f"{split}.csv")
    df = pd.read_csv(path, keep_default_na=True)
    X = df[feature_columns(task)]
    y = df[fs["target"]] if fs["target"] else None
    if with_groups:
        return X, y, df["GroupId"]
    return X, y
