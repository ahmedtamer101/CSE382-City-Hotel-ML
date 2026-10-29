"""End-to-end preprocessing run (Member 2).

    python -m src.preprocessing.run --config configs/preprocessing.yaml

Steps: load + verify H2.csv -> clean -> engineer features -> grouped stratified
60/20/20 split -> task CSVs -> fit preprocessors on TRAIN only -> save artifacts,
summary tables and the feature contract.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml

from ..features import add_features, contract_as_dict, feature_columns, get_feature_set
from .cleaning import EXPECTED_COLUMNS, clean_training_data, load_raw
from .pipelines import build_preprocessor
from .split import assign_groups, make_splits, split_summary


def leakage_checks(train: pd.DataFrame) -> pd.DataFrame:
    """Cancellation rate behind suspicious features — computed on TRAIN rows only."""
    checks = {
        "RequiredCarParkingSpaces > 0": train["RequiredCarParkingSpaces"] > 0,
        "AssignedRoomType != ReservedRoomType": train["AssignedRoomType"] != train["ReservedRoomType"],
        "BookingChanges > 0": train["BookingChanges"] > 0,
        "Country == PRT": train["Country"] == "PRT",
        "DepositType == Non Refund": train["DepositType"] == "Non Refund",
        "ReservationStatus == Canceled/No-Show": train["ReservationStatus"].isin(["Canceled", "No-Show"]),
    }
    rows = []
    for name, mask in checks.items():
        rows.append({
            "condition": name,
            "rows_true": int(mask.sum()),
            "cancel_rate_if_true_%": round(train.loc[mask, "IsCanceled"].mean() * 100, 2) if mask.any() else np.nan,
            "cancel_rate_if_false_%": round(train.loc[~mask, "IsCanceled"].mean() * 100, 2) if (~mask).any() else np.nan,
        })
    return pd.DataFrame(rows)


def task_frame(df: pd.DataFrame, task: str, cfg: dict) -> pd.DataFrame:
    fs = get_feature_set(task)
    out = df
    if task == "regression":
        out = out[out["TotalNights"] >= cfg["cleaning"]["regression_min_nights"]]
    cols = feature_columns(task) + ([fs["target"]] if fs["target"] else []) + ["GroupId", "Split"]
    return out[cols].reset_index(drop=True)


def main(config_path: str = "configs/preprocessing.yaml", raw_path: str | None = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    P = cfg["paths"]
    raw_path = raw_path or P["raw"]
    for d in (P["interim"], P["processed"], P["artifacts"], P["outputs"]):
        os.makedirs(d, exist_ok=True)

    # 1. Load and verify
    raw = load_raw(raw_path)
    print(f"[1/6] Loaded {raw_path}: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    if raw.shape[0] != 79330:
        print("      WARNING: official H2 has 79,330 rows — check you loaded City Hotel only.")
    print(f"      Exact duplicate rows: {raw.duplicated().sum():,} (kept, handled by grouped split)")

    # 2. Clean (deterministic rules only)
    clean, log = clean_training_data(raw, adr_max=cfg["cleaning"]["adr_max"],
                                     drop_zero_guests=cfg["cleaning"]["drop_zero_guests"])
    log.to_csv(os.path.join(P["outputs"], "cleaning_log.csv"), index=False)
    print(f"[2/6] Cleaned: {len(clean):,} rows remain\n{log.to_string(index=False)}")

    # 3. Feature engineering (row-wise, no learned statistics)
    data = add_features(clean)
    print(f"[3/6] Engineered features added -> {data.shape[1]} columns")

    # 4. Grouped, stratified split
    data["GroupId"] = assign_groups(data, EXPECTED_COLUMNS)
    data["Split"] = make_splits(data, data["GroupId"], stratify_on=cfg["split"]["stratify_on"],
                                test_folds=cfg["split"]["test_folds"],
                                val_folds=cfg["split"]["val_folds"], seed=cfg["seed"])
    summary = split_summary(data)
    summary.to_csv(os.path.join(P["outputs"], "split_summary.csv"))
    overlap = data.groupby("GroupId")["Split"].nunique().max()
    assert overlap == 1, "A duplicate group was split across train/val/test!"
    print(f"[4/6] Split done (no group crosses splits)\n{summary.to_string()}")
    data.to_csv(os.path.join(P["interim"], "h2_clean_features_split.csv"), index=False)

    train_full = data[data["Split"] == "train"]
    lk = leakage_checks(train_full)
    lk.to_csv(os.path.join(P["outputs"], "leakage_checks_train.csv"), index=False)

    # 5. Task datasets + preprocessors fitted on TRAIN only
    min_freq = cfg["encoding"]["min_frequency"]
    shapes = []
    for task in ("regression", "classification"):
        fs = get_feature_set(task)
        tdir = os.path.join(P["processed"], task)
        os.makedirs(tdir, exist_ok=True)
        tf = task_frame(data, task, cfg)
        parts = {s: tf[tf["Split"] == s].drop(columns="Split") for s in ("train", "val", "test")}
        for s, part in parts.items():
            part.to_csv(os.path.join(tdir, f"{s}.csv"), index=False)

        prep = build_preprocessor(task, min_frequency=min_freq)
        X_train = parts["train"][feature_columns(task)]
        prep.fit(X_train)  # <- fitted on training rows only
        joblib.dump(prep, os.path.join(P["artifacts"], f"{task}_preprocessor.joblib"))
        names = prep.get_feature_names_out()

        for s, part in parts.items():
            Xt = prep.transform(part[feature_columns(task)])
            shapes.append({"task": task, "split": s, "rows": Xt.shape[0],
                           "raw_input_cols": len(feature_columns(task)),
                           "model_features_after_encoding": Xt.shape[1],
                           "NaN_after_transform": int(np.isnan(Xt).sum())})
            if cfg.get("export_transformed", False):
                out = pd.DataFrame(Xt, columns=names)
                out[fs["target"]] = part[fs["target"]].to_numpy()
                out.to_csv(os.path.join(tdir, f"{s}_transformed.csv.gz"), index=False,
                           compression="gzip")

    # Clustering: unsupervised, uses all cleaned rows, never sees IsCanceled.
    cdir = os.path.join(P["processed"], "clustering")
    os.makedirs(cdir, exist_ok=True)
    cf = task_frame(data, "clustering", cfg)
    cf.to_csv(os.path.join(cdir, "clustering_all.csv"), index=False)
    cprep = build_preprocessor("clustering", min_frequency=min_freq).fit(cf[feature_columns("clustering")])
    joblib.dump(cprep, os.path.join(P["artifacts"], "clustering_preprocessor.joblib"))
    Xc = cprep.transform(cf[feature_columns("clustering")])
    shapes.append({"task": "clustering", "split": "all", "rows": Xc.shape[0],
                   "raw_input_cols": len(feature_columns("clustering")),
                   "model_features_after_encoding": Xc.shape[1],
                   "NaN_after_transform": int(np.isnan(Xc).sum())})
    if cfg.get("export_transformed", False):
        pd.DataFrame(Xc, columns=cprep.get_feature_names_out()).to_csv(
            os.path.join(cdir, "clustering_all_transformed.csv.gz"), index=False, compression="gzip")

    shapes = pd.DataFrame(shapes)
    shapes.to_csv(os.path.join(P["outputs"], "task_matrix_shapes.csv"), index=False)
    print(f"[5/6] Task datasets and fitted preprocessors saved\n{shapes.to_string(index=False)}")

    # 6. Contract
    with open(os.path.join(P["outputs"], "feature_contract.json"), "w") as f:
        json.dump({"seed": cfg["seed"], "split": cfg["split"], "tasks": contract_as_dict()}, f, indent=2)
    print(f"[6/6] Feature contract written to {P['outputs']}/feature_contract.json")
    print("\nLeakage checks (train split only):\n" + lk.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CSE382 H2 preprocessing")
    ap.add_argument("--config", default="configs/preprocessing.yaml")
    ap.add_argument("--raw", default=None, help="override path to H2.csv")
    a = ap.parse_args()
    main(a.config, a.raw)
