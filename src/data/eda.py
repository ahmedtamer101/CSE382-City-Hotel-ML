"""Reproducible, non-mutating City Hotel H2 data audit and EDA figures.

Run from repository root:
    python -m src.data.eda --data data/H2.csv --out outputs/eda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EXPECTED_COLUMNS = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "AssignedRoomType",
    "BookingChanges",
    "DepositType",
    "Agent",
    "Company",
    "DaysInWaitingList",
    "CustomerType",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ReservationStatus",
    "ReservationStatusDate",
]
MISSING_MARKERS = {"null", "none", "n/a", "na", "nan", ""}
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]

def validate_schema(df: pd.DataFrame) -> None:
    """Validate the exact H2 dataset schema."""

    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        missing = [
            col for col in EXPECTED_COLUMNS
            if col not in actual_columns
        ]

        unexpected = [
            col for col in actual_columns
            if col not in EXPECTED_COLUMNS
        ]

        raise ValueError(
            "Invalid H2 dataset schema.\n"
            f"Missing columns: {missing}\n"
            f"Unexpected columns: {unexpected}\n"
            "Column order must match the expected H2 schema."
        )
def audit(df: pd.DataFrame, out: Path) -> None:
    """Save EDA tables without modifying df or inferring row-level identity."""
    out.mkdir(parents=True, exist_ok=True)

    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "unique_non_null": df.nunique(dropna=True),
        "actual_missing": df.isna().sum(),
    })
    profile.to_csv(out / "column_profile.csv", index_label="feature")

    hidden_rows = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        text = df[col].astype("string").str.strip().str.lower()
        count = int(text.isin(MISSING_MARKERS).sum())
        if count:
            hidden_rows.append({
                "feature": col, "text_missing_count": count,
                "text_missing_pct": round(count / len(df) * 100, 3),
            })
    pd.DataFrame(hidden_rows, columns=[
        "feature", "text_missing_count", "text_missing_pct"
    ]).to_csv(out / "text_missing.csv", index=False)

    guest_count = df["Adults"].fillna(0) + df["Children"].fillna(0) + df["Babies"].fillna(0)
    checks = {
        "identical_rows_after_first": int(df.duplicated().sum()),
        "zero_total_guests": int((guest_count == 0).sum()),
        "zero_total_nights": int(((df["StaysInWeekNights"] + df["StaysInWeekendNights"]) == 0).sum()),
        "negative_lead_time": int((df["LeadTime"] < 0).sum()),
        "negative_adr": int((df["ADR"] < 0).sum()),
        "adr_over_1000": int((df["ADR"] > 1000).sum()),
    }
    pd.DataFrame([
        {"check": key, "count": value, "percentage": round(value / len(df) * 100, 3)}
        for key, value in checks.items()
    ]).to_csv(out / "quality_checks.csv", index=False)

    outlier_rows = []
    for col in ["LeadTime", "ADR", "StaysInWeekNights", "StaysInWeekendNights", "DaysInWaitingList"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag = (df[col] < lower) | (df[col] > upper)
        outlier_rows.append({
            "feature": col, "lower": lower, "upper": upper,
            "flagged": int(flag.sum()), "percentage": round(flag.mean() * 100, 3),
        })
    pd.DataFrame(outlier_rows).to_csv(out / "iqr_screening.csv", index=False)

    for col in ["MarketSegment", "DepositType", "CustomerType", "ArrivalDateMonth"]:
        summary = df.groupby(col, dropna=False)["IsCanceled"].agg(
            bookings="size", cancellation_rate="mean"
        )
        summary["cancellation_rate"] *= 100
        if col == "ArrivalDateMonth":
            summary = summary.reindex(MONTHS)
        else:
            summary = summary.sort_values("bookings", ascending=False)
        summary.to_csv(out / f"cancel_by_{col}.csv")

    numeric = df.select_dtypes(include="number")
    numeric.corr().to_csv(out / "pearson_correlations.csv")


def plots(df: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.countplot(data=df, x="IsCanceled", color="#4b8ba5", ax=axes[0])
    axes[0].set_title("Booking cancellation counts")
    axes[0].set_xticks([0, 1], ["Not canceled", "Canceled"])
    sns.histplot(df["LeadTime"], bins=50, color="#4b8ba5", ax=axes[1])
    axes[1].set_title("Lead time (days)")
    sns.histplot(df.loc[df["ADR"].between(0, 500), "ADR"], bins=50,
                 color="#d18464", ax=axes[2])
    axes[2].set_title("ADR (display range 0–500 only)")
    fig.tight_layout()
    fig.savefig(out / "numerical_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flat, ["MarketSegment", "DepositType", "CustomerType", "ArrivalDateMonth"]):
        order = MONTHS if col == "ArrivalDateMonth" else df[col].value_counts().index
        sns.countplot(data=df, y=col, order=order, color="#438f87", ax=ax)
        ax.set_title(f"{col} distribution")
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out / "categorical_distributions.png", dpi=160)
    plt.close(fig)

    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson correlations of numerical columns")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/H2.csv"),
        help="Path to the original H2.csv dataset",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eda"),
        help="Directory for generated EDA outputs",
    )

    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(
            f"Dataset not found: {args.data}. "
            "Place H2.csv inside data/ or specify --data PATH."
        )

    df = pd.read_csv(
        args.data,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )

    validate_schema(df)
    if len(df) != 79330:
        raise ValueError(
            f"Unexpected H2 row count: {len(df)}; expected 79330"
        )
    # Convert Children to numeric for the EDA calculations.
    df["Children"] = pd.to_numeric(
        df["Children"],
        errors="coerce",
    )

    audit(df, args.out)
    plots(df, args.out)

    print("H2 schema validation passed.")
    print(f"Dataset shape: {df.shape}")
    print(f"EDA outputs written to {args.out.resolve()}")

if __name__ == "__main__":
    main()
