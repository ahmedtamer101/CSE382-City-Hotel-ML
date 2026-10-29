"""Deterministic, row-wise feature engineering (no statistics learned -> no leakage).

Each feature only uses columns of the SAME booking that are available at the
prediction point, so it can be computed identically at training and inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_TO_NUM = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

ENGINEERED_FEATURES = {
    "ArrivalMonthNum": "Arrival month as 1-12",
    "ArrivalMonthSin": "Cyclical month encoding (December is next to January)",
    "ArrivalMonthCos": "Cyclical month encoding (December is next to January)",
    "ArrivalDayOfWeek": "Weekday of arrival, derived from year/month/day",
    "TotalNights": "StaysInWeekendNights + StaysInWeekNights",
    "WeekendNightRatio": "Share of booked nights that are weekend nights",
    "TotalGuests": "Adults + Children + Babies",
    "HasKids": "1 if Children + Babies > 0",
    "HasAgent": "1 if booked through a travel agent",
    "HasCompany": "1 if a company is attached (Company itself is 95 % missing)",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    month = out["ArrivalDateMonth"].map(MONTH_TO_NUM)
    out["ArrivalMonthNum"] = month
    out["ArrivalMonthSin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["ArrivalMonthCos"] = np.cos(2 * np.pi * (month - 1) / 12)

    dates = pd.to_datetime(
        pd.DataFrame({"year": out["ArrivalDateYear"], "month": month,
                      "day": out["ArrivalDateDayOfMonth"]}),
        errors="coerce")
    out["ArrivalDayOfWeek"] = dates.dt.day_name()

    weekend = out["StaysInWeekendNights"].fillna(0)
    week = out["StaysInWeekNights"].fillna(0)
    total = weekend + week
    out["TotalNights"] = total
    out["WeekendNightRatio"] = np.where(total > 0, weekend / total.where(total > 0, 1), 0.0)

    kids = out["Children"].fillna(0) + out["Babies"].fillna(0)
    out["TotalGuests"] = out["Adults"].fillna(0) + kids
    out["HasKids"] = (kids > 0).astype(int)

    agent = out["Agent"].astype("object")
    out["HasAgent"] = (agent.notna() & (agent != "None")).astype(int)
    out["HasCompany"] = out["Company"].notna().astype(int)
    return out
