"""Small synthetic H2-like data so tests run without the real CSV (same schema + quirks)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def make_fake_h2(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    canceled = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "IsCanceled": canceled,
        "LeadTime": rng.integers(0, 400, n),
        "ArrivalDateYear": rng.choice([2015, 2016, 2017], n),
        "ArrivalDateMonth": rng.choice(MONTHS, n),
        "ArrivalDateWeekNumber": rng.integers(1, 54, n),
        "ArrivalDateDayOfMonth": rng.integers(1, 29, n),
        "StaysInWeekendNights": rng.integers(0, 3, n),
        "StaysInWeekNights": rng.integers(0, 5, n),
        "Adults": rng.choice([0, 1, 2, 3], n, p=[0.02, 0.2, 0.7, 0.08]),
        "Children": rng.choice([0.0, 1.0, np.nan], n, p=[0.9, 0.09, 0.01]),
        "Babies": rng.choice([0, 1], n, p=[0.98, 0.02]),
        "Meal": rng.choice(["BB       ", "HB       ", "SC       ", "FB       "], n),
        "Country": rng.choice(["PRT", "FRA", "DEU", "GBR", None, "ZZZ"], n, p=[0.4, 0.2, 0.2, 0.15, 0.02, 0.03]),
        "MarketSegment": rng.choice(["Online TA", "Offline TA/TO", "Groups", "Direct", "Undefined"], n,
                                    p=[0.5, 0.2, 0.15, 0.14, 0.01]),
        "DistributionChannel": rng.choice(["TA/TO", "Direct", "Corporate"], n),
        "IsRepeatedGuest": rng.choice([0, 1], n, p=[0.97, 0.03]),
        "PreviousCancellations": rng.choice([0, 1, 5], n, p=[0.9, 0.08, 0.02]),
        "PreviousBookingsNotCanceled": rng.choice([0, 1, 10], n, p=[0.9, 0.08, 0.02]),
        "ReservedRoomType": rng.choice(["A               ", "D               ", "E               "], n),
        "AssignedRoomType": rng.choice(["A               ", "D               ", "F               "], n),
        "BookingChanges": rng.choice([0, 1, 2], n),
        "DepositType": rng.choice(["No Deposit     ", "Non Refund     "], n, p=[0.85, 0.15]),
        "Agent": rng.choice(["          9", "          1", "       NULL", "         14"], n),
        "Company": rng.choice(["       NULL", "         40"], n, p=[0.95, 0.05]),
        "DaysInWaitingList": rng.choice([0, 0, 0, 30], n),
        "CustomerType": rng.choice(["Transient", "Transient-Party", "Contract", "Group"], n),
        "ADR": np.round(rng.gamma(5, 20, n), 2),
        "RequiredCarParkingSpaces": np.where(canceled == 1, 0, rng.choice([0, 1], n, p=[0.9, 0.1])),
        "TotalOfSpecialRequests": rng.integers(0, 4, n),
        "ReservationStatus": np.where(canceled == 1, "Canceled", "Check-Out"),
        "ReservationStatusDate": "2016-05-01",
    })
    df.loc[0, "ADR"] = 5400.0                              # extreme error value
    dup = df.sample(n // 10, random_state=seed)            # exact duplicate bookings
    return pd.concat([df, dup], ignore_index=True)


@pytest.fixture
def fake_h2():
    return make_fake_h2()
