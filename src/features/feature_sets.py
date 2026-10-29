"""Task-specific feature contracts (the hand-off to Members 3, 4 and 5).

Prediction point for both supervised tasks: AT BOOKING CONFIRMATION — the
booking has been entered in the PMS (and left the waiting list, if any), but the
guest has not arrived and the final outcome (stay / cancel / no-show) is unknown.
Only information available at that moment may be used as a predictor.

Column groups and how the preprocessor treats them:
  numeric      -> median impute -> StandardScaler
  skewed       -> median impute -> log1p -> StandardScaler   (right-skewed counts / days)
  binary       -> most-frequent impute -> passed through as 0/1
  categorical  -> constant "Unknown" impute -> OneHotEncoder (rare -> "infrequent")
"""
from __future__ import annotations

TASKS = ("regression", "classification", "clustering")

# Reasons are written once and reused in the contract JSON and the leakage audit.
REASONS = {
    "ReservationStatus": "DIRECT LEAK: final status (Canceled / No-Show / Check-Out) is the outcome itself.",
    "ReservationStatusDate": "DIRECT LEAK: date the final status was set — only exists after cancellation or check-out.",
    "IsCanceled": "Future outcome: unknown at booking time (it is the classification target).",
    "ADR_target": "Target of the regression task.",
    "AssignedRoomType": "Post-booking: decided by the hotel at/near check-in; differs from ReservedRoomType mostly for guests who actually arrived.",
    "BookingChanges": "Post-booking: counts amendments made until check-in OR cancellation, so it grows with the booking's future.",
    "RequiredCarParkingSpaces": "Post-booking / indirect leak: parking is typically requested or recorded at arrival, so it is almost always 0 for cancelled bookings.",
    "Country": "Indirect leak for cancellation: country is often only confirmed at check-in, so its value depends on whether the guest arrived.",
    "Company": "95.35 % missing; replaced by the HasCompany flag.",
    "ArrivalDateMonth": "Replaced by ArrivalMonthSin / ArrivalMonthCos (cyclical encoding).",
    "ArrivalDateYear_clf": "Year is a calendar trend that will not repeat for future bookings; excluded to keep the classifier time-agnostic.",
    "DepositType_clust": "Excluded from clustering: 'Non Refund' is ~99.8 % cancelled, so clusters would be built around a near-outcome proxy.",
    "Agent_clust": "High cardinality ID (224 values) dominates distance metrics; HasAgent is used instead.",
    "History_clust": "Previous-booking counts are ~0 for almost everyone; IsRepeatedGuest carries the same information for segmentation.",
}

_SHARED_SKEWED = ["LeadTime", "DaysInWaitingList", "PreviousCancellations",
                  "PreviousBookingsNotCanceled", "StaysInWeekendNights",
                  "StaysInWeekNights", "TotalNights"]
_SHARED_NUMERIC = ["ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "Adults",
                   "Children", "Babies", "TotalGuests", "TotalOfSpecialRequests",
                   "WeekendNightRatio", "ArrivalMonthSin", "ArrivalMonthCos"]
_SHARED_BINARY = ["IsRepeatedGuest", "HasKids", "HasAgent", "HasCompany"]
_SHARED_CATEGORICAL = ["Meal", "MarketSegment", "DistributionChannel",
                       "ReservedRoomType", "DepositType", "CustomerType", "Agent",
                       "ArrivalDayOfWeek"]

FEATURE_SETS = {
    "regression": {
        "target": "ADR",
        "row_filter": "TotalNights >= regression_min_nights (ADR undefined for 0-night stays)",
        "numeric": _SHARED_NUMERIC + ["ArrivalDateYear"],
        "skewed": list(_SHARED_SKEWED),
        "binary": list(_SHARED_BINARY),
        "categorical": _SHARED_CATEGORICAL + ["Country"],
        "excluded": {
            "ReservationStatus": REASONS["ReservationStatus"],
            "ReservationStatusDate": REASONS["ReservationStatusDate"],
            "IsCanceled": REASONS["IsCanceled"],
            "AssignedRoomType": REASONS["AssignedRoomType"],
            "BookingChanges": REASONS["BookingChanges"],
            "RequiredCarParkingSpaces": REASONS["RequiredCarParkingSpaces"],
            "Company": REASONS["Company"],
            "ArrivalDateMonth": REASONS["ArrivalDateMonth"],
        },
    },
    "classification": {
        "target": "IsCanceled",
        "row_filter": None,
        "numeric": _SHARED_NUMERIC + ["ADR"],
        "skewed": list(_SHARED_SKEWED),
        "binary": list(_SHARED_BINARY),
        "categorical": list(_SHARED_CATEGORICAL),
        "excluded": {
            "ReservationStatus": REASONS["ReservationStatus"],
            "ReservationStatusDate": REASONS["ReservationStatusDate"],
            "AssignedRoomType": REASONS["AssignedRoomType"],
            "BookingChanges": REASONS["BookingChanges"],
            "RequiredCarParkingSpaces": REASONS["RequiredCarParkingSpaces"],
            "Country": REASONS["Country"],
            "Company": REASONS["Company"],
            "ArrivalDateMonth": REASONS["ArrivalDateMonth"],
            "ArrivalDateYear": REASONS["ArrivalDateYear_clf"],
        },
    },
    "clustering": {
        "target": None,
        "row_filter": None,
        "numeric": ["TotalGuests", "TotalOfSpecialRequests", "WeekendNightRatio",
                    "ArrivalMonthSin", "ArrivalMonthCos", "ADR"],
        "skewed": ["LeadTime", "TotalNights", "DaysInWaitingList"],
        "binary": ["IsRepeatedGuest", "HasKids", "HasAgent", "HasCompany"],
        "categorical": ["MarketSegment", "CustomerType", "Meal", "DistributionChannel"],
        "excluded": {
            "IsCanceled": "Outcome label — clustering must stay unsupervised.",
            "ReservationStatus": REASONS["ReservationStatus"],
            "ReservationStatusDate": REASONS["ReservationStatusDate"],
            "DepositType": REASONS["DepositType_clust"],
            "RequiredCarParkingSpaces": REASONS["RequiredCarParkingSpaces"],
            "AssignedRoomType": REASONS["AssignedRoomType"],
            "BookingChanges": REASONS["BookingChanges"],
            "Agent": REASONS["Agent_clust"],
            "Company": REASONS["Company"],
            "Country": "166 categories make distances meaningless; not needed to describe stay behaviour.",
            "PreviousCancellations": REASONS["History_clust"],
            "PreviousBookingsNotCanceled": REASONS["History_clust"],
        },
    },
}


def get_feature_set(task: str) -> dict:
    if task not in FEATURE_SETS:
        raise ValueError(f"Unknown task '{task}'. Choose one of {TASKS}.")
    return FEATURE_SETS[task]


def feature_columns(task: str) -> list[str]:
    fs = get_feature_set(task)
    return fs["numeric"] + fs["skewed"] + fs["binary"] + fs["categorical"]


def contract_as_dict() -> dict:
    out = {}
    for task, fs in FEATURE_SETS.items():
        out[task] = {
            "target": fs["target"],
            "prediction_point": ("at booking confirmation (before arrival, outcome unknown)"
                                 if fs["target"] else "descriptive booking attributes, no outcome"),
            "row_filter": fs["row_filter"],
            "input_columns": feature_columns(task),
            "column_groups": {k: fs[k] for k in ("numeric", "skewed", "binary", "categorical")},
            "excluded_with_reason": fs["excluded"],
        }
    return out
