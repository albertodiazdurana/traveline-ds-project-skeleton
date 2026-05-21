"""Data ingress with schema validation as the first defensive boundary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "age",
    "destination",
    "booking_channel",
    "budget",
    "days_since_last_booking",
    "booking_month",
    "device_type",
    "loyalty_tier",
    "prior_complaints",
    "party_size",
    "rebooked",
)
TARGET_COLUMN = "rebooked"


def load_bookings(path: Path | str) -> pd.DataFrame:
    """Load the bookings CSV and validate its schema.

    Raises ValueError if columns are missing, extra columns are present,
    or the target column contains null values.
    """
    df = pd.read_csv(path)
    actual = set(df.columns)
    expected = set(REQUIRED_COLUMNS)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected extra columns: {sorted(extra)}")
    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains null values")
    return df[list(REQUIRED_COLUMNS)]
