"""Unit tests for load_bookings.

The loader is the data-ingress boundary: it's where untrusted CSV becomes
typed in-memory DataFrame. Tests cover the happy path plus the three
ValueError branches (missing column, extra column, null target).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rebooking.data.loader import REQUIRED_COLUMNS, load_bookings


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Tiny but schema-complete DataFrame matching REQUIRED_COLUMNS."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "age": [30, 45, 60],
            "destination": ["BCN", "PAR", "ROM"],
            "booking_channel": ["web", "app", "agent"],
            "budget": [1000.0, 2000.0, 1500.0],
            "days_since_last_booking": [100, 500, 1200],
            "booking_month": [3, 7, 11],
            "device_type": ["mobile", "desktop", "tablet"],
            "loyalty_tier": ["silver", "gold", "none"],
            "prior_complaints": [0, 1, 0],
            "party_size": [2, 4, 3],
            "rebooked": [0, 1, 0],
        }
    )


def _write_csv(df: pd.DataFrame, tmp_path: Path, name: str = "bookings.csv") -> Path:
    """Write df to tmp_path/name and return the path."""
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_happy_path_returns_dataframe(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """A well-formed CSV loads, shape matches input, columns in canonical order."""
    path = _write_csv(valid_df, tmp_path)
    df = load_bookings(path)
    assert df.shape == (3, len(REQUIRED_COLUMNS))
    assert tuple(df.columns) == REQUIRED_COLUMNS


def test_missing_column_raises(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """Dropping a required column raises ValueError naming the missing column."""
    path = _write_csv(valid_df.drop(columns=["age"]), tmp_path)
    with pytest.raises(ValueError, match=r"Missing required columns: \['age'\]"):
        load_bookings(path)


def test_extra_column_raises(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """Adding an unexpected column raises ValueError naming the extra column."""
    df = valid_df.copy()
    df["junk"] = 1
    path = _write_csv(df, tmp_path)
    with pytest.raises(ValueError, match=r"Unexpected extra columns: \['junk'\]"):
        load_bookings(path)


def test_null_target_raises(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """A null in the target column raises ValueError."""
    df = valid_df.copy()
    df.loc[0, "rebooked"] = np.nan
    path = _write_csv(df, tmp_path)
    with pytest.raises(ValueError, match=r"Target column 'rebooked' contains null values"):
        load_bookings(path)


def test_column_order_does_not_matter(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """The loader reorders columns to canonical regardless of CSV column order.

    This protects downstream code (the transformer, the API) from depending
    on CSV authoring conventions.
    """
    shuffled = valid_df[list(reversed(REQUIRED_COLUMNS))]
    path = _write_csv(shuffled, tmp_path)
    df = load_bookings(path)
    assert tuple(df.columns) == REQUIRED_COLUMNS


def test_accepts_string_and_path_input(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """load_bookings accepts both a str path and a pathlib.Path.

    The function signature is `Path | str`; this test pins that contract.
    """
    path = _write_csv(valid_df, tmp_path)
    from_path = load_bookings(path)
    from_str = load_bookings(str(path))
    pd.testing.assert_frame_equal(from_path, from_str)
