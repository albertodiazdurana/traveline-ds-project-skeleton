"""Unit tests for FeatureTransformer.

These tests exercise the transformer's contract in isolation. They pass
regardless of whether the calling code in train.py is correct. The planted
leakage bug lives in train.py's orchestration (calling fit_transform on
the full dataset before train_test_split), not in this transformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rebooking.features.transform import FeatureTransformer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small in-memory DataFrame matching the schema of data/raw/bookings.csv.

    Built explicitly here so the tests don't depend on the generator script
    or a pre-existing CSV. Includes one null in days_since_last_booking
    and a categorical with multiple destinations.
    """
    return pd.DataFrame(
        {
            "customer_id": list(range(1, 11)),
            "age": [25, 40, 55, 30, 65, 45, 22, 38, 50, 60],
            "destination": ["BCN", "PAR", "BCN", "ROM", "PAR", "BCN", "LIS", "ROM", "PAR", "BCN"],
            "booking_channel": ["web", "app", "web", "agent", "app", "web", "phone", "app", "web", "agent"],
            "budget": [800.0, 1500.0, 2000.0, 1200.0, 3000.0, 900.0, 600.0, 1800.0, 2500.0, 1100.0],
            "days_since_last_booking": [100, 500, 1200, 200, np.nan, 800, 50, 1500, 300, 700],
            "booking_month": [1, 6, 12, 3, 7, 11, 2, 8, 5, 10],
            "device_type": ["mobile", "desktop", "mobile", "tablet", "desktop", "mobile", "mobile", "desktop", "mobile", "tablet"],
            "loyalty_tier": ["none", "silver", "gold", "none", "platinum", "silver", "none", "gold", "platinum", "none"],
            "prior_complaints": [0, 1, 0, 2, 0, 0, 1, 0, 0, 1],
            "party_size": [2, 4, 2, 3, 5, 1, 2, 6, 3, 2],
            "rebooked": [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        }
    )


def test_fit_transform_output_shape(sample_df: pd.DataFrame) -> None:
    """fit_transform produces (n_rows, n_features) with n_rows preserved."""
    X = sample_df.drop(columns=["customer_id", "rebooked"])
    ft = FeatureTransformer()
    out = ft.fit_transform(X)
    assert out.shape[0] == len(X)
    # 5 numeric + cat one-hot (variable) + 2 cyclic + 1 target-enc
    # Fixture has 4 destinations + 4 channels + 3 devices + 4 loyalty = 15 one-hot
    # so width = 5 + 15 + 2 + 1 = 23
    assert out.shape[1] == 23


def test_train_test_widths_match(sample_df: pd.DataFrame) -> None:
    """A transformer fit on train data and used to transform test data
    produces the same number of columns. This is the train-serve parity
    contract: serve-time rows go through the same pipeline as train rows.
    """
    X = sample_df.drop(columns=["customer_id", "rebooked"])
    ft = FeatureTransformer()
    train_out = ft.fit_transform(X.iloc[:7])
    test_out = ft.transform(X.iloc[7:])
    assert train_out.shape[1] == test_out.shape[1]


def test_no_nulls_in_output(sample_df: pd.DataFrame) -> None:
    """Median imputation eliminates the null in days_since_last_booking;
    nothing else in the output should be NaN.
    """
    X = sample_df.drop(columns=["customer_id", "rebooked"])
    assert X["days_since_last_booking"].isnull().sum() == 1
    out = FeatureTransformer().fit_transform(X)
    assert not np.isnan(out).any()


def test_unseen_category_survives_transform(sample_df: pd.DataFrame) -> None:
    """A destination that never appeared during fit must not crash
    transform. It produces an all-zeros row in its one-hot block (via
    OneHotEncoder(handle_unknown='ignore')) and falls back to the global
    rate in the target-encoded column.
    """
    X = sample_df.drop(columns=["customer_id", "rebooked"])
    y = sample_df["rebooked"]
    ft = FeatureTransformer().fit(X, y)

    unseen_row = X.iloc[[0]].copy()
    unseen_row["destination"] = "ZZZ_NEVER_SEEN"
    out = ft.transform(unseen_row)

    assert out.shape == (1, 23)
    # target_rate_ has no entry for 'ZZZ_NEVER_SEEN' → last column falls back to global_rate_
    assert out[0, -1] == pytest.approx(ft.global_rate_)


def test_median_learned_from_fit_data_only(sample_df: pd.DataFrame) -> None:
    """The median used for imputation must reflect the data passed to fit,
    not a hardcoded value or data from other calls.
    """
    X = sample_df.drop(columns=["customer_id", "rebooked"])
    # Slice 1: first 5 rows -> median of [100, 500, 1200, 200, nan]
    ft1 = FeatureTransformer().fit(X.iloc[:5])
    # Slice 2: last 5 rows -> median of [800, 50, 1500, 300, 700]
    ft2 = FeatureTransformer().fit(X.iloc[5:])
    assert ft1.median_days_ != ft2.median_days_
    # Slice 1 median is from {100, 500, 1200, 200} -> 350
    assert ft1.median_days_ == pytest.approx(350.0)


def test_target_encoding_optional(sample_df: pd.DataFrame) -> None:
    """fit(X) without y is backward-compatible: target_rate_ stays empty,
    global_rate_ stays 0.0. Useful for callers that don't have labels at
    fit time (e.g., a pipeline that fits on unlabeled data).
    """
    X = sample_df.drop(columns=["customer_id", "rebooked"])
    ft = FeatureTransformer().fit(X)
    assert ft.target_rate_ == {}
    assert ft.global_rate_ == 0.0


def test_target_encoding_populated_when_y_passed(sample_df: pd.DataFrame) -> None:
    """fit(X, y) populates target_rate_ with one entry per destination
    seen in X, with values matching the per-destination mean of y.
    """
    X = sample_df.drop(columns=["customer_id", "rebooked"])
    y = sample_df["rebooked"]
    ft = FeatureTransformer().fit(X, y)

    expected_destinations = set(X["destination"].unique())
    assert set(ft.target_rate_.keys()) == expected_destinations
    # global_rate_ matches y.mean()
    assert ft.global_rate_ == pytest.approx(float(y.mean()))
    # BCN appears 4 times with rebooked = [0, 0, 0, 0] -> rate 0.0
    assert ft.target_rate_["BCN"] == pytest.approx(0.0)
    # PAR appears 3 times with rebooked = [1, 1, 1] -> rate 1.0
    assert ft.target_rate_["PAR"] == pytest.approx(1.0)


def test_fit_transform_equals_fit_then_transform(sample_df: pd.DataFrame) -> None:
    """The canonical sklearn identity: fit_transform(X) == fit(X).transform(X).
    This is the contract that lets callers reason about the two equivalences.
    """
    X = sample_df.drop(columns=["customer_id", "rebooked"])
    y = sample_df["rebooked"]
    via_fit_transform = FeatureTransformer().fit_transform(X, y)
    via_fit_then_transform = FeatureTransformer().fit(X, y).transform(X)
    np.testing.assert_array_almost_equal(via_fit_transform, via_fit_then_transform)
