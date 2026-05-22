"""Feature transformer: shared train/serve transformation, fitted on train only.

Includes a target-encoded `destination` feature: each city is replaced by
its rebooking rate in the fit data. This is the classic textbook leakage
trap — if fit is called on data that includes test labels, those labels
get baked into the feature. Under correct fit-on-train-only ordering, the
encoding contains only train-label information; unseen test destinations
fall back to the global mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["destination", "booking_channel", "device_type", "loyalty_tier"]
NUMERIC = ["age", "budget", "prior_complaints", "party_size", "days_since_last_booking"]
TARGET_ENCODE = "destination"


class FeatureTransformer:
    """Fit on train, transform train and serve identically. Prevents train-serve skew."""

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.median_days_: float | None = None
        self.target_rate_: dict[str, float] = {}
        self.global_rate_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureTransformer":
        self.median_days_ = float(X["days_since_last_booking"].median())
        numeric = self._numeric_block(X)
        self.encoder.fit(X[CATEGORICAL].astype(str))
        self.scaler.fit(numeric)
        if y is not None:
            self.global_rate_ = float(y.mean())
            grouped = pd.DataFrame(
                {TARGET_ENCODE: X[TARGET_ENCODE].to_numpy(), "_y": y.to_numpy()}
            )
            means = grouped.groupby(TARGET_ENCODE)["_y"].mean()
            self.target_rate_ = {str(k): float(v) for k, v in means.items()}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cat = np.asarray(self.encoder.transform(X[CATEGORICAL].astype(str)))
        num = self.scaler.transform(self._numeric_block(X))
        month = X["booking_month"].to_numpy()
        cyclic = np.column_stack([np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)])
        target_enc = X[TARGET_ENCODE].map(self.target_rate_).fillna(self.global_rate_).to_numpy()
        target_enc = target_enc.reshape(-1, 1)
        return np.concatenate([num, cat, cyclic, target_enc], axis=1)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _numeric_block(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[NUMERIC].copy()
        df["days_since_last_booking"] = df["days_since_last_booking"].fillna(self.median_days_)
        return df
