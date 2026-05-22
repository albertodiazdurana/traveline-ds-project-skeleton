"""Feature transformer: shared train/serve transformation, fitted on train only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["destination", "booking_channel", "device_type", "loyalty_tier"]
NUMERIC = ["age", "budget", "prior_complaints", "party_size", "days_since_last_booking"]
DROP = ["customer_id", "rebooked"]


class FeatureTransformer:
    """Fit on train, transform train and serve identically. Prevents train-serve skew."""

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.median_days_: float | None = None

    def fit(self, X: pd.DataFrame) -> "FeatureTransformer":
        self.median_days_ = float(X["days_since_last_booking"].median())
        numeric = self._numeric_block(X)
        self.encoder.fit(X[CATEGORICAL].astype(str))
        self.scaler.fit(numeric)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cat = np.asarray(self.encoder.transform(X[CATEGORICAL].astype(str)))
        num = self.scaler.transform(self._numeric_block(X))
        month = X["booking_month"].to_numpy()
        cyclic = np.column_stack([np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)])
        return np.concatenate([num, cat, cyclic], axis=1)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def _numeric_block(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[NUMERIC].copy()
        df["days_since_last_booking"] = df["days_since_last_booking"].fillna(self.median_days_)
        return df
