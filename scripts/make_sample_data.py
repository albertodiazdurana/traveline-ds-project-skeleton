"""Generate synthetic customer-booking data for the rebooking skeleton.

The data is tuned so that the planted data-leakage bug in train.py
produces a visible AUC gap: train AUC vs. (leaky) test AUC of ~1 pt,
vs. ~7 pt under a correct split-before-fit ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_ROWS = 1_000
RNG_SEED = 42

DESTINATIONS = [
    "BCN", "PAR", "ROM", "LIS", "AMS", "BER",
    "VIE", "PRG", "DUB", "CPH", "ATH", "MAD",
]
CHANNELS = ["web", "app", "agent", "phone"]
DEVICES = ["mobile", "desktop", "tablet"]
LOYALTY = ["none", "silver", "gold", "platinum"]


def _generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 81, size=n)
    destination = rng.choice(DESTINATIONS, size=n, p=_dest_probs())
    booking_channel = rng.choice(CHANNELS, size=n, p=[0.45, 0.30, 0.15, 0.10])
    budget = rng.gamma(shape=2.0, scale=600, size=n).clip(200, 5_000).round(2)
    days_since = rng.integers(30, 1_801, size=n).astype(float)
    null_mask = rng.random(n) < 0.08
    days_since[null_mask] = np.nan
    booking_month = rng.integers(1, 13, size=n)
    device_type = rng.choice(DEVICES, size=n, p=[0.55, 0.35, 0.10])
    loyalty_tier = rng.choice(LOYALTY, size=n, p=[0.55, 0.25, 0.15, 0.05])
    prior_complaints = rng.poisson(lam=0.4, size=n)
    party_size = rng.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.30, 0.40, 0.15, 0.10, 0.03, 0.02])

    rebooked = _target(
        rng, age, booking_channel, budget, days_since, loyalty_tier, prior_complaints
    )

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": age,
            "destination": destination,
            "booking_channel": booking_channel,
            "budget": budget,
            "days_since_last_booking": days_since,
            "booking_month": booking_month,
            "device_type": device_type,
            "loyalty_tier": loyalty_tier,
            "prior_complaints": prior_complaints,
            "party_size": party_size,
            "rebooked": rebooked,
        }
    )


def _dest_probs() -> list[float]:
    """Heavy-tailed destination distribution: leakage signal lives in rare categories."""
    weights = np.array([0.22, 0.18, 0.14, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
    return (weights / weights.sum()).tolist()


def _target(
    rng: np.random.Generator,
    age: np.ndarray,
    channel: np.ndarray,
    budget: np.ndarray,
    days_since: np.ndarray,
    loyalty: np.ndarray,
    complaints: np.ndarray,
) -> np.ndarray:
    """Construct a moderately learnable target with enough noise to keep AUC ~0.85."""
    logit = (
        -1.5
        + 0.018 * (age - 40)
        + 0.0004 * (budget - 1_200)
        - 0.0008 * np.nan_to_num(days_since, nan=900)
        + np.where(channel == "app", 0.55, 0.0)
        + np.where(channel == "agent", -0.40, 0.0)
        + np.where(loyalty == "platinum", 1.2, 0.0)
        + np.where(loyalty == "gold", 0.6, 0.0)
        - 0.45 * complaints
    )
    logit += rng.normal(0, 0.6, size=age.size)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return (rng.random(age.size) < prob).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/bookings.csv"),
        help="Output CSV path (default: data/raw/bookings.csv)",
    )
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    df = _generate(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    pos_rate = df["rebooked"].mean()
    print(f"Wrote {len(df)} rows to {args.output} (positive rate: {pos_rate:.2%})")


if __name__ == "__main__":
    main()
