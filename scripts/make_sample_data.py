"""Generate synthetic customer-booking data for the rebooking skeleton.

The data is tuned so the planted data-leakage bug in train.py produces a
realistic, measurable AUC inflation of ~1-3 points (occasionally up to 5)
between buggy (fit-on-full-data-before-split) and correct ordering.

This is deliberately not a dramatic 30-point textbook example: realistic
leakage in production usually looks like "slightly better than expected"
test metrics, not an obvious red flag. That subtlety is the interview
talking point — the dangerous form of leakage is the one that ships.

The leakage signal lives primarily in the target-encoded `destination`
feature (see FeatureTransformer): each city is replaced by its rebooking
rate in the fit data. Under the buggy ordering, that rate includes test
labels — most damagingly for rare destinations where the leaked label is
nearly the entire signal for that group. Under correct ordering, rare
test-only destinations fall back to the global mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_ROWS = 250
RNG_SEED = 42

# 25 destinations, heavily skewed: top 4 cover ~50%, long tail of rare cities.
DESTINATIONS = [
    "BCN", "PAR", "ROM", "LIS",   # head: very common
    "AMS", "BER", "VIE", "PRG",   # body: common
    "DUB", "CPH", "ATH", "MAD",   # body: uncommon
    "STO", "OSL", "HEL", "WAW",   # tail: rare
    "BUD", "ZRH", "BRU", "LUX",   # tail: rare
    "TAL", "RIG", "VNO", "LJU",   # tail: very rare
    "VAL",                         # very rare
]
CHANNELS = ["web", "app", "agent", "phone"]
DEVICES = ["mobile", "desktop", "tablet"]
LOYALTY = ["none", "silver", "gold", "platinum"]


def _dest_probs() -> np.ndarray:
    """Steep power-law destination distribution: long tail amplifies leakage."""
    # First 4 destinations together carry ~50%; last 5 carry ~3% combined.
    ranks = np.arange(1, len(DESTINATIONS) + 1, dtype=float)
    weights = 1.0 / (ranks ** 1.2)
    return weights / weights.sum()


def _dest_effects(rng: np.random.Generator) -> dict[str, float]:
    """Each destination has a structured per-city base-rate effect on rebooking.

    Effects are drawn from N(0, 0.9), so each city shifts the logit by
    roughly ±1 unit. This is the signal that leakage steals: at correct
    ordering, rare test-only destinations get all-zero one-hot rows and
    contribute nothing; under buggy ordering, the encoder learned them
    and the model picks up their effect.
    """
    return {city: float(rng.normal(0.0, 1.6)) for city in DESTINATIONS}


def _generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dest_effects = _dest_effects(rng)

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
        rng,
        age=age,
        destination=destination,
        channel=booking_channel,
        budget=budget,
        days_since=days_since,
        loyalty=loyalty_tier,
        complaints=prior_complaints,
        dest_effects=dest_effects,
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


def _target(
    rng: np.random.Generator,
    *,
    age: np.ndarray,
    destination: np.ndarray,
    channel: np.ndarray,
    budget: np.ndarray,
    days_since: np.ndarray,
    loyalty: np.ndarray,
    complaints: np.ndarray,
    dest_effects: dict[str, float],
) -> np.ndarray:
    """Learnable target. Honest AUC target ~0.85.

    The destination contribution is the leakage-relevant term: per-city
    structured effects of ~±1 in the logit. When the encoder vocabulary
    is leaked from the full dataset, the model picks up those effects on
    test-only rare cities; under correct ordering, the model never sees
    them and gets the all-zero unseen-category row instead.
    """
    dest_contribution = np.array([dest_effects[c] for c in destination])
    logit = (
        -1.0
        + 0.030 * (age - 40)
        + 0.0008 * (budget - 1_200)
        - 0.0012 * np.nan_to_num(days_since, nan=900)
        + np.where(channel == "app", 0.85, 0.0)
        + np.where(channel == "agent", -0.60, 0.0)
        + np.where(loyalty == "platinum", 1.8, 0.0)
        + np.where(loyalty == "gold", 0.9, 0.0)
        - 0.70 * complaints
        + dest_contribution
    )
    logit += rng.normal(0, 0.30, size=age.size)
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
    n_dest = df["destination"].nunique()
    rarest = df["destination"].value_counts().min()
    print(
        f"Wrote {len(df)} rows to {args.output} "
        f"(positive rate: {pos_rate:.2%}, destinations: {n_dest}, rarest: {rarest} rows)"
    )


if __name__ == "__main__":
    main()
