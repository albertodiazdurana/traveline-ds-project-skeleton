"""Production-shape refactor of the rebooking feature transformation + training step.

This file is a reference, not the project's runtime code. It addresses three
things the runtime version trades off in exchange for being a tight 42-line
class:

1. Structural prevention of train/test data leakage via sklearn.Pipeline
   (the bug planted in src/rebooking/models/train.py becomes impossible
   to write here, because the framework calls fit_transform on train and
   transform on test for each split automatically).
2. cross_val_score and GridSearchCV ergonomics out of the box.
3. Custom transformers integrated as first-class sklearn estimators
   (BaseEstimator + TransformerMixin) rather than ad-hoc methods on a
   class.

Run end-to-end from the repo root with the project venv activated:

    python dsm-docs/guides/transform-fix.py

Requires the sample data to already exist:

    python scripts/make_sample_data.py

Expected output (numbers will vary slightly with rng):

    Fit on train (200), evaluate on test (50):
      Train AUC: 0.8xx
      Test  AUC: 0.6xx   <- gap is real, not the leakage-inflated value

    5-fold cross-validated test AUC: 0.7xx +/- 0.0xx

    GridSearchCV over C in [0.01, 0.1, 1, 10]:
      Best C: <value>
      Best CV AUC: 0.7xx

---

Measured impact of the planted bug
-----------------------------------

Comparing this file's correct-by-construction Pipeline against
src/rebooking/models/train.py's fit-on-full-data-before-split ordering,
on the same 250-row generated dataset:

  Single split (seed=42, same as train.py defaults):
    Buggy   Test AUC: 0.686
    Correct Test AUC: 0.674
    ---------------------------
    Inflation:        +0.012  (+1.2 points)

  20 random splits (seeds 0-19):
    Mean inflation:   +0.017  (+1.7 points)
    Max inflation:    +0.054  (+5.4 points)
    Min inflation:    -0.004  (negative on 2/20 splits)
    Positive in:      18/20 splits

  Both train AUCs are essentially identical (~0.88) — leakage doesn't
  inflate train metrics, only test metrics. That's why the bug is
  insidious: the inflated test AUC looks like generalization, when in
  fact the model has 'seen' the test set's destination-rate distribution
  through the target-encoded feature in src/rebooking/features/transform.py.

Why the inflation is modest, not dramatic
------------------------------------------

With 25 destinations spread across 200 train rows, the OneHotEncoder
sees nearly the full vocabulary regardless of split ordering — simple
one-hot leakage doesn't bite. The inflation that DOES show up comes
from the target-encoded destination column: when fit on full data, the
per-city mean rebooking rate includes test labels. For rare destinations
(e.g., 'VAL' with 1 row total) the leaked rate is essentially the test
row's own label.

This realistic 1-3 point inflation is the interview talking point.
Textbook leakage examples with 20-30 point gaps require artificial setups
(N=50, contrived features); production leakage almost always looks like
the modest signal here — 'test metrics slightly better than expected',
which is exactly what makes it dangerous and easy to ship.

Reproducing the measurement
---------------------------

Generate the dataset, then run this file's main() (single-split AUC) and
the across-splits comparison snippet inline:

    python scripts/make_sample_data.py
    python dsm-docs/guides/transform-fix.py
    # For the across-splits comparison, see dsm-docs/guides/smoke-tests.md
    # under the train.py (Item 11) section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

CATEGORICAL = ["destination", "booking_channel", "device_type", "loyalty_tier"]
NUMERIC = ["age", "budget", "prior_complaints", "party_size", "days_since_last_booking"]
CYCLIC = ["booking_month"]
TARGET_ENCODE = ["destination"]


class CyclicMonthEncoder(BaseEstimator, TransformerMixin):
    """Encode month-of-year as (sin, cos) pair.

    sklearn has no built-in cyclic encoder. Implementing it as a proper
    estimator (rather than a method on FeatureTransformer) lets it slot
    into ColumnTransformer cleanly and inherits cross_val_score / clone /
    set_params support for free.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CyclicMonthEncoder":
        return self  # stateless; sklearn estimators must implement fit

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        month = np.asarray(X).ravel().astype(float)
        return np.column_stack([np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)])


def build_pipeline(C: float = 1.0) -> Pipeline:
    """Build the train-serve-parity Pipeline.

    Pipeline + ColumnTransformer composition is the sklearn-idiomatic way
    to apply different transforms to different column groups. The
    framework guarantees that for any Pipeline.fit(X_train, y_train)
    followed by Pipeline.predict(X_test), the transformers see fit on
    train data only and transform on test data only -- the leakage bug
    in the runtime code becomes structurally unwriteable here.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("month", CyclicMonthEncoder(), CYCLIC),
            # TargetEncoder is the production-shape version of the per-destination
            # target encoding in the runtime FeatureTransformer. Pipeline guarantees
            # it sees only the train fold's labels during fit, structurally
            # preventing the leakage the runtime version is vulnerable to.
            ("dest_target", TargetEncoder(target_type="binary", random_state=42), TARGET_ENCODE),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", LogisticRegression(C=C, max_iter=200, random_state=42)),
        ]
    )


def main() -> None:
    df = pd.read_csv("data/raw/bookings.csv")
    X = df.drop(columns=["customer_id", "rebooked"])
    y = df["rebooked"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 1. Single train/test fit, correctly ordered by construction ----
    pipe = build_pipeline(C=1.0)
    pipe.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"Fit on train ({len(X_train)}), evaluate on test ({len(X_test)}):")
    print(f"  Train AUC: {train_auc:.3f}")
    print(f"  Test  AUC: {test_auc:.3f}   <- gap is real, not leakage-inflated")
    print()

    # ---- 2. Cross-validated AUC ----
    # cross_val_score handles split-and-fit-and-evaluate for each fold,
    # calling fit_transform on the fold's train side and transform on the
    # fold's test side. No way to accidentally leak.
    cv_scores = cross_val_score(build_pipeline(), X, y, cv=5, scoring="roc_auc")
    print(f"5-fold cross-validated test AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print()

    # ---- 3. GridSearchCV over the model's regularization strength ----
    # Note that the grid keys reference 'model__C' -- double underscore is
    # sklearn's syntax for 'parameter named C on the step named model'.
    # The Pipeline structure is what makes this work uniformly.
    grid = GridSearchCV(
        build_pipeline(),
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
    )
    grid.fit(X, y)
    print("GridSearchCV over C in [0.01, 0.1, 1, 10]:")
    print(f"  Best C: {grid.best_params_['model__C']}")
    print(f"  Best CV AUC: {grid.best_score_:.3f}")


if __name__ == "__main__":
    main()
