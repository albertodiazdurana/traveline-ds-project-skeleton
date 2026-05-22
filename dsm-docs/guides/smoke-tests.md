# Smoke Tests

Per-item smoke tests run against each file as it's built. A smoke test is a small, fast check that confirms the basic thing still works — the term originates from electrical hardware ("plug it in and see if it smokes") and was popularized for software by Microsoft's daily-build-and-smoke-test practice (McConnell, *Code Complete* Ch. 22). Each entry below pairs the exact command(s) executed against the user's `.venv/` with the expected output and the actual result. New items are appended as the build progresses.

**Why smoke-test per item:** Each file is exercised end-to-end by a human before moving to the next item. This catches integration mismatches that unit tests miss (wrong filename in an import, schema drift between modules, hardcoded path that only works in CI) and surfaces them at the moment they're introduced, when the relevant code is still in working memory. It also produces a defensible interview / demo narrative: each commit corresponds to an item that was *seen working* before being declared done.

**Format per entry:** subsection per file, numbered subsections per check, each block contains: command → expected → result. Mark each result `✓ <ISO date>` or `✗` with the diagnostic.

---

## Retune log — 2026-05-22

After the initial smoke tests on the 1000×12 dataset, the leakage signal was found to be functionally zero (categorical leakage doesn't bite when train has full vocabulary coverage). The sample data generator and `FeatureTransformer` were retuned:

- `make_sample_data.py`: shrunk to **250 rows**, expanded **destination** to 25 cities with a steep power-law tail, added per-destination structured target effects (~±2 logits), boosted honest AUC to ~0.85.
- `FeatureTransformer`: added a **target-encoded `destination`** feature (`fit(X, y)` now optionally accepts y; if passed, per-city rebooking-rate is stored and replaces destination at transform time, with global-mean fallback for unseen).
- `transform-fix.py`: added `sklearn.preprocessing.TargetEncoder` to the ColumnTransformer to match the runtime feature set.

The pre-retune entries below remain as historical record. The post-retune entries are appended at the section per file with the **(post-retune)** label. Both versions of `make_sample_data.py`, `loader.py`, and `transform.py` exercise the same contract; only the numbers changed.

**Post-retune leakage measurement** (250 rows, target encoding, 20 random splits):
- Honest test AUC: 0.775 ± 0.064
- Buggy test AUC: 0.792 ± 0.059
- Mean inflation: +1.7 pts (max +5.4 pts), positive in 18/20 splits

The realistic ~2pt inflation is the interview talking point: real-world leakage rarely looks like the textbook 30-point inflation; it looks like "test metrics slightly better than expected" — which is what makes it dangerous.

---

## `src/rebooking/__init__.py` (Item 4)

### 1. Editable install + import

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import rebooking; print(rebooking.__version__)"
```

**Expected:** install succeeds; final line prints `0.1.0`.

**Result:** ✓ 2026-05-22 — printed `0.1.0` from `.venv/bin/python`.

---

## `scripts/make_sample_data.py` (Item 17)

### 1. Generate sample CSV

```bash
python scripts/make_sample_data.py
```

**Expected:** `Wrote 1000 rows to data/raw/bookings.csv (positive rate: <somewhere around 15%>)`.

**Result:** ✓ 2026-05-22 — `Wrote 1000 rows to data/raw/bookings.csv (positive rate: 15.20%)`.

### 2. Inspect schema and stats

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/bookings.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Null counts:', df.isnull().sum()[df.isnull().sum() > 0].to_dict())
print('Target distribution:', df['rebooked'].value_counts().to_dict())
"
```

**Expected:** shape (1000, 12); 12-column schema in canonical order; ~80-100 nulls in `days_since_last_booking`; target ~15% positive.

**Result:** ✓ 2026-05-22 — shape (1000, 12), 87 nulls in `days_since_last_booking`, target {0: 848, 1: 152}.

---

## `src/rebooking/data/loader.py` (Item 6)

### 1. Happy path

```bash
python -c "
from rebooking.data.loader import load_bookings
df = load_bookings('data/raw/bookings.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Target nulls:', df['rebooked'].isnull().sum())
"
```

**Expected:** shape (1000, 12), 12 columns in canonical order, target nulls 0.

**Result:** ✓ 2026-05-22 — matched exactly.

### 2. Missing column → ValueError

```bash
python -c "
import pandas as pd
from rebooking.data.loader import load_bookings
df = pd.read_csv('data/raw/bookings.csv')
df.drop(columns=['age']).to_csv('/tmp/bad.csv', index=False)
try:
    load_bookings('/tmp/bad.csv')
except ValueError as e:
    print('Caught:', e)
"
rm /tmp/bad.csv
```

**Expected:** `Caught: Missing required columns: ['age']`.

**Result:** ✓ 2026-05-22 — exact match.

### 3. Extra column → ValueError

```bash
python -c "
import pandas as pd
from rebooking.data.loader import load_bookings
df = pd.read_csv('data/raw/bookings.csv')
df['junk'] = 1
df.to_csv('/tmp/bad2.csv', index=False)
try:
    load_bookings('/tmp/bad2.csv')
except ValueError as e:
    print('Caught:', e)
"
rm /tmp/bad2.csv
```

**Expected:** `Caught: Unexpected extra columns: ['junk']`.

**Result:** ✓ 2026-05-22 — exact match.

### 4. Null in target → ValueError

```bash
python -c "
import pandas as pd
import numpy as np
from rebooking.data.loader import load_bookings
df = pd.read_csv('data/raw/bookings.csv')
df.loc[0, 'rebooked'] = np.nan
df.to_csv('/tmp/bad3.csv', index=False)
try:
    load_bookings('/tmp/bad3.csv')
except ValueError as e:
    print('Caught:', e)
"
rm /tmp/bad3.csv
```

**Expected:** `Caught: Target column 'rebooked' contains null values`.

**Result:** ✓ 2026-05-22 — exact match.

---

## `src/rebooking/features/transform.py` (Item 7)

### 1. Shape preservation across train/test

```bash
python -c "
from rebooking.data.loader import load_bookings
from rebooking.features.transform import FeatureTransformer

df = load_bookings('data/raw/bookings.csv')
ft = FeatureTransformer()
X_train = ft.fit_transform(df.iloc[:800])
X_test = ft.transform(df.iloc[800:])
print('Train:', X_train.shape, 'Test:', X_test.shape)
print('Same width:', X_train.shape[1] == X_test.shape[1])
"
```

**Expected:** `Train: (800, 30) Test: (200, 30) Same width: True`. The 30 = 5 numeric (scaled) + 23 one-hot (12 dest + 4 channel + 3 device + 4 loyalty) + 2 cyclic month.

**Result:** ✓ 2026-05-22 — exact match.

### 2. No nulls after median imputation

```bash
python -c "
import numpy as np
from rebooking.data.loader import load_bookings
from rebooking.features.transform import FeatureTransformer

df = load_bookings('data/raw/bookings.csv')
print('Nulls in input (days_since_last_booking):', df['days_since_last_booking'].isnull().sum())
ft = FeatureTransformer()
X = ft.fit_transform(df)
print('Nulls in output:', np.isnan(X).sum())
print('Median learned:', ft.median_days_)
"
```

**Expected:** input nulls ~87, output nulls 0, median around 885.

**Result:** ✓ 2026-05-22 — input nulls 87, output nulls 0, median 885.5.

### 3. Unseen-category survival

```bash
python -c "
from rebooking.data.loader import load_bookings
from rebooking.features.transform import FeatureTransformer

df = load_bookings('data/raw/bookings.csv')
ft = FeatureTransformer().fit(df.iloc[:800])

serve_row = df.iloc[[0]].copy()
serve_row['destination'] = 'NEVER_SEEN_BEFORE'
out = ft.transform(serve_row)
print('Survived unseen category:', out.shape)
print('All zeros in destination columns for unseen value:', (out[0, 5:17] == 0).all())
"
```

**Expected:** shape `(1, 30)`, unseen category produces all-zeros in the destination one-hot block.

**Result:** ✓ 2026-05-22 — exact match. `handle_unknown='ignore'` working as designed.

### 4. Train-only learning (median changes with train slice)

```bash
python -c "
from rebooking.data.loader import load_bookings
from rebooking.features.transform import FeatureTransformer

df = load_bookings('data/raw/bookings.csv')
ft1 = FeatureTransformer().fit(df.iloc[:500])
ft2 = FeatureTransformer().fit(df.iloc[500:])
print('Median from first half:', ft1.median_days_)
print('Median from second half:', ft2.median_days_)
print('Different (proves learning, not hardcoded):', ft1.median_days_ != ft2.median_days_)
"
```

**Expected:** two different medians, `Different: True`.

**Result:** ✓ 2026-05-22 — confirmed (medians differ).

### 5. End-to-end smoke test

```bash
python -c "
from rebooking.data.loader import load_bookings
from rebooking.features.transform import FeatureTransformer
df = load_bookings('data/raw/bookings.csv')
X = FeatureTransformer().fit_transform(df)
print(f'OK: {df.shape} -> {X.shape}')
"
```

**Expected:** `OK: (1000, 12) -> (1000, 30)`.

**Result:** ✓ 2026-05-22 — exact match.

---

## Pending validations (added as items land)

- `src/rebooking/models/baseline.py` (Item 8) — instantiation + fit/predict on a 30-feature synthetic matrix
- `src/rebooking/training/train.py` (Item 10) — end-to-end run; leakage gap observable; `mlruns/` populated
- `src/rebooking/config.py` (Item 5) — train.py still produces identical metrics after config extraction
- `src/rebooking/serving/schemas.py` (Item 12) — Pydantic round-trip on a sample request body
- `src/rebooking/serving/app.py` (Item 11) — `/health` 200, `/predict` returns probability shape
- `tests/unit/*.py` (Items 9, 13) — `pytest tests/` green
- `Dockerfile` (Item 14) — `docker build .` succeeds
- `.github/workflows/ci.yml` (Item 16) — YAML parses, action triggers as expected once remote is added

---

## `dsm-docs/guides/transform-fix.py` (reference, not a build item)

Production-shape refactor of the feature transformation + training step using `sklearn.Pipeline` + `ColumnTransformer` + custom `CyclicMonthEncoder`. Demonstrates that the planted data-leakage bug in `train.py` becomes structurally impossible when the Pipeline framework handles fit-transform ordering. Includes `cross_val_score` and `GridSearchCV` demonstrations.

### 1. Runs end-to-end and produces the three expected blocks

```bash
python dsm-docs/guides/transform-fix.py
```

**Expected:** three labeled blocks — single train/test fit (with train + test AUC), 5-fold CV AUC ± std, GridSearchCV best C + best CV AUC.

**Result:** ✓ 2026-05-22 —
```
Fit on train (800), evaluate on test (200):
  Train AUC: 0.743
  Test  AUC: 0.732   <- gap is real, not leakage-inflated
5-fold cross-validated test AUC: 0.689 +/- 0.043
GridSearchCV over C in [0.01, 0.1, 1, 10]:
  Best C: 10.0
  Best CV AUC: 0.690
```

Single-split test AUC (0.732) lands on the optimistic end of the CV distribution (mean 0.689 ± 0.043). This is expected: a single split is one draw from the CV distribution. Cross-validation gives the more honest estimate of generalization.

---

## Post-retune smoke tests (2026-05-22)

These re-exercise the same files against the new 250×12 dataset and the FeatureTransformer with target encoding.

### `make_sample_data.py` (post-retune)

```bash
python scripts/make_sample_data.py
```

**Expected:** `Wrote 250 rows to data/raw/bookings.csv (positive rate: ~26%, destinations: 24-25, rarest: 1 row)`.

**Result:** ✓ 2026-05-22 — `Wrote 250 rows to data/raw/bookings.csv (positive rate: 26.40%, destinations: 24, rarest: 1 rows)`.

### `loader.py` (post-retune)

```bash
python -c "
from rebooking.data.loader import load_bookings
df = load_bookings('data/raw/bookings.csv')
print('Shape:', df.shape)
print('Target nulls:', df['rebooked'].isnull().sum())
print('Null counts:', df.isnull().sum()[df.isnull().sum() > 0].to_dict())
"
```

**Expected:** shape (250, 12), target nulls 0, ~5-10% nulls in `days_since_last_booking`.

**Result:** ✓ 2026-05-22 — Shape: (250, 12); Target nulls: 0; Null counts: {'days_since_last_booking': 20}.

### `transform.py` (post-retune, no y)

```bash
python -c "
from rebooking.data.loader import load_bookings
from rebooking.features.transform import FeatureTransformer
df = load_bookings('data/raw/bookings.csv')
ft = FeatureTransformer()
X = ft.fit_transform(df.iloc[:200])
print('Shape:', X.shape, '(no y -> target encoding is constant 0)')
print('global_rate_:', ft.global_rate_)
"
```

**Expected:** shape (200, 43); target encoding column = 0 everywhere (no y provided → fallback to global_rate_ = 0). 43 = 5 numeric + 35 one-hot (~24 destinations + 4 channels + 3 devices + 4 loyalty) + 2 cyclic + 1 target-encoded.

**Result:** ✓ 2026-05-22 — Shape: (200, 43); global_rate_: 0.0.

### `transform.py` (post-retune, with y — target-encoding branch)

```bash
python -c "
from rebooking.data.loader import load_bookings
from rebooking.features.transform import FeatureTransformer
df = load_bookings('data/raw/bookings.csv')
ft = FeatureTransformer()
X_train = ft.fit_transform(df.iloc[:200].drop(columns=['rebooked']), df.iloc[:200]['rebooked'])
X_test = ft.transform(df.iloc[200:].drop(columns=['rebooked']))
print('Train:', X_train.shape, 'Test:', X_test.shape, 'Same width:', X_train.shape[1] == X_test.shape[1])
print('global_rate_:', ft.global_rate_, 'destinations in target_rate_:', len(ft.target_rate_))
"
```

**Expected:** Train (200, 43), Test (50, 43), same width True; `global_rate_` ~0.25-0.30; ~24 destinations in `target_rate_`.

**Result:** ✓ 2026-05-22 — Train: (200, 43); Test: (50, 43); Same width: True; global_rate_: 0.28; # destinations seen at fit: 24.

### `transform-fix.py` (post-retune, with TargetEncoder added)

```bash
python dsm-docs/guides/transform-fix.py
```

**Expected:** three labeled blocks; train AUC ~0.85-0.88; test AUC ~0.65-0.75 (single-split variance high on small dataset); CV mean ~0.74 ± 0.07.

**Result:** ✓ 2026-05-22 —
```
Fit on train (200), evaluate on test (50):
  Train AUC: 0.875
  Test  AUC: 0.659   <- gap is real, not leakage-inflated
5-fold cross-validated test AUC: 0.742 +/- 0.075
GridSearchCV over C in [0.01, 0.1, 1, 10]:
  Best C: 10.0
  Best CV AUC: 0.749
```

The single-split gap (Train 0.875 vs Test 0.659) is large because the 50-row test set has high variance. The CV estimate (0.742 ± 0.075) is the honest number. The gap to remember: under correct ordering, mean test AUC is ~0.775; under buggy ordering, ~0.792 (a ~1.7 point inflation, see Retune log).

---

## `tests/unit/test_transform.py` (Item 9)

Pytest unit tests for the FeatureTransformer in isolation. These tests pass regardless of the bug that will arrive in `train.py` — the bug is at the orchestration level, the transformer's own contract is correct.

### 1. All tests green via `pytest tests/`

```bash
pytest tests/
```

**Expected:** 8 tests pass, no warnings, completes in <3 seconds.

**Result:** ✓ 2026-05-22 — `8 passed in 1.05s`. Tests:
- `test_fit_transform_output_shape`
- `test_train_test_widths_match`
- `test_no_nulls_in_output`
- `test_unseen_category_survives_transform`
- `test_median_learned_from_fit_data_only`
- `test_target_encoding_optional`
- `test_target_encoding_populated_when_y_passed`
- `test_fit_transform_equals_fit_then_transform`

### 2. Verbose run for individual test status

```bash
pytest tests/unit/test_transform.py -v
```

**Expected:** each test listed individually with PASSED status.

**Result:** ✓ 2026-05-22 — all 8 PASSED.

### 3. Test fixture is self-contained (does not depend on data/raw/bookings.csv)

The `sample_df` pytest fixture builds a 10-row DataFrame in memory with the canonical 12-column schema. This means `pytest tests/` works in any environment — fresh clone, CI runner, container — without first running `make_sample_data.py`. Verified by running the suite from a temp directory equivalent.

**Result:** ✓ 2026-05-22 — fixture pattern intentional. CI-friendly.

---

## `tests/unit/test_loader.py` (Item 8)

Pytest unit tests for `load_bookings`. The loader is the data-ingress boundary; the suite covers the happy path plus the three ValueError branches (missing column, extra column, null target) plus two invariants (column-order independence, str/Path acceptance).

### 1. Full suite green via `pytest tests/`

```bash
pytest tests/
```

**Expected:** 14 tests pass (6 loader + 8 transform), completes in <3 seconds.

**Result:** ✓ 2026-05-22 — `14 passed in 2.22s`.

### 2. Loader tests in isolation

```bash
pytest tests/unit/test_loader.py -v
```

**Expected:** 6 tests, all PASSED:
- `test_happy_path_returns_dataframe`
- `test_missing_column_raises`
- `test_extra_column_raises`
- `test_null_target_raises`
- `test_column_order_does_not_matter`
- `test_accepts_string_and_path_input`

**Result:** ✓ 2026-05-22 — all 6 PASSED.

### 3. Each test uses pytest's `tmp_path` fixture (no test artifacts persist)

Each test writes a CSV under pytest's `tmp_path` directory and reads it back through `load_bookings`. The directory is auto-cleaned by pytest after the run; no `/tmp/` cleanup required. This pattern is the modern replacement for the `/tmp/bad.csv` manual smoke tests recorded earlier in this file.

**Result:** ✓ 2026-05-22 — confirmed pattern: tests are hermetic.

---

## `configs/training.yaml` (Item 10)

Declarative hyperparameters consumed by `src/rebooking/models/train.py` (Item 11). Three top-level sections: `data:`, `model:`, `mlflow:`. Loaded via PyYAML (now a direct dependency in `pyproject.toml`) and parsed into a Pydantic settings model for type-safe access (added in Item 11).

### 1. YAML parses with correct Python types

```bash
python -c "
import yaml
with open('configs/training.yaml') as f:
    cfg = yaml.safe_load(f)
print('Sections:', list(cfg.keys()))
print('test_size type:', type(cfg['data']['test_size']).__name__)
print('class_weight is None:', cfg['model']['class_weight'] is None)
print('log_model type:', type(cfg['mlflow']['log_model']).__name__)
"
```

**Expected:** Sections `['data', 'model', 'mlflow']`; `test_size` is `float`; `class_weight` is `None` (not the string `'null'`); `log_model` is `bool`.

**Result:** ✓ 2026-05-22 — exact match. PyYAML correctly coerces `null` → `None`, `0.2` → float, `true` → bool.

### 2. data.raw_csv resolves to an actual file

```bash
python -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(open('configs/training.yaml'))
p = Path(cfg['data']['raw_csv'])
print(f'Path: {p}')
print(f'Exists: {p.exists()}')
"
```

**Expected:** `data/raw/bookings.csv`, exists True (assuming `make_sample_data.py` has been run).

**Result:** ✓ 2026-05-22 — `data/raw/bookings.csv`, Exists: True.

### 3. Schema design check (will be enforced via Pydantic in Item 11)

The YAML's three sections map to three Pydantic models that train.py will define:
- `DataConfig`: raw_csv (Path), test_size (float, 0 < x < 1), random_state (int)
- `ModelConfig`: type (Literal["logistic_regression"]), C (float, > 0), max_iter (int, > 0), class_weight (Literal["balanced"] | None)
- `MLflowConfig`: experiment_name (str), tracking_uri (str), log_model (bool)

Typo in a YAML key → Pydantic ValidationError at load time, not at first access. Defensible interview claim: "this is type-safe config; the alternative is dict access with KeyError at runtime."

---

## `src/rebooking/models/train.py` (Item 11)

The training orchestrator. Loads data, calls `FeatureTransformer.fit_transform` on the full feature matrix, then splits — **this is where the planted bug lives** (fit-on-full-data-before-split). The transformer and loader are correct in isolation (their unit tests are green); the bug is only at this orchestration level.

### 1. End-to-end run on default config

```bash
python -m rebooking.models.train
```

**Expected:** runs without error; prints train AUC, test AUC, artifact path; writes `mlruns/<exp_id>/<run_id>/` and `artifacts/model.joblib`. May emit two warnings (MLflow filesystem-backend FutureWarning, sklearn pickle vs. skops note); both are informational.

**Result:** ✓ 2026-05-22 —
```
Train AUC: 0.884
Test  AUC: 0.686
Artifact: artifacts/model.joblib
```

### 2. A/B measurement of the planted bug (seed=42 single split)

Compare train.py-as-committed against the same logic with correct ordering (split first, fit transformer on train only):

| Variant | Train AUC | Test AUC |
|---|---|---|
| Buggy (train.py as committed) | 0.884 | 0.686 |
| Correct (split first) | 0.885 | 0.674 |

**Test-AUC inflation due to leakage: +1.2 points** on this seed. Across 20 random splits (from the retune-turn measurement) the mean inflation was +1.7 pts (max +5.4 pts), positive in 18/20 splits.

This is the realistic-leakage scale: small, consistent, and dangerous specifically because it doesn't look obviously wrong.

### 3. Artifact contents

```bash
python -c "
import joblib
d = joblib.load('artifacts/model.joblib')
print('Keys:', list(d.keys()))
print('Transformer median:', d['transformer'].median_days_)
print('Model coef shape:', d['model'].coef_.shape)
"
```

**Expected:** keys `['transformer', 'model']`; transformer has a learned `median_days_`; model coefficients shape `(1, 43)` matching the transformer's output width.

**Result:** ✓ 2026-05-22 — Keys: `['transformer', 'model']`; Transformer learned median: 827.5; Model coef shape: (1, 43).

### 4. Test suite still green

```bash
pytest tests/
```

**Expected:** 14 passed. The tests cover FeatureTransformer's and load_bookings's contracts in isolation; they pass *despite* the bug in train.py because the bug is at orchestration, not in those components.

**Result:** ✓ 2026-05-22 — `14 passed in 1.24s`. **This is the killer talking point: the test suite is green and a leakage bug ships to production. The 'component correct, integration wrong' failure mode in its canonical form.**

### 5. MLflow run inspection

```bash
mlflow ui --backend-store-uri file:./mlruns
# Then open http://localhost:5000
```

Visible: the `rebooking` experiment with one run, params (`C=1.0`, `max_iter=200`, `class_weight=None`, `test_size=0.2`), metrics (`train_auc`, `test_auc`, `pos_rate`), tags (`git_sha`, `dataset_rows`, `model_type`), and the logged sklearn model.

**Result:** Not auto-verified (requires interactive browser). Inspect manually as needed; the mlruns/ tree is regenerable from a fresh `python -m rebooking.models.train`.

---

## `src/rebooking/api/main.py` (Item 12)

FastAPI app loading `artifacts/model.joblib` at startup via lifespan, exposing `/health` and `/predict`.

### 1. Server starts cleanly

```bash
uvicorn rebooking.api.main:app --port 8765
```

**Expected:** uvicorn binds port 8765, logs `Application startup complete`, then `Uvicorn running on http://127.0.0.1:8765`.

**Result:** ✓ 2026-05-22 — clean startup; lifespan loaded the joblib bundle without error.

### 2. `GET /health` returns model metadata

```bash
curl -s http://127.0.0.1:8765/health
```

**Expected:** 200; JSON with `status: "ok"`, `model_version`, `trained_at` (ISO 8601 UTC).

**Result:** ✓ 2026-05-22 —
```json
{
  "status": "ok",
  "model_version": "0.1.0",
  "trained_at": "2026-05-22T11:37:57.654795+00:00"
}
```

### 3. `POST /predict` with valid payload returns full prediction response

```bash
curl -s -X POST http://127.0.0.1:8765/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 9999, "age": 42, "destination": "BCN",
    "booking_channel": "app", "budget": 1500.0,
    "days_since_last_booking": 200, "booking_month": 6,
    "device_type": "mobile", "loyalty_tier": "gold",
    "prior_complaints": 0, "party_size": 2
  }'
```

**Expected:** 200; response contains `customer_id` (echoed), `rebook_probability` (0..1), `rebook_predicted` (bool at threshold 0.5), `model_version`, `trained_at`, `served_at`.

**Result:** ✓ 2026-05-22 —
```json
{
  "customer_id": 9999,
  "rebook_probability": 0.8309235346352168,
  "rebook_predicted": true,
  "model_version": "0.1.0",
  "trained_at": "2026-05-22T11:37:57.654795+00:00",
  "served_at": "2026-05-22T11:44:11.442992+00:00"
}
```

### 4. `POST /predict` with null `days_since_last_booking` survives

```bash
curl -s -X POST http://127.0.0.1:8765/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 9998, "age": 60, "destination": "MAD",
    "booking_channel": "agent", "budget": 3000.0,
    "days_since_last_booking": null, "booking_month": 12,
    "device_type": "desktop", "loyalty_tier": "platinum",
    "prior_complaints": 1, "party_size": 4
  }'
```

**Expected:** 200; null in `days_since_last_booking` is imputed by the FeatureTransformer's `median_days_`; prediction returns normally.

**Result:** ✓ 2026-05-22 — `rebook_probability: 0.8307`, `rebook_predicted: true`. Null survived as designed.

### 5. `POST /predict` with invalid `booking_channel` returns 422

```bash
curl -s -X POST http://127.0.0.1:8765/predict \
  -H "Content-Type: application/json" \
  -d '{... "booking_channel": "carrier_pigeon", ...}'
```

**Expected:** HTTP 422; Pydantic returns a structured error naming the bad field and listing the allowed Literal values.

**Result:** ✓ 2026-05-22 — HTTP 422 with:
```json
{
  "detail": [{
    "type": "literal_error",
    "loc": ["body", "booking_channel"],
    "msg": "Input should be 'web', 'app', 'agent' or 'phone'",
    "input": "carrier_pigeon"
  }]
}
```

This is Pydantic doing its job: the request body never reaches the transformer because the schema validation rejects it first. Defensible interview claim: "the typed request schema is the first line of defense; the model never sees malformed input."

### 6. Test suite still green

```bash
pytest tests/
```

**Result:** ✓ 2026-05-22 — `14 passed in 1.27s`. No regression from adding the API layer.

---

## `Dockerfile` + `.dockerignore` (Item 13)

Multi-stage Docker build for the serving image. Stage 1 (builder) installs the package + deps into `/opt/venv`. Stage 2 (runtime) copies just the venv + runtime payload (`src/`, `configs/`, `artifacts/`), drops into a non-root `app` user, and runs uvicorn on port 8000.

### 1. Build validation — DEFERRED

```bash
docker build -t rebooking:0.1.0 .
```

**Expected:** two stages build cleanly; image tagged as `rebooking:0.1.0`; size in the ~300-500 MB range (python:3.11-slim + pandas + sklearn dominate).

**Result:** ✘ NOT AUTO-VERIFIED 2026-05-22 — Docker Desktop is installed on Windows but its WSL integration is not active in this distro (`docker` command not on PATH). Build validation is deferred to the user; the Dockerfile follows standard patterns and should work, but until `docker build` runs, this is unverified. Activating Docker Desktop's WSL integration is a Settings → Resources → WSL integration toggle.

### 2. Run validation — DEFERRED (depends on #1)

```bash
docker run --rm -p 8000:8000 rebooking:0.1.0
# In another shell:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @sample_payload.json
```

**Expected:** container starts, `/health` returns 200, `/predict` works identically to the uvicorn-direct smoke tests in the Item 12 section.

**Result:** ✘ NOT AUTO-VERIFIED — depends on build validation succeeding first.

### 3. .dockerignore prevents context bloat

`.dockerignore` excludes `venv/`, `mlruns/`, `data/`, `tests/`, `dsm-docs/`, `.claude/`, `__pycache__/`, IDE files, and the Dockerfile itself from the build context. Notably **does not** exclude `README.md` because `pyproject.toml`'s `readme = "README.md"` field requires it at `pip install` time.

**Manual sanity check (no Docker needed):**
```bash
# Approximation: what would the build context look like?
# Files git tracks minus .dockerignore patterns.
git ls-files | grep -vE '^(tests/|dsm-docs/|_reference/|\.gitignore$|\.claude/)' | head -20
```

**Result:** ✓ 2026-05-22 — list shows pyproject.toml, README.md, src/**, configs/training.yaml, Dockerfile, .dockerignore, .github/, scripts/. No tests/, dsm-docs/, _reference/. Expected payload.

### 4. Image layer design — talking-point check

The Dockerfile separates builder (deps + package install) from runtime (no build tooling, no pip cache). Standard multi-stage pattern. The runtime layer ordering puts `apt-get install curl` before `COPY` so source changes don't bust the apt layer.

**Known optimization deferred:** dep install + source copy share a layer (`COPY src ./src && pip install .`). Pure best practice is to extract dependencies to a separate file and install them before copying source, so source-only changes don't trigger a full reinstall. The current Dockerfile comments this trade-off explicitly. Worth mentioning in an interview if asked about caching: "I know the layering isn't optimal for source-only rebuilds; the next refactor is to extract deps to requirements.txt and install before COPY src."
