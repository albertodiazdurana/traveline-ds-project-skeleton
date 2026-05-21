# Smoke Tests

Per-item smoke tests run against each file as it's built. A smoke test is a small, fast check that confirms the basic thing still works — the term originates from electrical hardware ("plug it in and see if it smokes") and was popularized for software by Microsoft's daily-build-and-smoke-test practice (McConnell, *Code Complete* Ch. 22). Each entry below pairs the exact command(s) executed against the user's `.venv/` with the expected output and the actual result. New items are appended as the build progresses.

**Why smoke-test per item:** Each file is exercised end-to-end by a human before moving to the next item. This catches integration mismatches that unit tests miss (wrong filename in an import, schema drift between modules, hardcoded path that only works in CI) and surfaces them at the moment they're introduced, when the relevant code is still in working memory. It also produces a defensible interview / demo narrative: each commit corresponds to an item that was *seen working* before being declared done.

**Format per entry:** subsection per file, numbered subsections per check, each block contains: command → expected → result. Mark each result `✓ <ISO date>` or `✗` with the diagnostic.

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
