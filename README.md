# Rebooking

Customer rebooking prediction skeleton — minimal Python data-science project demonstrating src-layout, modern packaging, MLflow tracking, FastAPI serving, and a multi-stage Dockerfile.

Binary classification: given customer demographics, destination, dates, budget, and acquisition channel, predict whether the customer will rebook within 12 months.

## Stack

Python 3.11, scikit-learn, pandas, MLflow, FastAPI + Pydantic, pytest, ruff, mypy, GitHub Actions, Docker.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
# Generate synthetic sample data
python scripts/make_sample_data.py

# Train (logs to mlruns/)
python -m rebooking.models.train

# Serve
uvicorn rebooking.api.main:app --reload

# Test
pytest tests/
```

## Hidden challenge

This skeleton contains **one intentional bug** — a common, subtle data-science pitfall. Try to find it by inspecting the code.

Hint: it produces test-set metrics that look better than they should.

If you'd like to compare your finding against a worked production-shape refactor (with structural prevention of the bug, plus `cross_val_score` and `GridSearchCV` properly wired), see the guides directory: [`dsm-docs/guides/`](dsm-docs/guides/).

## Status

Skeleton in progress. See [`dsm-docs/plans/Sprint_1_skeleton_build.md`](dsm-docs/plans/Sprint_1_skeleton_build.md) for the build order.
