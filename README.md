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
python -m rebooking.training.train

# Serve
uvicorn rebooking.serving.app:app --reload

# Test
pytest tests/
```

## Status

Skeleton in progress. See [`dsm-docs/plans/Sprint_1_skeleton_build.md`](dsm-docs/plans/Sprint_1_skeleton_build.md) for the build order.
