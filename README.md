# Rebooking

[![CI](https://github.com/albertodiazdurana/traveline-ds-project-skeleton/actions/workflows/ci.yml/badge.svg)](https://github.com/albertodiazdurana/traveline-ds-project-skeleton/actions/workflows/ci.yml)

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

## Repo structure

```
src/rebooking/      Package code
  ├── data/         Loader (schema-validated CSV ingestion)
  ├── features/     FeatureTransformer (encoding + scaling + target encoding)
  ├── models/       Training orchestrator (load, fit, log to MLflow, serialize)
  └── api/          FastAPI app (/health, /predict)

configs/            Declarative training configs (YAML)
scripts/            One-off scripts (synthetic data generator)
tests/unit/         pytest suite (transformer + loader contracts)
artifacts/          Serialized trained model (gitignored; produced by training)
mlruns/             MLflow tracking store (gitignored)
data/raw/           Generated sample CSV (gitignored)

dsm-docs/           Methodology artifacts (sprint plan, smoke tests, BL feedback)
  └── guides/       Smoke-test log + production-shape refactor reference
_reference/         Project rationale (preliminary plan, archived)

Dockerfile          Multi-stage serving image
.github/workflows/  CI (ruff + mypy + pytest on push/PR)
pyproject.toml      Deps, build backend, tool config
```

## Status

**Sprint 1 complete (14/14 items).** Delivered: synthetic data generator, schema-validated loader, FeatureTransformer with target encoding, training orchestrator with MLflow logging and joblib serialization, FastAPI serving layer with typed Pydantic models, multi-stage Dockerfile, GitHub Actions CI. Local checks green: `ruff check .`, `mypy src`, `pytest tests/` (14 passed).

See [`dsm-docs/plans/Sprint_1_skeleton_build.md`](dsm-docs/plans/Sprint_1_skeleton_build.md) for the build order with commit references and [`dsm-docs/guides/smoke-tests.md`](dsm-docs/guides/smoke-tests.md) for the per-item validation log.
