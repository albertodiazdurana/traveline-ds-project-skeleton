# Sprint 1: Traveline DS Project Skeleton Build

**Duration:** 1 session (target 3-4 hours wall-clock)
**Goal:** Deliver a minimal-but-complete Python DS project skeleton (customer-rebooking prediction) usable as a substrate for the Traveline R3-2 Live Pair Coding interview on **2026-05-22 14:30 CEST**.
**Prerequisites:** Git initialized, `.gitignore` committed, DSM scaffold committed, `_reference/preliminary-plan.md` as the menu, `_reference/bugs-to-find.md` as the private answer key.

## Research Assessment

Scope is fully defined in `_reference/preliminary-plan.md` (26 items, each with path, scope budget, and interview hook). The user has already confirmed:
- Stack: Python + pip + venv, FastAPI, sklearn, MLflow, pytest, ruff + mypy, GitHub Actions, Docker.
- Domain: customer rebooking prediction (binary classification).
- ONE planted bug: data leakage in `FeatureTransformer.fit()` called pre-split.
- Two commits already on `session-1/2026-05-21`: `.gitignore`, scaffold.

No unresolved unknowns. No targeted research needed.

## Experiment Gate

**Performance-only sprint** (no new capability gating production behavior). The project is a teaching/demo artifact, not a production system. Experiment skip justified: every "feature" is a single-purpose scaffolding file with a defined scope budget and a defensible interview hook; the artifact is judged by demo-readiness, not by an empirical success metric on held-out data.

## Branch Strategy

Single session, single deliverable. Work commits directly to the session branch `session-1/2026-05-21`. No Level 3 branches needed since:
- No methodology files being modified (this is a spoke, not Central).
- No command files being modified.
- All 26 items are mechanical scaffolding writes, each one its own commit.

**Commit cadence:** one commit per build item (Item N → "Add Item N — {file}: {one-liner}"). The git log itself becomes part of the demo (incremental build narrative).

---

## Deliverables

Items are numbered to match `_reference/preliminary-plan.md`. Read that file for path/scope-budget/interview-hook detail per item.

### MUST (15 items — sprint fails without these)

Foundation:
- [x] **Item 1** — `.gitignore` *(committed `5c95661`)*
- [ ] **Item 2** — `pyproject.toml` (deps + ruff + mypy config, ~40 lines)
- [ ] **Item 3** — `README.md` (project overview + run instructions, ~50 lines)

Package skeleton:
- [ ] **Item 4** — `src/rebooking/__init__.py` (empty init, src-layout marker)
- [ ] **Item 5** — `src/rebooking/config.py` (config loader + model hyperparams, ~25 lines)

Data layer:
- [ ] **Item 6** — `src/rebooking/data/load.py` (CSV/parquet loader with schema validation, ~30 lines)

Features (bug planted here):
- [ ] **Item 7** — `src/rebooking/features/transform.py` (FeatureTransformer with fit/transform/fit_transform, ~40 lines) — **planted bug lives here**

Model + training:
- [ ] **Item 8** — `src/rebooking/models/baseline.py` (LogisticRegression wrapper, ~20 lines)
- [ ] **Item 10** — `src/rebooking/training/train.py` (orchestrator: load → split → fit → eval → MLflow log, ~50 lines)

Serving:
- [ ] **Item 11** — `src/rebooking/serving/app.py` (FastAPI app with `/predict` + `/health`, ~40 lines)
- [ ] **Item 12** — `src/rebooking/serving/schemas.py` (Pydantic request/response models, ~20 lines)

Tests:
- [ ] **Item 9** — `tests/unit/test_transform.py` (shape, no-nulls, unseen-category, ~30 lines)
- [ ] **Item 13** — `tests/unit/test_serving.py` (FastAPI TestClient: /health 200, /predict shape, ~25 lines)

Container + CI:
- [ ] **Item 14** — `Dockerfile` (multi-stage: builder → runtime, ~25 lines)
- [ ] **Item 16** — `.github/workflows/ci.yml` (lint + type-check + test on PR, ~30 lines)

### SHOULD (defer if blocked)
- [ ] **Item 17** — Sample synthetic data generator script (`scripts/make_sample_data.py`, ~25 lines) for self-contained demos
- [ ] **Item 18** — `Makefile` (or `justfile`) with `install`, `train`, `serve`, `test`, `lint` targets

### COULD (stretch — only if interview prep slack remains)
- [ ] **Item 19+** — Anything else from the preliminary plan beyond the 16 above (review menu the morning of)

---

## Phases

### Phase 1: Foundation (Items 1-3)
- **Focus:** Repo files an interviewer expects to see when they `ls` the project root.
- **Deliverables:** `.gitignore` (done), `pyproject.toml`, `README.md`.
- **Execution mode:** script (file writes + small commits).
- **DSM references:** DSM_4.0 software engineering adaptation (src-layout, modern packaging).
- **Success criteria:** `pip install -e .` works in a clean venv; ruff + mypy invocable from `pyproject.toml` config; README answers "what is this and how do I run it" in <60 seconds of reading.

### Phase 2: Package skeleton (Items 4-5)
- **Focus:** Importable `rebooking` package + config layer.
- **Deliverables:** `src/rebooking/__init__.py`, `src/rebooking/config.py`.
- **Execution mode:** script.
- **Success criteria:** `python -c "import rebooking"` succeeds inside the editable install; config loader returns a typed dict / dataclass with `C`, `random_state`, paths.

### Phase 3: Data → features → model pipeline (Items 6-8, 10)
- **Focus:** The DS substance. This is where the planted bug lives.
- **Deliverables:** `load.py`, `transform.py` (bug planted), `baseline.py`, `train.py`.
- **Execution mode:** script.
- **Success criteria:** `python -m rebooking.training.train` runs end-to-end on synthetic data, logs to MLflow `mlruns/`, prints train + test AUC. The bug should be reproducible (test AUC suspiciously close to train AUC).

### Phase 4: Serving (Items 11-12)
- **Focus:** Train-serve parity demonstration via the FastAPI layer.
- **Deliverables:** `schemas.py`, `app.py`.
- **Execution mode:** script + manual smoke (uvicorn + curl).
- **Success criteria:** `uvicorn rebooking.serving.app:app` starts; `curl localhost:8000/health` returns 200; `POST /predict` with a valid Pydantic body returns a probability.

### Phase 5: Tests (Items 9, 13)
- **Focus:** Demonstrate testing pyramid (unit on transformer, light integration via FastAPI TestClient).
- **Deliverables:** `test_transform.py`, `test_serving.py`.
- **Execution mode:** script + pytest.
- **Success criteria:** `pytest tests/` passes. Tests should be defensible — invariant-style assertions (shape, no-null, unseen-category survival) not just "function returns something."

### Phase 6: Container + CI (Items 14, 16)
- **Focus:** Production-engineering polish.
- **Deliverables:** `Dockerfile` (multi-stage), `.github/workflows/ci.yml`.
- **Execution mode:** script.
- **Success criteria:** `docker build .` succeeds (skip docker run if time-pressed); GitHub Actions YAML parses (`gh workflow view` or local act) — actual CI run depends on remote being added.

### Phase 7 (optional): SHOULD items + final dress rehearsal
- **Focus:** Round-out items if time permits + rehearsal pass.
- **Deliverables:** Sample data generator, Makefile/justfile, README polish.
- **Success criteria:** User can demo each Phase 1-6 deliverable in <30 seconds without consulting notes.

---

## Sprint Boundary Checklist

- [ ] Checkpoint document created in `dsm-docs/checkpoints/`
- [ ] Feedback files updated (if any DSM-process lessons surfaced)
- [ ] Decision log updated with sprint decisions (e.g., stack choice rationale if questioned in interview)
- [ ] Tests passing (`pytest tests/` green)
- [ ] Blog journal entry written (optional for interview-prep sprint; skip with justification)
- [ ] Blog publication tracker updated (N/A — no public blog for this private prep project)
- [ ] Repository README updated (status, structure, how-to-run)
- [ ] Next steps summary: post-interview retrospective; possibly Sprint 2 to fix the planted bug deliberately if the repo becomes a portfolio piece

---

## Risks & mitigations

- **Risk:** Scope creep — temptation to add more files than the plan calls for.
  **Mitigation:** Plan caps at 16 MUST items; reject additions during build.
- **Risk:** Bug discoverable only after running training — synthetic data might mask leakage signal.
  **Mitigation:** Item 17 (sample data) should produce a dataset where leakage is visible (e.g., enough categorical cardinality + small enough test split that leakage shifts the metrics by >2 points).
- **Risk:** Time pressure — interview at 14:30 CEST tomorrow; not all 16 items may finish in one session.
  **Mitigation:** Hard ordering by phase. Phases 1-4 are critical; Phase 5 (tests) and Phase 6 (CI/Docker) can be partially deferred and built live during the call if needed (also a talking point — "we test as we go").

## Defensive talking points (interview rehearsal hooks)

Each MUST item has an interview hook documented in `_reference/preliminary-plan.md`. Rehearse one hook per item during the build by reading it aloud after each commit.
