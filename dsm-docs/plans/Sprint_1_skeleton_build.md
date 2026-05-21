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

**Ordering principle:** the list below is **execution order**, not the menu order from `_reference/preliminary-plan.md`. Item numbers reference the preliminary-plan menu and are deliberately out of sequence here, because a human DS engineer:
1. Writes things bottom-up by data dependency (sample data → loader → transformer → model → train), not menu order.
2. Interleaves each test with the code it covers, not in a final test pass.
3. Defers `config.py` until duplication exists (avoid premature abstraction).
4. Promotes sample data to a foundation step because nothing downstream can be exercised without it.

### MUST (15 items — sprint fails without these)

Foundation:
- [x] **Item 1** — `.gitignore` *(committed `5c95661`)*
- [x] **Item 2** — `pyproject.toml` *(committed `4d90b92`)* — PEP 621, hatchling backend, ruff + mypy inline; 44 lines
- [x] **Item 3** — `README.md` stub *(committed `8125d6c`)* — 37 lines; expansion deferred to Phase 6

End-to-end runnable pipeline (hardcoded values where reasonable):
- [x] **Item 4** — `src/rebooking/__init__.py` *(committed `7374aea`)* — `__version__ = "0.1.0"`; `pip install -e ".[dev]"` validated end-to-end in user's `.venv/`
- [x] **Item 17** — `scripts/make_sample_data.py` *(committed `115f56a`)* — 1000 rows × 12 cols; extended schema + tuned target for visible leakage; over 25-line budget (115 lines) by design
- [x] **Item 6** — `src/rebooking/data/loader.py` *(committed below)* — `load_bookings()` with strict-column + lenient-dtype schema validation; 43 lines (filename aligned with preliminary plan; was `load.py` in this plan)
- [x] **Item 7** — `src/rebooking/features/transform.py` *(committed `e5ce20b`)* — 42 lines; OneHotEncoder (handle_unknown='ignore'), StandardScaler, median imputation, cyclic booking_month. Validated end-to-end including unseen-category survival
- [ ] **Item 9** — `tests/unit/test_transform.py` *(interleaved)* — shape, no-nulls, unseen-category, fit-on-train-only invariant (~30 lines)
- [ ] **Item 8** — `src/rebooking/models/baseline.py` (LogisticRegression wrapper, ~20 lines)
- [ ] **Item 10** — `src/rebooking/training/train.py` — orchestrator: load → split → fit → eval → MLflow log (~50 lines). **Planted bug lives here** (fit-before-split in the orchestration, not inside `transform.py`).

Extract config once duplication exists:
- [ ] **Item 5** — `src/rebooking/config.py` *(demoted; extract after `train.py` reveals what to extract)* — config loader + hyperparams (~25 lines)

Serving:
- [ ] **Item 12** — `src/rebooking/serving/schemas.py` *(before app.py)* — Pydantic request/response models as the contract (~20 lines)
- [ ] **Item 11** — `src/rebooking/serving/app.py` — FastAPI app with `/predict` + `/health` (~40 lines)
- [ ] **Item 13** — `tests/unit/test_serving.py` *(interleaved)* — TestClient: `/health` 200, `/predict` shape (~25 lines)

Container + CI:
- [ ] **Item 14** — `Dockerfile` (multi-stage: builder → runtime, ~25 lines)
- [ ] **Item 16** — `.github/workflows/ci.yml` (lint + type-check + test on PR, ~30 lines)

### SHOULD (defer if blocked)
- [ ] **Item 3 expansion** — Flesh out `README.md` with structure overview, talking-point map, run examples
- [ ] **Item 18** — `Makefile` (or `justfile`) with `install`, `train`, `serve`, `test`, `lint` targets

### COULD (stretch — only if interview prep slack remains)
- [ ] **Item 19+** — Anything else from the preliminary plan beyond the items above (review menu the morning of)

---

## Phases

### Phase 1: Foundation (Items 1, 2, 3-stub)
- **Focus:** Repo files an interviewer expects to see when they `ls` the project root. README is a stub at this stage — it gets fleshed out in Phase 6 once the code it describes actually exists.
- **Deliverables:** `.gitignore` (done), `pyproject.toml`, `README.md` (stub).
- **Execution mode:** script (file writes + small commits).
- **DSM references:** DSM_4.0 software engineering adaptation (src-layout, modern packaging).
- **Success criteria:** `pip install -e .` works in a clean venv; ruff + mypy invocable from `pyproject.toml` config; README stub answers "what is this and how do I run it" in two paragraphs.

### Phase 2: Make something runnable end-to-end (Items 4, 17, 6, 7, 9, 8, 10)
- **Focus:** Bottom-up by data dependency. Generate sample data first so everything downstream has a concrete schema to bind to. Interleave the transform test with the transformer itself. The planted bug is introduced at the very end (`train.py`), not earlier.
- **Deliverables (in execution order):**
  1. `src/rebooking/__init__.py` — make the package importable
  2. `scripts/make_sample_data.py` — synthetic CSV; schema anchor for everything downstream
  3. `src/rebooking/data/load.py` — read + validate against that schema
  4. `src/rebooking/features/transform.py` — `FeatureTransformer` with correct fit/transform semantics
  5. `tests/unit/test_transform.py` — invariant tests on the transformer (shape, no-nulls, unseen-category, fit-on-train-only)
  6. `src/rebooking/models/baseline.py` — LogisticRegression wrapper
  7. `src/rebooking/training/train.py` — orchestrator; **planted bug introduced here** (fit-on-full-data-before-split at the orchestration level)
- **Execution mode:** script with frequent runs (`python scripts/make_sample_data.py`, then `python -m rebooking.training.train` once `train.py` lands).
- **Success criteria:** `python -m rebooking.training.train` runs end-to-end on the generated sample CSV, logs to `mlruns/`, prints train + test AUC. The bug should be reproducible: test AUC suspiciously close to train AUC (gap < ~2 points on the synthetic data). `pytest tests/unit/test_transform.py` passes (because the bug is at orchestration, not in the transformer's own behavior).

### Phase 3: Extract config (Item 5)
- **Focus:** Pull hardcoded values out of `train.py` into `config.py` only once duplication exists. This is a deliberate "extract on the second use" move — defensible in the interview as a YAGNI choice.
- **Deliverables:** `src/rebooking/config.py`; `train.py` updated to read from it.
- **Execution mode:** script + re-run training to confirm no behavior change.
- **Success criteria:** `train.py` no longer contains hardcoded `C`, `random_state`, `test_size`, or path constants; behavior unchanged.

### Phase 4: Serving (Items 12, 11, 13)
- **Focus:** Train-serve parity via FastAPI. Pydantic schemas first (the contract), then the app that implements it, then the test against the contract.
- **Deliverables (in execution order):**
  1. `src/rebooking/serving/schemas.py` — Pydantic request/response models
  2. `src/rebooking/serving/app.py` — FastAPI app with `/predict` + `/health`
  3. `tests/unit/test_serving.py` — TestClient: `/health` 200, `/predict` shape
- **Execution mode:** script + manual smoke (`uvicorn rebooking.serving.app:app` + curl) + pytest.
- **Success criteria:** `uvicorn rebooking.serving.app:app` starts; `curl localhost:8000/health` returns 200; `POST /predict` with a valid body returns a probability; `pytest tests/unit/test_serving.py` passes.

### Phase 5: Container + CI (Items 14, 16)
- **Focus:** Production-engineering polish.
- **Deliverables:** `Dockerfile` (multi-stage), `.github/workflows/ci.yml`.
- **Execution mode:** script.
- **Success criteria:** `docker build .` succeeds (skip `docker run` if time-pressed); GitHub Actions YAML parses (`gh workflow view` or local `act`) — actual CI run depends on remote being added.

### Phase 6: Polish (SHOULD items if time)
- **Focus:** README expansion + Makefile/justfile + rehearsal pass.
- **Deliverables:** `README.md` (full version with structure overview + talking-point map), Makefile/justfile.
- **Success criteria:** User can demo each Phase 1-5 deliverable in <30 seconds without consulting notes; README's "How to run" section is copy-pasteable into a terminal.

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
  **Mitigation:** Plan caps at 15 MUST items; reject additions during build.
- **Risk:** Bug discoverable only after running training — synthetic data might mask leakage signal.
  **Mitigation:** `scripts/make_sample_data.py` (Item 17, now in Phase 2) is generated *before* the transformer/trainer exists, so it can be tuned to produce a dataset where leakage is visible — enough categorical cardinality + small enough test split that leakage shifts the metrics by >2 points. If the gap isn't visible after the first training run, tune the sample-data generator before continuing.
- **Risk:** Time pressure — interview today at 14:30 CEST; not all items may finish in one session.
  **Mitigation:** Hard ordering by phase. Phases 1-4 are critical; Phase 5 (Docker/CI) can be partially deferred and built live during the call if needed (also a talking point — "we set this up incrementally"). Phase 6 (README polish + Makefile) is optional.
- **Risk:** Premature config extraction (Item 5) before duplication exists wastes time and creates "designed-for-the-future" smells visible in code review.
  **Mitigation:** Item 5 moved to Phase 3, deliberately after `train.py` lands. The agent should resist the urge to write `config.py` earlier even if it "feels cleaner."

## Defensive talking points (interview rehearsal hooks)

Each MUST item has an interview hook documented in `_reference/preliminary-plan.md`. Rehearse one hook per item during the build by reading it aloud after each commit.
