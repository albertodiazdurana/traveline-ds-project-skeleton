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

**Note on scope budgets (revised 2026-05-22, mid-build):** the preliminary plan attached `~N lines` budgets to each item. Those budgets pressured toward incompleteness in practice (`make_sample_data.py` came in at 115 vs a 25-line "budget" and the extra lines were all load-bearing). The right constraint is **"every line is defensible in an interview"**, not a line count. Remaining-item budgets have been stripped from the list below. The historical line counts on already-built items are kept as a record.

**Note on item numbering (revised 2026-05-22, after Item 9):** this file is the source of truth for the build. Item numbers below happen to match `_reference/preliminary-plan.md` (now archived) for the items that came from it, but the preliminary plan is no longer authoritative — if a future change diverges from it, this plan wins. The list below is what we follow.

One deviation from the preliminary menu that we kept: Item 5 is implemented as `scripts/make_sample_data.py` (gitignored generator) rather than a checked-in `data/sample.csv`; defensible upgrade for reproducibility. There is no separate `models/baseline.py` file — the LogisticRegression instantiation lives inside `models/train.py` (Item 11) directly.

### MUST (14 items — sprint fails without these)

Foundation:
- [x] **Item 1** — `.gitignore` *(committed `5c95661`)*
- [x] **Item 2** — `pyproject.toml` *(committed `4d90b92`)* — PEP 621, hatchling backend, ruff + mypy inline; 44 lines. pytest config added in `f18a91f`
- [x] **Item 3** — `README.md` stub *(committed `8125d6c`)*; "Hidden challenge" section added in `a7b7364`. Expansion deferred

End-to-end runnable pipeline (hardcoded values where reasonable):
- [x] **Item 4** — `src/rebooking/__init__.py` *(committed `7374aea`)* — `__version__ = "0.1.0"`; install validated
- [x] **Item 5** — `scripts/make_sample_data.py` *(committed `115f56a`, retuned `d9ec057`)* — generates `data/raw/bookings.csv` (gitignored); 250 rows × 12 cols. *Deviation from preliminary plan: generator script in `scripts/` rather than a checked-in CSV in `data/`*
- [x] **Item 6** — `src/rebooking/data/loader.py` *(committed `2937c0a`)* — `load_bookings()` with strict-column + lenient-dtype schema validation
- [x] **Item 7** — `src/rebooking/features/transform.py` *(committed `e5ce20b`; typecheck fix `56a6e77`; target encoding `d9ec057`)* — OneHotEncoder (handle_unknown='ignore'), StandardScaler, median imputation, cyclic booking_month, target-encoded destination
- [x] **Item 8** — `tests/unit/test_loader.py` *(committed below)* — 6 pytest cases covering loader contract + 3 ValueError branches
- [x] **Item 9** — `tests/unit/test_transform.py` *(committed `f18a91f`)* — 8 pytest cases covering FeatureTransformer's isolated contract
- [x] **Item 10** — `configs/training.yaml` *(committed below)* — three sections (data, model, mlflow); PyYAML added as direct dep in `pyproject.toml`; Pydantic settings model will land in Item 11
- [x] **Item 11** — `src/rebooking/models/train.py` *(committed below)* — orchestrator with Pydantic-typed config; load → fit_transform on full data (**bug**) → split → fit LogisticRegression → eval → MLflow log → joblib dump. Measured: +1.2pt test-AUC inflation on seed=42, tests still green

Serving:
- [ ] **Item 12** — `src/rebooking/api/main.py` — FastAPI app with `/predict` + `/health`, Pydantic request/response models, loads the joblib-serialized transformer + model

Container + CI:
- [ ] **Item 13** — `Dockerfile` — multi-stage (builder + runtime), python:3.11-slim base, non-root user, layered for caching
- [ ] **Item 14** — `.github/workflows/ci.yml` — lint (ruff) + type-check (mypy) + test (pytest) on PR

### Built but not in preliminary MUST list
- [x] **Item 15 (optional)** — `_reference/bugs-to-find.md` *(gitignored)* — private answer key; rehearsal talking points added in retune cycle
- [x] **Extras** — `dsm-docs/guides/transform-fix.py` (production refactor reference, committed `8370d10`), `dsm-docs/guides/smoke-tests.md` (rolling smoke-test log, committed `c3033ea`), `dsm-docs/feedback-to-dsm/2026-05-22_s1_backlogs.md` (BL to DSM Central, committed `6378980`)

### SHOULD (defer if blocked)
- [ ] **Item 3 expansion** — Flesh out `README.md` with structure overview, talking-point map, run examples
- [ ] **test_serving.py** — FastAPI TestClient unit tests (preliminary plan doesn't include this; consider adding when Item 12 lands)
- [ ] **Makefile** (or `justfile`) with `install`, `train`, `serve`, `test`, `lint` targets

### COULD (stretch — only if interview prep slack remains)
- [ ] Anything else from the preliminary plan beyond the items above (review menu the morning of)

---

## Phases

### Phase 1: Foundation (Items 1, 2, 3-stub)
- **Focus:** Repo files an interviewer expects to see when they `ls` the project root. README is a stub at this stage — it gets fleshed out in the polish phase once the code it describes actually exists.
- **Deliverables:** `.gitignore` (done), `pyproject.toml`, `README.md` (stub).
- **Execution mode:** script (file writes + small commits).
- **DSM references:** DSM_4.0 software engineering adaptation (src-layout, modern packaging).
- **Success criteria:** `pip install -e .` works in a clean venv; ruff + mypy invocable from `pyproject.toml` config; README stub answers "what is this and how do I run it" in two paragraphs.

### Phase 2: Make something runnable end-to-end (Items 4, 5, 6, 7, 9, 8, 10, 11)
- **Focus:** Bottom-up by data dependency. Generate sample data first so everything downstream has a concrete schema to bind to. Interleave each test with the code it covers. The planted bug is introduced at the very end (`models/train.py`, Item 11), not earlier.
- **Deliverables (in execution order):**
  1. `src/rebooking/__init__.py` (Item 4) — make the package importable
  2. `scripts/make_sample_data.py` (Item 5) — synthetic CSV generator; schema anchor for everything downstream
  3. `src/rebooking/data/loader.py` (Item 6) — read + validate against that schema
  4. `src/rebooking/features/transform.py` (Item 7) — `FeatureTransformer` with correct fit/transform semantics
  5. `tests/unit/test_transform.py` (Item 9) — invariant tests on the transformer (shape, no-nulls, unseen-category, fit-on-train-only, target encoding contract)
  6. `tests/unit/test_loader.py` (Item 8) — happy path + missing/extra column + null target + column order invariant + path-vs-str
  7. `configs/training.yaml` (Item 10) — declarative hyperparams that `train.py` reads
  8. `src/rebooking/models/train.py` (Item 11) — orchestrator; **planted bug introduced here** (fit-on-full-data-before-split at the orchestration level). LogisticRegression is instantiated inline; there is no separate `models/baseline.py`.
- **Execution mode:** script with frequent runs (`python scripts/make_sample_data.py`, `pytest tests/`, then `python -m rebooking.models.train` once `train.py` lands).
- **Success criteria:** `python -m rebooking.models.train` runs end-to-end on the generated sample CSV, logs to `mlruns/`, prints train + test AUC. The bug should be **observable as a consistent positive inflation** of test AUC across random splits (measured: mean +1.7 pts, max +5.4 pts on the retuned dataset; this is realistic-leakage scale, not textbook-30pts scale — see `dsm-docs/guides/smoke-tests.md` Retune log). `pytest tests/` is green (the bug is at orchestration, not in the transformer or loader). The production refactor at `dsm-docs/guides/transform-fix.py` demonstrates how `sklearn.Pipeline` structurally prevents the bug.

### Phase 3: Serving (Item 12)
- **Focus:** Train-serve parity via FastAPI. Pydantic schemas, the endpoint surface, and a TestClient suite live together in `api/main.py` (preliminary plan does not split them into separate files).
- **Deliverables:**
  - `src/rebooking/api/main.py` (Item 12) — FastAPI app with `/predict` + `/health`; loads the joblib-serialized transformer + model
  - (SHOULD) `tests/unit/test_serving.py` — FastAPI TestClient: `/health` 200, `/predict` returns probability with correct shape, validation error on bad payload
- **Execution mode:** script + manual smoke (`uvicorn rebooking.api.main:app` + curl) + pytest.
- **Success criteria:** `uvicorn rebooking.api.main:app` starts; `curl localhost:8000/health` returns 200; `POST /predict` with a valid body returns a probability; if test_serving.py exists, `pytest tests/` remains green.

### Phase 4: Container + CI (Items 13, 14)
- **Focus:** Production-engineering polish.
- **Deliverables:** `Dockerfile` (Item 13, multi-stage), `.github/workflows/ci.yml` (Item 14).
- **Execution mode:** script.
- **Success criteria:** `docker build .` succeeds (skip `docker run` if time-pressed); GitHub Actions YAML parses (`gh workflow view` or local `act`) — actual CI run depends on remote being added.

### Phase 5: Polish (SHOULD items if time)
- **Focus:** README expansion + Makefile/justfile + rehearsal pass.
- **Deliverables:** `README.md` (full version with structure overview + talking-point map), Makefile/justfile.
- **Success criteria:** User can demo each Phase 1-4 deliverable in <30 seconds without consulting notes; README's "How to run" section is copy-pasteable into a terminal.

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
