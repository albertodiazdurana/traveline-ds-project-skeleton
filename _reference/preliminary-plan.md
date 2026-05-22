# Traveline DS Project Skeleton , Preliminary Plan

> **ARCHIVED (2026-05-22).** This file was the initial menu used to seed the build. It is no longer the source of truth. The active source of truth is [`dsm-docs/plans/Sprint_1_skeleton_build.md`](../dsm-docs/plans/Sprint_1_skeleton_build.md), which captures the actual build order, item numbering, and completion state.
>
> Keep this file as historical reference for the rationale behind each item (the "Interview hook" paragraphs are the most reusable part). Do not edit it to reflect mid-build decisions; record those in the Sprint 1 plan instead.

**Purpose:** A demonstrable Data Science project skeleton for the Traveline R3-2 Live Pair Coding interview (Fri 2026-05-22 14:30 CEST). Used during the call to (a) show "how I structure a DS project repo" when asked, (b) host live coding exercises, (c) provide a debugging substrate.

**Stack (confirmed):** Python + pip + venv. FastAPI for the API layer. sklearn for modeling. MLflow for tracking. pytest for testing. ruff + mypy for quality. GitHub Actions for CI. Docker for containerization.

**Domain:** Customer rebooking prediction (binary classification , inspired by Yurii's R3-1 scenario: predict whether a customer will rebook within 12 months given demographics, destination, dates, budget, channel).

**Principle being applied:** Take AI Bite , one item at a time, user-generated, agent supports per-item. The plan below is the **menu**, not the meal. You generate each item; I help on demand.

**Objective & Scope:** Build a minimal-but-complete Python data-science project skeleton , loader, feature transformer, training script with MLflow tracking, FastAPI serving layer, unit tests, multi-stage Dockerfile, GitHub Actions CI , wrapped around a fictional customer-rebooking prediction problem. Scope is deliberately narrow (~15 files, under 500 lines total, buildable in 3-4 hours) so the artifact is a teaching/demo piece focused on production-engineering practices (src-layout, type hints, dependency management, testing pyramid, train-serve parity), not a production system. Every line should be defensible in a live conversation.

---

## Open decisions before starting

1. **Planted bug for the debugging exercise** , one bug (data leakage in `transform.py`, classic interview gold) or two (data leakage + mutable-default-arg)? Default suggestion: ONE (focused, easier to demo). Answer: ONE. The bug will be a subtle data leakage issue in the `FeatureTransformer` where the `fit()` method is called on the full dataset before the train/test split, instead of fitting only on the training data. This is a common mistake that can lead to overly optimistic performance metrics, and it's a great talking point for the interview.
2. **Git init** , should the skeleton be its own git repo (`git init`)? Default suggestion: YES (shows in `git log`, more realistic). Answer: YES, we will initialize a git repository for the project skeleton. This allows us to demonstrate good commit practices and makes the project feel more real. We can also use the commit history to show the incremental building of the project during the interview.
3. **`bugs-to-find.md`** , do you want me to maintain a private answer-key file (excluded via .gitignore) listing the planted bugs and their fixes, so you can quiz yourself before tomorrow? Answer: YES, we will create a private `bugs-to-find.md` file under the `_reference/` directory. This file will list each planted bug, its location, why it's a bug, and how to fix it. This will be for our personal rehearsal and won't be part of the public repo or the interview discussion.

---

## Build order (rationale)

The order below matches the natural data-flow of an ML project AND the natural order you would explain it in a live interview: **foundation → data → features → models → tests → serving → containerization → automation**.

Each item lists:
- **Path** , exact location relative to repo root
- **What** , one-sentence description
- **Purpose** , why it exists (interview talking-point value)
- **Scope budget** , how much to write so you don't over-engineer
- **Depends on** , which prior items must exist
- **Interview hook** , the question this item lets you answer demonstrably

---

## Item 1 , `.gitignore`

**Path:** `.gitignore`
**What:** Git's ignore-list for venv, `__pycache__`, MLflow runs, raw data, IDE files.
**Purpose:** Prevents committing local-only artifacts. First file because all later commits depend on it.
**Scope budget:** ~15 lines.
**Depends on:** nothing.
**Interview hook:** "How do you keep credentials and large files out of git?" Answer: "I use a `.gitignore` file to specify patterns for files and directories that should not be tracked by git. This typically includes the virtual environment directory (e.g., `venv/`), `__pycache__/` directories, MLflow run artifacts, raw data files, and IDE-specific files. This way, I ensure that sensitive information and large files are not accidentally committed to the repository."

---

## Item 2 , `pyproject.toml`

**Path:** `pyproject.toml`
**What:** Project metadata + dependencies, declares the package installable via `pip install -e .`.
**Purpose:** The single source-of-truth for dependencies and package configuration. Demonstrates modern Python packaging (PEP 621).
**Scope budget:** ~40 lines. Sections: `[project]` (name, version, deps), `[project.optional-dependencies]` (dev: pytest, ruff, mypy), `[build-system]` (setuptools or hatchling), `[tool.ruff]`, `[tool.mypy]`.
**Depends on:** nothing.
**Interview hook:** "How do you manage dependencies?" "How is your project installable?" Answer: "I use a `pyproject.toml` file to manage dependencies and package configuration. This file serves as the single source of truth for the project. It includes the project metadata, the list of dependencies required for installation, and optional dependencies for development. By running `pip install -e .`, we can install the package in editable mode, which is great for development. This modern approach to Python packaging follows PEP 621 standards and is widely adopted in the Python community."

---

## Item 3 , `README.md`

**Path:** `README.md`
**What:** Short markdown explaining purpose, stack, how to install, how to run tests, how to start the API.
**Purpose:** First thing an interviewer would open if they cloned this. Signals you care about onboarding others.
**Scope budget:** ~50 lines. Sections: Overview, Stack, Quickstart (3 commands max), Repository Layout (the directory tree), Testing, Running the API.
**Depends on:** rest of the plan exists conceptually (you write the README as if the rest already exists).
**Interview hook:** Implicit , a good README is a credibility signal.

---

## Item 4 , `src/rebooking/__init__.py`

**Path:** `src/rebooking/__init__.py`
**What:** Empty file that marks `rebooking/` as a Python package.
**Purpose:** Enables `from rebooking.data import loader` imports.
**Scope budget:** 0 lines (empty file) or 1 line (`__version__ = "0.1.0"`).
**Depends on:** Item 2 (pyproject.toml references this package name).
**Interview hook:** "Why `src/` layout?" (talking-point: avoids accidental imports from cwd, tests use real import paths). Answer: "The `src/` layout is a best practice in Python packaging that helps prevent accidental imports from the current working directory. By placing our code inside a `src/` directory, we ensure that when we run tests or the API, they import the package as it would be installed, rather than accidentally importing from the source files in the root. This leads to more reliable testing and a clearer structure." What does src stand for? "Source". It's a convention that indicates this directory contains the source code of the package. It also helps to separate the package code from other files in the repository, such as tests, configs, and documentation.

---

## Item 5 , `data/sample.csv`

**Path:** `data/sample.csv`
**What:** ~20 rows of synthetic customer/booking data with columns: customer_id, age, destination, travel_date, budget, booking_channel, rebooked_within_12mo.
**Purpose:** Lets every subsequent piece (loader, tests, train, API smoke test) run end-to-end without needing real data.
**Scope budget:** 20-30 rows, ~7 columns. Include a few nulls and a few obvious outliers so tests can catch them.
**Depends on:** nothing (but schema decisions cascade to Items 6, 7, 9).
**Interview hook:** "How do you handle test data?" (talking-point: small, real-shape, version-controlled). Answer: "I create synthetic test data that mimics the structure and characteristics of real data, but is small enough to be manageable and version-controlled."

---

## Item 6 , `src/rebooking/data/loader.py`

**Path:** `src/rebooking/data/loader.py`
**What:** Function `load_bookings(path: Path) -> pd.DataFrame` that reads the CSV and validates the schema (column presence, dtypes, no all-null columns).
**Purpose:** Data ingress layer with schema validation as the first defensive boundary. Demonstrates type hints, error handling, single-responsibility.
**Scope budget:** ~30 lines. One function, one schema constant, raise `ValueError` with clear message on mismatch.
**Depends on:** Item 5 (data schema).
**Interview hook:** "How do you validate data at ingestion?" Answer: "I write a data loader function that reads the CSV and checks for the expected schema. This includes verifying that all required columns are present, that they have the correct data types, and that there are no columns that are entirely null. If any of these checks fail, I raise a ValueError with a clear message indicating what the issue is. This way, we catch data issues early in the pipeline."

---

## Item 7 , `src/rebooking/features/transform.py`

**Path:** `src/rebooking/features/transform.py`
**What:** Functions to transform raw bookings into model features: encode categorical (booking_channel, destination), derive features (days_since_last_booking, budget_quartile), scale numerics.
**Purpose:** The shared transformation layer used by both train AND serve. This is THE artifact that prevents training-serving skew.
**Scope budget:** ~40 lines. One class `FeatureTransformer` with `fit(X)`, `transform(X)`, `fit_transform(X)`.
**Planted bug (if planting one):** fit-on-full-data-before-split (data leakage). The bug should look subtle but be discoverable in code review.
**Depends on:** Item 6.
**Interview hook:** "How do you avoid training-serving skew?" Answer: We can avoid training-serving skew by implementing a shared transformation layer that is used both during training and serving. This typically involves creating a class (e.g., `FeatureTransformer`) that has methods for fitting on the training data and transforming both the training and test data. By ensuring that the same transformations are applied in both contexts, we can maintain consistency and prevent discrepancies between how features are processed during training and at serve time. 
 + "Spot the bug." (the killer talking-point) + "How do you handle unseen categories at serve time?" Answer: "The `FeatureTransformer` is designed to be fitted only on the training data, and then used to transform both train and test. This way, we avoid data leakage. If we fit it on the full dataset before splitting, we would be leaking information from the test set into the training process, which can lead to overly optimistic performance metrics. To handle unseen categories at serve time, we can use an encoder that has an 'unknown' category or simply ignore unseen categories, depending on the use case."

---

## Item 8 , `tests/unit/test_loader.py`

**Path:** `tests/unit/test_loader.py`
**What:** pytest tests for `load_bookings`: happy path, missing column raises, wrong dtype raises, empty file raises.
**Purpose:** Demonstrates testing the data ingress boundary. Showcases pytest fixtures and parametrize.
**Scope budget:** ~40 lines, 4-5 tests.
**Depends on:** Items 5, 6.
**Interview hook:** "How do you test data loaders?" Answer: "I write pytest tests that cover the expected schema. For example, I test the happy path with a well-formed CSV, and then I write tests that check for missing columns, wrong data types, or an empty file. Each of these should raise a ValueError with a clear message. I use pytest fixtures to create temporary CSV files for testing, and I can use parametrize to test multiple schema violation cases in a clean way."

---

## Item 9 , `tests/unit/test_transform.py`

**Path:** `tests/unit/test_transform.py`
**What:** pytest tests for `FeatureTransformer`: shape preservation, no nulls after transform, encoder fitted on train doesn't crash on unseen categories.
**Purpose:** Showcases model-behavior testing (invariants, not just happy path).
**Scope budget:** ~50 lines, 5-6 tests.
**Depends on:** Items 5, 7.
**Interview hook:** "How do you test feature transformations?" + "What if the API gets an unseen category at serve time?" Answer: "We can test that the encoder is fitted only on the training data and that it can handle unseen categories gracefully, either by ignoring them or mapping them to an 'unknown' category. This way, we prevent the API from crashing when it encounters new data. Related to invariant testing, we can check that the shape of the data is preserved after transformation, that there are no null values in the output, and that the expected columns are present. These tests ensure that our feature transformation behaves as expected under various conditions."

---

## Item 10 , `configs/training.yaml`

**Path:** `configs/training.yaml`
**What:** YAML config with hyperparameters: model type, learning rate, regularization strength, train/test split, random seed, MLflow tracking URI.
**Purpose:** No-hardcoded-values pattern. Lets you reproduce a run by checking in the config.
**Scope budget:** ~20 lines.
**Depends on:** nothing structurally, but Item 11 reads it.
**Interview hook:** "How do you manage hyperparameters?" Answer: "I use a YAML config file that contains all the hyperparameters and settings for training. This way, I avoid hardcoding values in the code, and I can easily reproduce runs by checking in the config file. For example, I have parameters for the model type, learning rate, regularization strength, train/test split ratio, random seed for reproducibility, and the MLflow tracking URI."

---

## Item 11 , `src/rebooking/models/train.py`

**Path:** `src/rebooking/models/train.py`
**What:** `train()` function: load data → split → fit transformer on TRAIN ONLY → fit sklearn LogisticRegression → evaluate on test → log params, metrics, and model to MLflow → return artifact path.
**Purpose:** The training entry point. Demonstrates MLflow integration, train/test discipline, and the canonical sklearn pipeline.
**Scope budget:** ~60 lines. CLI entry via `if __name__ == "__main__"` reading the YAML.
**Depends on:** Items 6, 7, 10.
**Interview hook:** "Walk me through your training script." + MLflow patterns + the regularization parameter from the config. Answer: "The training script starts by loading the data using our loader, then splits it into train and test sets. It fits the `FeatureTransformer` only on the training data to avoid data leakage, then transforms both train and test. Next, it fits a LogisticRegression model on the transformed training data. the regularization parameter is read from the config LogisticRegression(C=config["C"]). After training, it evaluates the model on the test set and logs the parameters, metrics, and the model itself to MLflow for tracking and reproducibility."

---

## Item 12 , `src/rebooking/api/main.py`

**Path:** `src/rebooking/api/main.py`
**What:** FastAPI app with `/health` (returns OK) and `/predict` (Pydantic model in, prediction out). Loads the latest registered MLflow model at startup.
**Purpose:** The serving layer. Demonstrates Pydantic validation, async/sync trade-offs, MLflow model loading at serve time.
**Scope budget:** ~50 lines. Two endpoints, one Pydantic input class, one Pydantic output class, startup hook to load model.
**Depends on:** Items 7, 11 (and an MLflow run must have happened).
**Interview hook:** "How do you serve the model?" + "How does the API know which model version to load?" Answer: "In this simple example, it loads the latest registered model from MLflow at startup. In a more complex setup, we could specify the model version in a config or use MLflow's model registry to manage staging/production versions."

---

## Item 13 , `Dockerfile`

**Path:** `Dockerfile`
**What:** Multi-stage build: stage 1 installs deps and the package, stage 2 copies only what's needed and runs `uvicorn`.
**Purpose:** Production-readiness signal. Shows you know multi-stage builds reduce image size.
**Scope budget:** ~25 lines. Two `FROM` stages, EXPOSE 8000, CMD running uvicorn.
**Depends on:** Items 2, 12.
**Interview hook:** "How do you containerize?" + "Why multi-stage?" Answer: "I use a multi-stage Docker build to keep the final image small and secure. In the first stage, I install all the dependencies and the package itself. In the second stage, I copy only the necessary files (like the API code and any required configs) and set the command to run the API with uvicorn. This way, we avoid including development tools and caches in the final image, which reduces its size and attack surface."

---

## Item 14 , `.github/workflows/ci.yml`

**Path:** `.github/workflows/ci.yml`
**What:** GitHub Actions workflow: on push/PR → install deps → ruff check → mypy → pytest → (optionally) build Docker image.
**Purpose:** Demonstrates CI thinking and the "block-merge-on-failure" pattern.
**Scope budget:** ~40 lines. One job, ~5 steps.
**Depends on:** Items 2, 8, 9 (something for CI to run).
**Interview hook:** "Walk me through your CI pipeline." Answer: "The CI pipeline is triggered on every push or pull request. It starts by setting up the Python environment and installing the dependencies. Then it runs ruff for linting, mypy for type checking, and pytest for running the tests. Optionally, we can also include a step to build the Docker image to ensure that it builds successfully with every change. The idea is to catch any issues early and prevent merging code that breaks the build or fails tests."

---

## Item 15 (optional) , `bugs-to-find.md`

**Path:** `_reference/bugs-to-find.md` (kept under `_reference/`, excluded from `git add` if you want to keep it private)
**What:** Answer key listing each planted bug, where it is, why it's a bug, and the fix.
**Purpose:** Self-rehearsal artifact. NOT for the interview, only for the dry-run later today.
**Scope budget:** ~30 lines.
**Depends on:** Item 7 (and any other bug-planted file).
**Interview hook:** None (private).

---

## Interview-readiness checklist (after building all items)

Before tomorrow, verify in order:

- [ ] `pip install -e ".[dev]"` from a clean venv succeeds
- [ ] `pytest tests/` passes (excluding the planted bug, or skipping it with a marker)
- [ ] `python -m rebooking.models.train --config configs/training.yaml` runs end-to-end and writes to MLflow
- [ ] `uvicorn rebooking.api.main:app --reload` starts; `curl localhost:8000/health` returns OK; `curl -X POST localhost:8000/predict -d '...'` returns a prediction
- [ ] `ruff check src/` clean
- [ ] `mypy src/` clean
- [ ] `docker build .` succeeds
- [ ] Repo opens cleanly in VS Code with no broken imports
- [ ] Mental dry-run: explain each top-level folder in <30 seconds each
- [ ] Mental dry-run: explain the train→serve data flow in <60 seconds
- [ ] Find the planted bug in `transform.py` from memory

---

## How I support the per-item generation

For each item, ping me with:
- "Item N , show me the code" → I draft, you read, modify, commit.
- "Item N , give me hints, not the answer" → I list 3-5 things the item must do, you write it.
- "Item N , review what I wrote" → you paste, I review for correctness, style, interview-talking-point gaps.

Recommended cadence: 15-20 minutes per item for code items, 5-10 minutes for config/docs items. Total elapsed time for a full build: ~3-4 hours, can be split across this session and tomorrow morning.

---

## Final note on the principle

Take AI Bite: small steps, deliberate generation, judgment-first. You build it, you understand it, you can defend it under live questioning tomorrow. If I generate it all in one shot, the artifact exists but the understanding does not transfer. The interview tests the transfer.
