# Project Retrospective: traveline-ds-project-skeleton

**Date:** 2026-05-22
**Duration:** 2026-05-21 (bootstrap) to 2026-05-22 (final wrap-up + finalize), 1 working day
**Sessions:** 1 (continuous, ~12 hours including breaks)
**Project type:** Spoke (interview-demo / teaching artifact)

## Project summary

Built a minimal-but-complete Python data-science project skeleton from green-field empty to a runnable end-to-end pipeline: synthetic-data generator (250 rows × 12 cols with 25-destination heavy-tailed signal), schema-validated loader, FeatureTransformer (one-hot + StandardScaler + median imputation + cyclic-month + target-encoded destination), training orchestrator with Pydantic-typed config and MLflow logging, FastAPI serving layer with typed Pydantic request/response, multi-stage Dockerfile, and GitHub Actions CI. Includes one intentional data-leakage bug at the training-orchestration level (fit_transform on full data before split, ~1.7 pt mean test-AUC inflation across 20 splits) as a code-review demo substrate. The artifact is interview-ready for a Live Pair Coding session on 2026-05-22 at 14:30 CEST.

## DSM effectiveness across the project

- **Tracks used:** DSM 1.0 (planning + sprint structure) + DSM 4.0 (software engineering adaptation: src-layout, modern packaging, lint/type/test triad, multi-stage Dockerfile, CI). Some DSM 2.0 (sprint plan structure, Sprint Boundary Checklist). No 3.0 / 5.0 / 6.0 usage.
- **Overall methodology score:** High. DSM provided the structural scaffolding (sprint plan, smoke-tests log, feedback-to-DSM channel, ecosystem registry, reasoning-lessons distillation) without slowing down the build. The 14-item plan was followed in execution order (not menu order), and the per-item smoke-test discipline meant every commit was preceded by a documented validation.
- **Strongest contributions:** The Sprint plan as living source of truth (vs the preliminary plan as archived menu) was the right pattern. The smoke-tests.md log produced a defensible per-item validation narrative that the git log + CI status alone couldn't deliver.
- **Weakest moment:** Silent item renumbering between preliminary plan and Sprint plan (caught at Item 8) cost one turn to untangle. The fix (Sprint plan as authority, preliminary archived) was clean but should have been the framing from the start.

## Phase-level assessment

| Phase | What worked | What was missing |
|-------|------------|-----------------|
| Research/Grounding | Preliminary plan loaded from `_reference/`; stack confirmed; planted-bug strategy specified up front. No deep-dive research was needed (well-bounded scope). | n/a |
| Planning | Sprint plan with MUST/SHOULD/COULD + 7 phases + Sprint Boundary Checklist. Execution order deliberately diverged from menu order (bottom-up by data dependency, tests interleaved). | Initial line-count budgets pressured incompleteness — dropped mid-build with an explicit policy note. |
| Implementation | Item-at-a-time commits; smoke test after each item; CI checks pre-validated locally before publishing the workflow; honest A/B measurement of the planted bug rather than contriving a flashier number. | One pre-publish lint gate would have caught the 7 accumulated ruff errors earlier. |
| Communication | README with stack/install/run/Hidden challenge/Repo structure; transform-fix.py as the worked answer to the planted bug; smoke-tests.md as the reproducible validation log. | None significant. |

## Cross-project transferability

**Patterns that should inform other projects:**
- **Per-item smoke-test log** (`dsm-docs/guides/smoke-tests.md`) pairing exact commands with expected + actual output. Already filed as a BL to DSM Central proposing it as a named artifact ("smoke tests" per McConnell / SRE runbook precedent). Particularly useful for scaffolding / teaching builds where test infrastructure is itself being built.
- **Sprint plan as source of truth + preliminary plan archived.** Avoids the silent-renumbering trap where two plan documents diverge.
- **Pre-publish CI gate.** Run the workflow's own checks locally before committing the workflow file. The first push to a fresh CI pipeline is the worst place to discover accumulated lint debt.
- **Honest demo signal.** When a teaching bug doesn't manifest dramatically, the realistic signal IS the lesson. "Real-world leakage looks like 1-3 pts of inflation, which is what makes it dangerous" is a stronger interview claim than a contrived 30-point inflation.
- **Volunteer the killer counter-argument.** For design-choice questions, leading with the strongest counter-argument to your own decision earns trust ("Pipeline would have structurally prevented the bug I planted").

**Project-specific, do NOT generalize:**
- Specific data shape (250 rows × 12 cols, 25 destinations, target-encoded destination) is tuned for one specific bug to be visible. Other projects need their own tuning.
- README's "Hidden challenge" section is appropriate for a portfolio demo, not for production projects.
- 1-session timeline + interview deadline are project-specific constraints. Don't generalize the time pressure into a general "ship fast" rule.

## Recommendations for DSM

- **BL filed during session 1:** `dsm-docs/feedback-to-dsm/done/2026-05-22_s1_backlogs.md` proposes adopting "smoke tests" as a named DSM artifact in `dsm-docs/guides/`, with a per-entry (command, expected, result) structure. Cites McConnell / Microsoft daily-build / Google SRE runbook precedent. General scope (not narrowed to demo projects after user pushback).
- **Reasoning lessons pushed:** 10 entries (2 ecosystem, 6 pattern, 2 project) via session-1 wrap-up notification. Top candidates for ecosystem-wide adoption: hardcoded scope budgets pressure incompleteness; name what already exists in industry rather than coining new terms.
- **STAA recommended:** session involved complex multi-option decisions and recurring patterns worth deeper analysis. Specifically: planning-fidelity (the silent renumbering case) and demo-design-honesty (the leakage-signal retune across 3 iterations) generalize across teaching/scaffolding projects.

## Outcomes

- Public repo at https://github.com/albertodiazdurana/traveline-ds-project-skeleton with MIT license, 10 topics, branch protection (PR + CI green required), CI badge in README.
- All 14 Sprint 1 MUST items shipped. SHOULD items (test_serving.py, README expansion, Makefile) deferred without urgency.
- Test suite: 14 passed. Lint (ruff): clean. Type-check (mypy): clean. CI green on both PR runs.
- Interview-ready as of 2026-05-22 evening, ~14 hours before the live pair-coding session.
