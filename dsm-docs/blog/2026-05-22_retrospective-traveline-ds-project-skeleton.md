# Building a Pair-Coding Demo: 14 Items, 1 Bug, 1 Day

**Status:** Seed (outline only; flesh out before publishing)
**Source material:** `_reference/preliminary-plan.md` (archived), `dsm-docs/plans/Sprint_1_skeleton_build.md`, `dsm-docs/feedback-to-dsm/2026-05-22_project-retrospective.md`, `dsm-docs/guides/smoke-tests.md`, `dsm-docs/guides/transform-fix.py`, `_reference/bugs-to-find.md` (private)
**Target length:** 1800–2400 words
**Audience:** mid-career data scientists, ML engineers, hiring managers evaluating portfolio repos

## Hook

A live pair-coding interview is announced for tomorrow at 14:30 CEST. You have one working day to ship a Python data-science project skeleton that demonstrates how you structure a repo, hosts live coding, and gives the interviewer a planted bug to find. What do you build, and what do you intentionally leave out?

## Narrative arc

1. **Constraints first.** ~15 files, < 500 lines, < 4 hours, every line defensible. Why the deliberately narrow scope produces a better artifact than a maximal one.
2. **The execution order isn't the menu order.** Why "data → loader → transformer → tests → model → orchestrator" beats "files-by-folder-listing." Sample-data generator promoted to a foundation step; config extraction demoted to "after duplication exists."
3. **The bug that didn't bite.** First attempt at planting a data-leakage bug produced zero measurable inflation. Why: with 25 destinations and 1600 training rows, the OneHotEncoder vocabulary saturated. Fix: shrink the dataset + add a target-encoded feature where leakage actually has somewhere to go. Final result: realistic 1.7 pt mean test-AUC inflation across 20 splits — which IS the lesson.
4. **"Tests pass, leakage ships."** 14 unit tests green, the data leakage still ships to production. The transformer is correct in isolation; the bug lives at the orchestration level. Canonical "component correct, integration wrong" failure mode in its most defensible form.
5. **The Pipeline answer you don't dodge.** Why I wrote a custom FeatureTransformer class instead of sklearn.Pipeline + ColumnTransformer + TargetEncoder. The honest concession: the strongest argument FOR Pipeline is exactly the bug I planted. Volunteer the counter-argument.
6. **Push hygiene.** Pre-validating CI checks locally before publishing the workflow. Why the first push to a fresh CI pipeline is the worst place to discover accumulated lint debt. (Caught 7 ruff errors that would have lit up the first PR.) Verifying main contains what you think it does before going public.
7. **Realistic over dramatic.** The metric I tried hardest to inflate ended up small and consistent. Why that's the more defensible interview claim ("real-world leakage looks like 1-3 pts of inflation, which is what makes it dangerous") than a contrived 30-pt textbook example.

## Talking-point map

Map sections above to the interview questions they pre-emptively answer:
- "How do you structure a DS project repo?" → §1
- "Walk me through your training pipeline." → §2
- "Spot the bug." → §3 + §4
- "Why didn't you use sklearn.Pipeline?" → §5
- "Walk me through your CI." → §6
- "What's the most common bug you've seen in DS code?" → §3 + §7

## Closing

The repo is at https://github.com/albertodiazdurana/traveline-ds-project-skeleton. It's MIT-licensed, branch-protected, CI-green. The README has a "Hidden challenge" section inviting readers to find the bug; the fix lives at `dsm-docs/guides/transform-fix.py`. Try it. The 1.7-point inflation is the entry point; the conversation about why it's not 30 points is the demo.

---

## Outline notes (not for publication)

- Lead with the constraint, not the result. The interesting thing is the artifact-design decisions, not the artifact.
- Use code snippets sparingly. Most of the value is in the reasoning around the code.
- Cite McConnell's *Code Complete* for the smoke-test framing. Cite Karpathy's *Recipe for Training Neural Networks* for the engineering-logbook framing.
- Don't link to `_reference/bugs-to-find.md` (gitignored answer key). DO link to `dsm-docs/guides/transform-fix.py` (the worked answer that's public).
- Length budget: ~1800-2400 words. Closer to 1800 if I cut §6 to a paragraph; closer to 2400 if I do justice to the demo-data-design discussion in §3.
- Honest framing: do not over-claim novelty. The smoke-test pattern is well-established under several names; my contribution is naming + locating it inside a methodology, not inventing it.
