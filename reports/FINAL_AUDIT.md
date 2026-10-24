# Final Independent Audit

Auditor: independent review agent (no implementation role in this project).
Date: 2026-10-25. Scope: algorithm math, test credibility, benchmark data
consistency, licensing, architecture rules, first-time-user experience.

Method: the auditor re-ran the full test suite (51/51 PASS), independently
recomputed every benchmark summary from the raw CSV files, and reviewed the
core math term by term (quaternion conventions, SO(3) log error, DLS update,
RRT-Connect tree-swap path assembly).

## Verdict: APPROVE (after fixes below applied and re-verified)

Original verdict: APPROVE-WITH-FIXES (0 P0, 2 P1, 8 P2). All findings were
addressed in the same session and the suite re-run green (51/51, ruff clean).

## Verified claims

| Claim | Auditor's independent check | Result |
|---|---|---|
| IK 100% on 120 valid reachable targets, ≤5.0 mm / ≤2.8°, 0 false positives | recomputed from `ik_benchmark.csv` — identical numbers; unreachable targets genuinely beyond the 0.855 m reach | VERIFIED |
| RRT-Connect 100% on 3 scenes; 0 unsafe accepted paths | recomputed: 87/87 RRT-Connect successes; 178 produced paths, 0 failed independent re-validation (261 trials total) | VERIFIED (wording tightened in README) |
| Grasp 35/35 correct verdicts | recomputed from `grasp_trials.csv`; success = FSM verdict AND physical post-check; impossible targets counted correct only on failure | VERIFIED |
| 51 tests passing | re-ran pytest: 51/51, 0 skipped | VERIFIED |

## Findings and dispositions

### P1 (fixed)

1. **Two unit tests could pass vacuously** (`test_mid_segment_penetration_caught`,
   `test_gripper_cube_contact_whitelist`): their IK setups hit an early
   `return` under the current seeds, so the headline assertions never ran.
   FIXED: setups are now unconditional hard asserts, using setup-probe-verified
   poses and the production grasp geometry (`grasp_pose`/`pregrasp_pose`);
   the whitelist test now mirrors the FSM's two-stage (pregrasp→grasp) solve.
2. **UI "Move to solution" bypassed collision checking.** FIXED: `IKWorker`
   now solves with a collision `validate` callback, so an IK solution that
   would place the arm inside the table/obstacles is reported as a failure
   with an explicit reason, matching the planning pipeline.

### P2 (fixed)

1. TEST_REPORT unit-test count arithmetic corrected (37, not 28; total 51).
2. `IKSolver.solve` restart semantics aligned with the docstring
   (q_init + home + n_restarts random seeds).
3. `Trajectory.constraint_violation` docstring now states the grid-resolution
   caveat honestly (near-tight, not a strict upper bound for arbitrary
   durations).
4. `_Tree.add` raises a clear `RuntimeError` on capacity exhaustion instead
   of a raw `IndexError`.
5. `AGENTS.md` shared-interface line corrected to the real
   `GraspFSM(engine, spec)` constructor.
6. `grasp_summary.json` impossible-trial metric renamed to
   `correct_rejection_rate` to avoid misreading as task success.
7. Empty `tests/regression/` directory removed.
8. Dead `CollisionChecker.set_allowed()` removed (the only theoretical
   whitelist loophole; the one production relaxation — cube-rest pairs
   during RETREAT planning — is scoped in try/finally and physically
   justified).

## Auditor's structural conclusions (kept verbatim in substance)

- Math correct: quaternion (wxyz) handling, Shepperd branch picking, SO(3)
  error with near-π fallback, DLS update formula — checked term by term;
  Jacobian world-frame convention pinned by finite differences.
- RRT-Connect tree-swap path assembly correct (root tracking, duplicate
  meet-node dropped).
- No exception is ever swallowed into success; every failure funnels to
  explicit `_fail`/`raise`.
- Grasp verdicts are physical MuJoCo measurements (≥2 fingertip contacts,
  cube rise, settle velocity) — not tautologies.
- Architecture rules hold: zero Qt imports in core algorithm modules; no
  core math in the UI layer.
- Licensing accurate: bundled Panda model Apache-2.0 with license file;
  THIRD_PARTY_NOTICES.md matches reality; .gitignore correct; git tree clean.

## Remaining known limitations (documented, accepted)

- Windows-first tooling (`.cmd` scripts); Linux/macOS users must use
  `pip install -e .` manually.
- The desktop app requires a display; headless users are limited to the
  test/benchmark suite (which runs anywhere MuJoCo does).
- `impossible` grasp-trial semantics require the renamed key (now fixed).
- IK success is tolerance-defined (5 mm / 2.9°) — stated in BENCHMARKS.md.
