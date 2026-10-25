# Changelog

## 1.6.0 — 2026-10-25

### Fixed
- Scene JSON round-trip produced specs that compared UNEQUAL to their
  originals (JSON arrays -> lists vs dataclass tuples). `from_dict` now
  normalizes every sequence field; 12 regression tests pin save/load/build
  equality across all six presets.

### Verification
- Second 10-pass audit with fresh dimensions (install resolution, launch
  smoke, offline capability, JSON round-trip, dist contents, git health,
  error-handling sweep, units discipline, README structure, full physics
  regression) - all green.

## 1.5.2 — 2026-10-25

### Release-blocking fix (found by the 10-pass release audit)
- pyproject.toml declared `PySide6` while the project installs
  `PySide6-Essentials`: `pip check` reported broken requirements and a
  fresh install would pull the full PySide6. Corrected to
  `PySide6-Essentials>=6.6`; `pip check` clean.

### Audit
- 10-pass release verification executed (tests/lint, demo claim chain,
  packaged self-test, 7-preset scene cycling, doc links + image assets,
  benchmark claims recomputed from raw CSVs, architecture separation,
  TOPP-RA dominance + planner determinism re-verified live, release
  config, reproducibility contract). All green; one probe-script bug on
  our side was triaged and confirmed NOT a product defect (the planner
  correctly rejected an out-of-limits goal).

## 1.5.1 — 2026-10-25

### Documentation positioning
- Purpose headers on the three process files (AGENTS.md, TASKS.md,
  PORTFOLIO.md) so a first-time reviewer understands why they exist and how
  they relate to the contributor contract.

## 1.5.0 — 2026-10-25

### Project quality signals
- docs/REPRODUCIBILITY.md: an index mapping every benchmark number to its
  re-run command and raw data file, plus the determinism contract (what is
  seed-exact vs wall-clock variable).
- .gitattributes: deterministic line endings across platforms, binary
  markers for meshes/images.

## 1.4.1 — 2026-10-25

### Documentation integrity
- Fixed six chart links in reports/BENCHMARKS.md (wrong relative path -
  none of the benchmark charts were rendering from the report page).
- README badges normalized to fork-safe static shields.

## 1.4.0 — 2026-10-25

### Open-source readiness
- Community kit: CONTRIBUTING.md (project rules for contributors), SECURITY
  policy, Code of Conduct, CITATION.cff, GitHub issue/PR templates.
- `requirements.txt` flexible install path for non-Windows reviewers
  (ranges) alongside the locked benchmark environment.
- README: relative CI badge (fork-friendly), Linux quickstart promoted to a
  first-class block.

## 1.3.1 — 2026-10-25

### Repository hygiene (open-source readiness)
- Removed unused vendored menagerie variants (MJX models, standalone
  hand.xml, demo scene.xml, preview image) and a leftover parameter-probe
  scene file; vendored subset documented in THIRD_PARTY_NOTICES.md.
- Module-level debug-import tidy-up in toppra.py.

## 1.3.0 — 2026-10-25

### Closed-loop system
- Scripted moving obstacle (mocap panel) + GuardedExecutor: swept-volume
  collision semantics shared by planner and runtime guard, dual
  reference/physical-arm guarding with a 1.2 s horizon, cycle-verified park
  configuration with guarded retreats, and an online replan loop.
- ``moving_wall`` scene preset; UI auto-wraps guarded execution;
  ``python -m robotstudio.demo --moving`` headless proof.
- Replan benchmark vs an unguarded baseline (2 speeds x 5 trials).

### Hardening
- CollisionChecker rewritten clean (mode semantics + carried-cube transform
  + swept moving obstacle in one file).

## 1.2.0 — 2026-10-25

### Algorithms
- TOPP-RA time-optimal path parameterization (Pham & Pham 2018, greedy
  controllable-set variant): arc-length parameterization of the executed
  spline geometry, exact per-interval acceleration constraints, pointwise
  curvature bound, dense profile resampling and an explicit dominance
  guarantee (never slower than the conservative allocation). Measured: -58%
  median execution time on long paths; 0% (guarantee holds) where the
  conservative allocation is already near-optimal.
- Branded PandaForge (package import name unchanged: robotstudio).

### Product
- Headless proof-of-work demo: `python -m robotstudio.demo` prints physical
  evidence for every claim without a display.
- CI now runs the full suite on windows-latest AND ubuntu-latest.
- README restructured: CI/license badges, 60-second verification block,
  attribution table above the fold, 5-minute code tour, demo GIF with live
  FSM stage captions.

### Critical fixes
- Union scene layout: every preset XML now contains every body
  (position-hidden when disabled). Fixes engine construction on 'empty'
  (crash) and silently half-applied preset switching - MuJoCo bakes
  contype=0 geoms out of the collision pipeline at compile time, so runtime
  flag flips cannot re-enable them (verified experimentally).

### Algorithms
- Antipodal grasp candidate generation (top + 4 side grasps, preference
  ordered, declarative collision-free selection) with automatic side-grasp
  fallback when the top approach is blocked.
- Break-away support-pair discovery from the live contact list (correct
  lift-off semantics for grasps resting on any support, not just the table).

### Scenes & demo
- New ``roof_grasp`` preset: shelf-to-shelf transfer under a roof that
  blocks top-down grasping.
- README demo GIF (real simulation frames), generated by
  ``scripts/make_demo_gif.py``.

### Tests / benchmarks
- 66 tests (was 64): roof side-grasp end-to-end + candidate-generation
  properties; grasp benchmark adds the ``roof_side`` category.

## 1.1.0 — 2026-10-25

### Algorithms
- RRT\* (k-nearest, Karaman & Frazzoli 2011) and Informed RRT\*
  (Gammell et al. 2014) with per-iteration cost history and raw-vs-shortcut
  path reporting; quality/convergence benchmark + charts.
- Tree storage upgraded with incremental rewiring (cost propagation).

### Visualization & UX
- 3D view draws the planned path (green capsules), the live end-effector
  trace (cyan) and grasp contact points (red) via the viewer user scene.
- Toolbar: one-click Showcase (obstacle sweep + full pick & place),
  E-STOP, Reset, Pause; global typography and QSS polish.
- Engine protocol fixes found by the new tests: controllers may opt into
  surviving scene switches; e-stop notifies controllers via callback.

### Tests
- 64 tests (was 55): +5 RRT\* properties (incl. exact ellipsoid-membership
  check), +4 showcase/decoration tests.

## 1.0.0 — 2026-10-25

First complete release (P0 + P1 scope).

### Product
- Interactive desktop app (PySide6 control panel + MuJoCo passive viewer):
  scene presets, obstacle editing, joint control, IK, planning, execution,
  automatic pick & place, live metrics, logs.
- 4 preset scenes (empty / tabletop / single obstacle / narrow passage),
  JSON scene save & load, in-place scene switching.
- Franka Panda (mujoco_menagerie f3b6013, Apache-2.0) vendored offline.

### Algorithms (independent implementations)
- DLS IK: adaptive damping, SO(3) log error, nullspace bias, multi-start,
  collision validation.
- Whitelist collision checker: dense segment checks, mode-dependent grasp /
  rest pairs, carried-object transform, shadow MjData thread isolation.
- RRT, RRT-Connect, linear baseline behind one API; randomized shortcutting
  with independent re-validation; vectorized nearest-neighbour (3.2x).
- Time-scaled cubic-spline trajectory generation with numeric limit
  verification; settle-based execution completion; e-stop semantics.
- 15-state pick&place FSM with physical success criteria, per-state
  timeouts, retry and abort.

### Quality
- 51 automated tests (unit / integration / physics / GUI) all passing.
- Benchmarks with committed raw data: IK 100% (120 targets), planning 261
  trials with 0 invalid outputs, grasp 35/35 correct verdicts.
- setup/run/test/build.cmd for Windows; PyInstaller one-folder package with
  headless self-test; GitHub Actions CI (tests + build artifact).

### Docs
- README (EN), ARCHITECTURE, docs/ALGORITHMS, docs/DECISIONS,
  docs/RISK_REGISTER, reports/TEST_REPORT, reports/BENCHMARKS, PORTFOLIO,
  THIRD_PARTY_NOTICES, TASKS register.

### Known issues
- FSM planning blocks the engine thread briefly (viewer pause, GUI fine).
- RRT single-tree success on narrow scenes is budget-limited (~47% @ 8 s).
- Offscreen Qt (CI) cannot capture real fonts in screenshots; screenshots in
  docs/img are captured on a real Windows session.
