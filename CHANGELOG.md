# Changelog

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
