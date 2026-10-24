# RobotStudio — Task Register

Status legend: `PENDING` / `IN_PROGRESS` / `DONE` / `BLOCKED` / `DROPPED`.
A task may only be marked DONE with concrete acceptance evidence (test file
or report path). This file is updated as work progresses, never pre-filled.

## Phase 0 — Environment & Baseline

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Windows env check (Python 3.11.9, pip 26.1.2, git 2.55, RTX 4070 driver 591.44) | Main | DONE | session log; `.venv` created |
| Dependencies installed (mujoco 3.17.0, numpy 2.4.6, scipy 1.17.1, PySide6-Essentials 6.12.0, pyqtgraph 0.14.0, matplotlib 3.11.2, pytest 9.1.1) | Main | DONE | `requirements-lock.txt` |
| Panda model from mujoco_menagerie @ f3b6013 vendored + license kept | Main | DONE | `assets/robots/panda/LICENSE` (Apache-2.0) |
| 4 preset scenes compile & step physics 1 s (stable rest states) | Main | DONE | smoke run 2026-10-25: empty/tabletop/single_obs/narrow OK |
| Scene XML generation pipeline (`SceneSpec` → MJCF sibling of panda.xml) | Main | DONE | `src/robotstudio/simulation/scenes.py` |

## Phase 1 — Kinematics

| Task | Owner | Status | Evidence |
|---|---|---|---|
| FK via hand body + Jacobian frame convention verified by finite differences | Kinematics | DONE | `tests/unit/test_kinematics.py` (FD match < 1e-4) |
| DLS IK (adaptive damping, nullspace bias, multi-start, limits, NaN guards) | Kinematics | DONE | `tests/unit/test_ik.py` + `reports/benchmarks/ik_benchmark.json` |
| IK acceptance: ≥95% on ≥100 FK-generated reachable poses, ≤10 mm / ≤5° | Kinematics | DONE | 100% on 120 valid samples, max 5.0 mm / 2.8°, 0 false positives (`reports/benchmarks/ik_benchmark.json`) |

## Phase 2 — Collision & Planning

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Whitelist collision checker with shadow MjData; vs-mj_forward parity test | Planning | DONE | `tests/unit/test_collision.py` |
| Mid-segment penetration detection (dense interpolation) | Planning | DONE | `tests/unit/test_collision.py::test_mid_segment_penetration_caught` |
| RRT + RRT-Connect + linear baseline, unified API | Planning | DONE | `tests/unit/test_planners.py` |
| Shortcutting with independent re-validation | Planning | DONE | `tests/unit/test_planners.py::test_shortcut_reduces_length` |
| Benchmark: 3 scenarios × ~30 seeds × 3 planners | QA | DONE | `reports/benchmarks/planning_*.csv`, `reports/BENCHMARKS.md` |

## Phase 3 — Trajectory

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Time-scaled cubic spline generator with numeric limit verification | Trajectory | DONE | `tests/unit/test_trajectory.py` |
| Executor refuses infeasible trajectories; settle-based completion criterion | Control | DONE | `tests/integration/test_execution.py` |
| Real physics execution incl. e-stop freeze / completion | Control | DONE | `tests/integration/test_execution.py::test_estop_freezes_motion` |

## Phase 4 — Grasping

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Grasp pose computation (top-down, measured pad/palm geometry) | Manipulation | DONE | `src/robotstudio/manipulation/grasp.py`, physics tests |
| 15-state FSM with timeouts / retry / abort | Manipulation | DONE | `tests/physics/test_grasp_fsm.py` |
| End-to-end physical pick&place ≥18/20 on base scenario | Manipulation | DONE | `reports/benchmarks/grasp_summary.json` (base 20/20) |

## Phase 5 — Desktop UI

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Main window + 6 panels, all buttons wired to engine commands | UI | DONE | `tests/gui/test_ui_smoke.py` (offscreen) |
| Non-blocking planning (worker threads), clean shutdown | UI | DONE | `tests/gui/test_ui_smoke.py`, `closeEvent` wiring |

## Phase 6 — QA & Benchmarks

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Full pytest suite green (51/51) + ruff clean | QA | DONE | `reports/TEST_REPORT.md`, `reports/FINAL_AUDIT.md` |
| Performance profile before/after optimizations, same seeds | QA | DONE | `reports/BENCHMARKS.md` §5 (NN vectorization 3.2x, sampling fix 75s->0.6s) |

## Phase 7 — Release

| Task | Owner | Status | Evidence |
|---|---|---|---|
| setup/run/test/build .cmd scripts, all executed | Release | DONE | session log; setup self-check OK, build self-test exit 0 |
| PyInstaller one-folder package, launch-tested (model+physics+plan+execute) | Release | DONE | `dist/RobotStudio/`, self-test exit 0 in 3.7 s |
| GitHub Actions CI (windows-latest: tests + build artifact) | Release | DONE | `.github/workflows/ci.yml` |
| Docs: README/ARCHITECTURE/ALGORITHMS/BENCHMARKS/TEST_REPORT/PORTFOLIO/DECISIONS/RISK_REGISTER/CHANGELOG/THIRD_PARTY | Release | DONE | repo root + `docs/` + `reports/` |
| Independent final audit (APPROVE after 2 P1 + 8 P2 fixed) | Reviewer | DONE | `reports/FINAL_AUDIT.md` |

## Known Issues / Decisions Log

- 2026-10-25: first pip install stalled at 0 KB/s after ~20 min (large PySide6
  wheels on flaky connection). Mitigation: killed and reinstalled in two
  batches with `--timeout 60 --retries 8`; switched to `PySide6-Essentials`
  (sufficient for QtWidgets/QtGui used by pyqtgraph).
- 2026-10-25: panda.xml already defines a `home` keyframe; scene XMLs must not
  re-define it (duplicate key name compile error). Scene generator omits it.
- 2026-10-25: single_obs wall originally intersected the arm at home pose
  (ncon=12); moved to y=-0.24 side approach blocker.
