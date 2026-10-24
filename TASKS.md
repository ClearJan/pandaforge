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
| FK via hand body + Jacobian frame convention verified by finite differences | Kinematics | PENDING | — |
| DLS IK (adaptive damping, nullspace bias, multi-start, limits, NaN guards) | Kinematics | PENDING | — |
| IK acceptance: ≥95% on ≥100 FK-generated reachable poses, ≤10 mm / ≤5° | Kinematics | PENDING | — |

## Phase 2 — Collision & Planning

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Whitelist collision checker with shadow MjData; vs-mj_forward parity test | Planning | PENDING | — |
| Mid-segment penetration detection (dense interpolation) | Planning | PENDING | — |
| RRT + RRT-Connect + linear baseline, unified API | Planning | PENDING | — |
| Shortcutting with independent re-validation | Planning | PENDING | — |
| Benchmark: 3 scenarios × 30 seeds × 3 planners | QA | PENDING | — |

## Phase 3 — Trajectory

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Time-scaled cubic spline generator with numeric limit verification | Trajectory | PENDING | — |
| Executor refuses infeasible trajectories; tracks errors | Control | PENDING | — |
| Real physics execution incl. pause / estop / completion | Control | PENDING | — |

## Phase 4 — Grasping

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Grasp pose computation (top-down, cube frame) | Manipulation | PENDING | — |
| 15-state FSM with timeouts / failure / abort | Manipulation | PENDING | — |
| End-to-end physical pick&place ≥18/20 on base scenario | Manipulation | PENDING | — |

## Phase 5 — Desktop UI

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Main window + 6 panels, all buttons wired to engine commands | UI | PENDING | — |
| Non-blocking planning (worker threads), clean shutdown | UI | PENDING | — |

## Phase 6 — QA & Benchmarks

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Full pytest suite green | QA | PENDING | — |
| Performance profile before/after optimizations, same seeds | QA | PENDING | — |

## Phase 7 — Release

| Task | Owner | Status | Evidence |
|---|---|---|---|
| setup/run/test/build .cmd scripts | Release | PENDING | — |
| PyInstaller one-folder package, launch-tested | Release | PENDING | — |
| GitHub Actions CI (windows-latest) | Release | PENDING | — |
| Docs: README/ARCHITECTURE/ALGORITHMS/BENCHMARKS/TEST_REPORT/PORTFOLIO | Release | PENDING | — |
| Independent final audit | Reviewer | PENDING | — |

## Known Issues / Decisions Log

- 2026-10-25: first pip install stalled at 0 KB/s after ~20 min (large PySide6
  wheels on flaky connection). Mitigation: killed and reinstalled in two
  batches with `--timeout 60 --retries 8`; switched to `PySide6-Essentials`
  (sufficient for QtWidgets/QtGui used by pyqtgraph).
- 2026-10-25: panda.xml already defines a `home` keyframe; scene XMLs must not
  re-define it (duplicate key name compile error). Scene generator omits it.
- 2026-10-25: single_obs wall originally intersected the arm at home pose
  (ncon=12); moved to y=-0.24 side approach blocker.
