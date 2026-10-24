# Portfolio — RobotStudio as an interview project

A one-page technical summary for job applications: what I built, what is
genuinely hard in it, what the measured results are, and where to look in the
source. Every number links to committed raw data.

## 30-second pitch

> I built a Windows desktop application that turns MuJoCo + Franka Panda into
> an interactive motion-planning studio: inverse kinematics, RRT/RRT-Connect
> planning, time-parameterized trajectories executed through real physics,
> and a 15-state pick-and-place state machine whose success is judged by
> physical evidence, not scripts. Every benchmark in the repo is reproducible
> from committed seeds and raw CSVs.

## Core modules I implemented from scratch

| Module | Where | What makes it non-trivial |
|---|---|---|
| DLS inverse kinematics | `src/robotstudio/kinematics/ik.py` | adaptive damping, SO(3) log-map error (no Euler diffs), nullspace mid-range bias, collision-filtered multi-start; Jacobian frame convention proven by finite differences |
| Collision checking | `src/robotstudio/collision/checker.py` | explicit whitelist semantics, mode-dependent grasp/rest pairs, 0.02-rad dense segment checking, a carried-object transform so transfer paths are validated against the cube moving WITH the gripper |
| RRT / RRT-Connect | `src/robotstudio/planning/rrt.py` | bidirectional CONNECT with correct tree-swap bookkeeping; vectorized nearest-neighbour (profiled 3.2x speedup); deterministic per seed |
| Trajectory generation | `src/robotstudio/trajectory/timing.py` | time-scaled cubic splines; trapezoidal lower bounds; NUMERIC verification of velocity/acceleration limits with guaranteed-convergence rescaling |
| Execution control | `src/robotstudio/control/executor.py` | servo streaming at 500 Hz; settle-based completion (time-finishing is not enough — this was a real bug the trials caught) |
| Pick & place FSM | `src/robotstudio/manipulation/fsm.py` | 15 states with timeouts/retry/abort; success = fingertip contacts + the cube actually rising + resting position, all measured in the live contact/pose data |
| Simulation engine | `src/robotstudio/simulation/engine.py` | single-owner threading model with command queue; in-place scene switching (viewer never reloads); headless mode for tests |

## Hard problems I actually solved (with the receipts)

1. **"Successful" execution that wasn't.** Marking a trajectory done at
   `t >= duration` froze the servo lag as the next state's hold target;
   holding the *actual* position instead ratcheted the end-effector 22 mm
   downward under gravity. Fix: settle-based completion + static hold
   targets. Found by the grasp trials, fixed, and covered by physics tests.
   (`docs/DECISIONS.md` D5, `tests/physics/`)
2. **Grasp force calibration.** Commanding the gripper to the object width
   yields ~0.25 N — the cube slips at lift. I measured the force-vs-command
   curve of the tendon servo (1.34 N at 11.6 mm commanded opening vs 1.96 N
   at 5.3 mm) and chose a 20 mm squeeze depth with a 3.7x friction margin.
   (`docs/ALGORITHMS.md` §6)
3. **Planning against a moving held object.** A shadow-physics collision
   checker must place the held cube at (hand ∘ grasp-offset) for every
   sampled configuration, or transfer paths are validated against a cube
   that is simultaneously in the wrong place and not moving.
   (`collision/checker.py::cube_carry_offset`)
4. **Honest benchmarking.** The first IK run showed 94.2%; the "failures"
   had < 1.1 mm error — they were ground-truth targets only realizable by
   self-colliding configurations. Filtering the population (documented,
   not hidden) moved it to 100% and made the metric meaningful.
   (`docs/DECISIONS.md` D10)

## Measured results (all reproducible)

- IK: 100% on 120 valid reachable targets, ≤ 5.0 mm / 2.8°, mean 9.9 ms; 0
  false successes on 20 unreachable targets.
- Planning: RRT-Connect 100% on all 3 benchmark scenes (median 0.19–0.23 s);
  single-tree RRT 45–47% under an 8 s budget on blocked scenes — reported,
  not tuned away; 0 of 261 successful paths failed independent re-validation.
- Pick & place: 35/35 correct verdicts (20 base + 10 jittered + 5 table-edge
  successes; 3 impossible targets correctly rejected), median 2.87 s per
  task at 3x realtime.

## Where to look in 5 minutes

1. `src/robotstudio/kinematics/ik.py` — the DLS core (~120 lines).
2. `src/robotstudio/planning/rrt.py` — RRT-Connect (~60 lines).
3. `src/robotstudio/manipulation/fsm.py` — the state machine.
4. `reports/BENCHMARKS.md` — protocols + raw data.
5. `docs/ALGORITHMS.md` — the math with references.

## Questions an interviewer might ask (and where the answers live)

- *Why DLS instead of Levenberg-Marquardt or a nullspace projector alone?*
  → docs/ALGORITHMS.md §4 (the 6x6 solve keeps the redundancy handling in
  the nullspace term; damping is adaptive, bounded).
- *How do you know your Jacobian's rotation block is in world frame?*
  → finite-difference test (`tests/unit/test_kinematics.py`), not docs.
- *How do you prevent a planned path from clipping an obstacle between
  waypoints?* → 0.02-rad dense segment checks + independent full-path
  re-validation with a fresh checker (`tests/unit/test_planners.py`).
- *How is grasp success actually judged?* → ≥2 fingertip-cube contacts,
  cube rises ≥50% of lift height, final resting position within tolerance —
  all read from the live physics (`manipulation/fsm.py`).
- *What breaks if two threads touch MjData?* → the architecture forbids it:
  one owner thread, command queue, snapshots, shadow data
  (`ARCHITECTURE.md`, AGENTS.md rule 1).
- *RRT vs RRT-Connect — measured difference?* → reports/BENCHMARKS.md §2
  (100% vs 45-47% under the same budget; 20x median time).

## Resume bullet (draft, honest)

> Built RobotStudio, a Windows desktop app (Python/C++-free, MuJoCo-based)
> for interactive robot motion planning and physical manipulation:
> independently implemented damped-least-squares IK (100% on 120 validated
> reachable targets, ≤ 5 mm), RRT/RRT-Connect with independent path
> re-validation (100% success across 3 scenes, 0 unsafe outputs in 261
> trials), numerical trajectory time-parameterization, and a physically
> verified pick-and-place state machine (35/35 correct verdicts incl.
> out-of-reach rejection); packaged with reproducible benchmarks, 51-test
> CI suite and a self-testing PyInstaller distribution.

## Honest status

- Repo developed autonomously in this workspace; not yet pushed to GitHub
  (no credentials were configured in this session). `git log` contains the
  full incremental history; pushing is a `git remote add` + `git push` away.
- No stars/users/downloads are claimed anywhere - the project's evidence is
  its tests and raw benchmark data.
