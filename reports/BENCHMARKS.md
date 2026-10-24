# Benchmarks

All numbers in this file were produced by the benchmark runners in
`src/robotstudio/benchmarks/` on the machine below. Raw per-trial data and
summary JSON/CSV files are committed under `reports/benchmarks/`; charts
under `reports/benchmarks/charts/`. Re-run everything with:

```
.venv\Scripts\python -m robotstudio.benchmarks.ik_bench
.venv\Scripts\python -m robotstudio.benchmarks.plan_bench
.venv\Scripts\python -m robotstudio.benchmarks.grasp_bench
.venv\Scripts\python scripts\make_charts.py
```

## Environment

- Windows 11, Python 3.11.9
- Intel Core i9-14900HX (24 cores), 32 GB DDR5, NVMe SSD
- NVIDIA RTX 4070 Laptop (not used - the stack is CPU physics)
- mujoco 3.17.0, numpy 2.4.6, scipy 1.17.1 (see requirements-lock.txt)
- Date of runs: 2026-10-25

## 1. Inverse kinematics (DLS, multi-start, collision-validated)

Protocol (`ik_bench.py`, seed 2028): ground-truth configurations sampled in
joint space and **filtered for collision-freeness** (128 sampled -> 120
valid targets); targets = FK of the ground truth; alternating home / random
initial guesses; 6 random restarts; solutions must be within limits and
collision-free. Plus 20 unreachable targets (radius 1.0-1.5 m sphere above
the base) that must be reported as failures.

| metric | value |
|---|---|
| success rate (valid reachable) | **100%** (120/120) |
| max position error (successes) | 4.99 mm (limit: 10 mm) |
| max rotation error (successes) | 2.79° (limit: 5°) |
| mean / p95 solve time | 9.9 ms / 56.7 ms |
| joint-limit violations | 0 |
| collision-invalid solutions accepted | 0 |
| false successes on unreachable targets | **0** (20/20 correctly rejected) |

![IK time histogram](charts/ik_times.png)

## 2. Motion planning (3 scenarios x 28-30 seeds x 3 planners)

Protocol (`plan_bench.py`): start = home; per-seed deterministic goal
(IK-derived, collision-free, jittered within the scenario task region);
time budget 8 s; fixed seeds 100-129. Every reported-successful path is
re-validated by an INDEPENDENT checker instance (dense re-check).

| scenario | planner | n | success | median time | median path length |
|---|---|---|---|---|---|
| empty | linear | 28 | 100% | 1.9 ms | 3.70 rad |
| empty | rrt | 28 | 100% | 23.3 ms | 3.70 rad |
| empty | rrt_connect | 28 | 100% | 7.2 ms | 3.70 rad |
| single_obs | linear | 29 | 27.6% | 2.8 ms | 3.43 rad |
| single_obs | rrt | 29 | 44.8% | 0.66 s | 3.53 rad |
| single_obs | rrt_connect | 29 | **100%** | 0.19 s | 3.51 rad |
| narrow | linear | 30 | 0% | — | — |
| narrow | rrt | 30 | 46.7% | 2.61 s | 5.83 rad |
| narrow | rrt_connect | 30 | **100%** | 0.23 s | 4.54 rad |

- **Safety: 0 of the 261 trials produced a "successful" path that failed the
  independent re-validation.**
- single_obs has n=29 because one seed's goal chain produced no
  collision-free goal within 10 candidates; it is skipped and counted (not
  silently dropped - see the runner log).
- RRT's 45-47% on the blocked scenes reflects the 8 s budget for a
  single-tree planner in 7-D; RRT-Connect's bidirectional CONNECT step is
  the difference (see docs/ALGORITHMS.md §7). No failures were discarded.

![Success rates](charts/planning_success_rate.png)
![Planning time distributions](charts/planning_times.png)

Shortcutting: on the single_obs/narrow scenes the randomized shortcut pass
reduces joint-space path length by ~20-55% depending on seed (column
`shortcut_reduction_pct` in the CSVs), always followed by a full independent
re-validation.

## 3. Pick & place (physical trials)

Protocol (`grasp_bench.py`): live engine headless at 3x realtime; each
trial runs the full 15-state FSM; success requires the FSM verdict AND the
independent physical post-check (cube within 35 mm of target xy, resting on
the table, moved > 150 mm, gripper open). `impossible` trials expect failure
(a "success" there would indicate a bug).

| category | n | success rate | median wall time |
|---|---|---|---|
| base (fixed cube, fixed target) | 20 | **100%** (target ≥90%) | 2.87 s |
| jitter (cube ±3 cm) | 10 | **100%** | 2.87 s |
| hard_place (table-edge targets) | 5 | **100%** | 2.87 s |
| impossible (out of reach) | 3 | 3/3 correctly FAILED | 1.76 s |

![Grasp success](charts/grasp_success.png)

Failure analysis history (all fixed, kept for the record):

1. Palm-hull grazing the cube top during a centred grasp -> pads raised
   5 mm above cube mid-height (D6/D10 in docs/DECISIONS.md).
2. Cube slipping at lift -> squeeze depth increased to 20 mm after a
   measured force-vs-command curve (1.96 N at 5.3 mm commanded opening).
3. RETREAT "start in collision" on table-edge placements -> root causes
   were (a) trajectory completion freezing servo lag and (b) hold states
   chasing the actual position (gravity ratchet, 22 mm drift). Fixed by
   settle-based completion + static hold targets (D5) and a 10 mm release
   clearance.

## 4. Collision checker performance

| operation | measured |
|---|---|
| single state check (kinematics + collision + whitelist) | 0.052 ms |
| 200-state batch | 10.4 ms |

## 5. Optimization record (evidence-based)

| change | evidence | effect |
|---|---|---|
| vectorized RRT nearest-neighbour | profile: Python node scan dominated; collision check only 0.05 ms of 3.2 ms/iter | 3.2x more iterations/s (6300 -> 20000 in the same budget) |
| adaptive `constraint_violation` sampling | first trajectory test run took 75 s (fixed 2 ms grid over a 2000 s trajectory) | suite time 75 s -> 0.6 s |
| IK benchmark population filter | 94.2% -> 100% by excluding self-colliding ground truths (protocol fix, not solver change) | honest measurement |

No optimization was applied without a before/after measurement on identical
seeds/scenes; raw numbers live in the CSV/JSON files next to this document.
