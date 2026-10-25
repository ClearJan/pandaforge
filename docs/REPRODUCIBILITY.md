# Reproducibility

Every benchmark number in this repository traces to a committed raw file and
a re-runnable command. This page is the index.

## Environment

| Run | Date | OS | Python | Key versions |
|---|---|---|---|---|
| all committed benchmarks | 2026-10-25 | Windows 11 (i9-14900HX, 32 GB) | 3.11.9 | mujoco 3.17.0, numpy 2.4.6, scipy 1.17.1 (requirements-lock.txt) |

## Commands

| Report section | Command | Raw data |
|---|---|---|
| IK (S1) | `.venv\Scripts\python -m robotstudio.benchmarks.ik_bench` | `reports/benchmarks/ik_benchmark.{csv,json}` |
| Planning (S2) | `.venv\Scripts\python -m robotstudio.benchmarks.plan_bench` | `reports/benchmarks/planning_*.csv`, `planning_summary.json` |
| RRT*/Informed (S4) | `.venv\Scripts\python -m robotstudio.benchmarks.rrt_star_bench` | `reports/benchmarks/rrt_star_*.csv`, `rrt_star_summary.json` |
| TOPP-RA (S5) | `.venv\Scripts\python -m robotstudio.benchmarks.toppra_bench` | `reports/benchmarks/toppra_duration.csv`, `toppra_summary.json` |
| Replanning (S6) | `.venv\Scripts\python -m robotstudio.benchmarks.replan_bench` | `reports/benchmarks/replan_trials.csv`, `replan_summary.json` |
| Pick&place (S3) | `.venv\Scripts\python -m robotstudio.benchmarks.grasp_bench` | `reports/benchmarks/grasp_trials.csv`, `grasp_summary.json` |
| Charts | `.venv\Scripts\python scripts/make_charts.py` | `reports/benchmarks/charts/*.png` |
| Demo GIF | `.venv\Scripts\python scripts/make_demo_gif.py` | `docs/img/demo.gif` |
| Screenshots | `QT_QPA_PLATFORM=windows .venv\Scripts\python scripts/make_screenshots.py` | `docs/img/*.png` |

## Determinism contract

- Every randomized algorithm (planners, IK multi-start) takes an explicit
  seed; benchmarks fix seeds (100-129, 2028 etc. - stated per protocol).
- The moving obstacle is a pure function of simulation time.
- Contact dynamics have residual nondeterminism (solver warm-starting);
  grasp trials therefore use tolerance-based criteria and every single
  trial appears as a row in the CSV - nothing is aggregated away.

## What cannot be reproduced bit-exactly

Wall-clock timings vary across machines; the seed-fixed *search behavior*
(planner tree, path, waypoints) does not. Cross-machine planner outputs
should be identical given the same seeds; timings will differ.
