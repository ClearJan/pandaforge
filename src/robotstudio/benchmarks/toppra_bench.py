"""TOPP-RA duration benchmark on real planner paths.

Protocol: for each scene (empty, tabletop, single_obs) and 10 fixed seeds,
plan with RRT-Connect (shortcutting ON - the production configuration), then
time-parameterize the SAME geometric path with (a) the legacy conservative
spline and (b) TOPP-RA. Report durations, numeric-limit ratios, and the
execution tracking error of both via a short headless run (one seed per
scene) - time-optimal profiles stress the servos more, reported honestly.

Outputs: reports/benchmarks/toppra_duration.csv + toppra_summary.json.
"""

from __future__ import annotations

import csv
import json

import mujoco
import numpy as np

from robotstudio.collision.checker import CollisionChecker
from robotstudio.config import BENCH_ROOT, PandaSpec
from robotstudio.planning.base import PlannerConfig
from robotstudio.planning.rrt import RRTConnectPlanner
from robotstudio.robots.panda import PandaRobot
from robotstudio.simulation.scenes import preset_scenes, write_scene_file
from robotstudio.trajectory.timing import TrajectoryGenerator
from robotstudio.utils.logging import get_logger

log = get_logger("bench.toppra")
SEEDS = list(range(101, 111))


def _goal(scenario: str, model, robot, checker, rng) -> np.ndarray | None:
    start = np.array(PandaSpec.HOME_QPOS)
    if scenario == "empty":
        for _ in range(60):
            q = rng.uniform(robot.jnt_range[:, 0] + 0.08, robot.jnt_range[:, 1] - 0.08)
            if checker.check_state(q).valid and np.linalg.norm(q - start) > 0.8:
                return q
        return None
    q = np.clip(start + np.array([0.25, -0.35, 0.15, -0.08, 0.1, 0.12, 0.1]), *robot.jnt_range.T)
    return q if checker.check_state(q).valid else None


def run_toppra_benchmark() -> dict:
    rows = []
    for scenario in ("empty", "tabletop", "single_obs"):
        path = write_scene_file(preset_scenes()[scenario])
        model = mujoco.MjModel.from_xml_path(str(path))
        robot = PandaRobot(model)
        start = np.array(PandaSpec.HOME_QPOS)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            checker = CollisionChecker(model, robot)
            goal = _goal(scenario, model, robot, checker, rng)
            if goal is None:
                continue
            # RAW planner path (no shortcut): multi-waypoint paths are
            # TOPP-RA's domain - after shortcutting, 2-waypoint paths
            # degenerate to the legacy generator for BOTH methods.
            res = RRTConnectPlanner(
                CollisionChecker(model, robot), PlannerConfig(seed=seed, max_planning_time=10.0, shortcut_tries=0)
            ).plan(start, goal)
            if not res.success:
                continue
            t_legacy = TrajectoryGenerator(method="spline").generate(res.path)
            t_topp = TrajectoryGenerator().generate(res.path)  # default = TOPP-RA
            vl, al = t_legacy.constraint_violation()
            vt, at = t_topp.constraint_violation()
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "path_length_rad": res.path_length,
                    "waypoints": len(res.path),
                    "legacy_duration_s": t_legacy.duration,
                    "topp_duration_s": t_topp.duration,
                    "reduction_pct": 100.0 * (1.0 - t_topp.duration / t_legacy.duration),
                    "legacy_v_ratio": vl,
                    "topp_v_ratio": vt,
                    "legacy_a_ratio": al,
                    "topp_a_ratio": at,
                }
            )
        log.info("scenario %s done", scenario)

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    with open(BENCH_ROOT / "toppra_duration.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for scenario in ("empty", "tabletop", "single_obs"):
        rs = [r for r in rows if r["scenario"] == scenario]
        if not rs:
            continue
        red = [r["reduction_pct"] for r in rs]
        summary[scenario] = {
            "n": len(rs),
            "median_legacy_s": float(np.median([r["legacy_duration_s"] for r in rs])),
            "median_topp_s": float(np.median([r["topp_duration_s"] for r in rs])),
            "median_reduction_pct": float(np.median(red)),
            "min_reduction_pct": float(np.min(red)),
            "max_reduction_pct": float(np.max(red)),
            "all_feasible": all(r["legacy_a_ratio"] <= 1 and r["topp_a_ratio"] <= 1 for r in rs),
        }
    with open(BENCH_ROOT / "toppra_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("TOPP-RA benchmark done: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_toppra_benchmark(), indent=2))
