"""Planning benchmark: 3 scenarios x N seeds x 3 planners, fixed protocol.

Protocol per (scenario, planner, seed):
- start = home configuration
- goal = a collision-free goal derived per scenario (deterministic per seed)
- time budget 8 s; successful paths re-validated by an independent checker
All raw rows go to reports/benchmarks/planning_<scenario>.csv.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from robotstudio.collision.checker import CollisionChecker
from robotstudio.config import BENCH_ROOT, PandaSpec
from robotstudio.kinematics.ik import IKSolver
from robotstudio.kinematics.pose import Pose
from robotstudio.planning.base import PlannerConfig
from robotstudio.planning.linear import LinearPlanner
from robotstudio.planning.rrt import RRTConnectPlanner, RRTPlanner
from robotstudio.robots.panda import PandaRobot
from robotstudio.simulation.scenes import preset_scenes, write_scene_file
from robotstudio.utils.logging import get_logger

log = get_logger("bench.plan")

PLANNERS = {"linear": LinearPlanner, "rrt": RRTPlanner, "rrt_connect": RRTConnectPlanner}
TIME_BUDGET = 8.0
# stable per-scenario salt (Python's hash() is randomized per process)
_SCENARIO_SALT = {"empty": 1, "single_obs": 2, "narrow": 3}


def _goal_for_scenario(scenario: str, model, robot, checker, seed: int) -> tuple[np.ndarray, bool]:
    """Deterministic per-seed goal: jittered candidates until one is valid.

    Returns (q_goal, feasible). Candidate positions are constrained to the
    scenario's task region; the collision-free filter guarantees the goal
    itself is a valid configuration (some jittered poses are only reachable
    in collision and are skipped deterministically).
    """
    salt = _SCENARIO_SALT[scenario]
    rng = np.random.default_rng(seed * 31 + salt * 1009)
    ik = IKSolver(model, mujoco.MjData(model), robot, seed=seed)
    if scenario == "empty":
        q = rng.uniform(robot.jnt_range[:, 0] + 0.1, robot.jnt_range[:, 1] - 0.1)
        if checker.check_state(q).valid:
            return q, True
        return np.zeros(7), False
    if scenario == "single_obs":
        base = np.array([0.35, -0.40, 0.55])
        quat = [0.0, 1.0, 0.0, 0.0]
    elif scenario == "narrow":
        base = np.array([0.42, -0.38, 0.62])
        quat = [0.0, 0.9239, 0.0, -0.3827]
    else:
        raise ValueError(scenario)
    for _ in range(10):  # deterministic candidate chain
        pos = base + np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(-0.02, 0.02)])
        res = ik.solve(Pose.make(pos, quat), n_restarts=8, validate=lambda q: checker.check_state(q).valid)
        if res.success:
            return res.arm_q, True
    return np.zeros(7), False


def run_planning_benchmark(scenarios: list[str] | None = None, n_seeds: int = 30, seed0: int = 100) -> dict:
    scenarios = scenarios or ["empty", "single_obs", "narrow"]
    summary = {}
    for scenario in scenarios:
        path = write_scene_file(preset_scenes()[scenario])
        model = mujoco.MjModel.from_xml_path(str(path))
        robot = PandaRobot(model)
        start = np.array(PandaSpec.HOME_QPOS)
        rows = []
        for seed in range(seed0, seed0 + n_seeds):
            checker = CollisionChecker(model, robot)
            goal, feasible = _goal_for_scenario(scenario, model, robot, checker, seed)
            if not feasible:
                log.warning("%s seed %d: goal generation failed, skipping", scenario, seed)
                continue
            for name, cls in PLANNERS.items():
                # fresh checker per plan (independent validation uses another)
                plan_checker = CollisionChecker(model, robot)
                planner = cls(plan_checker, PlannerConfig(seed=seed, max_planning_time=TIME_BUDGET))
                t0 = time.perf_counter()
                res = planner.plan(start, goal)
                dt = time.perf_counter() - t0
                independent_ok = ""
                if res.success:
                    fresh = CollisionChecker(model, robot)
                    independent_ok = bool(fresh.check_path(res.path).valid)
                rows.append(
                    {
                        "scenario": scenario,
                        "planner": name,
                        "seed": seed,
                        "success": res.success,
                        "independently_validated": independent_ok,
                        "time_s": dt,
                        "path_length_rad": res.path_length,
                        "n_waypoints": res.n_waypoints,
                        "n_nodes": res.n_nodes,
                        "n_collision_checks": res.n_collision_checks,
                        "shortcut_reduction_pct": res.stats.get("shortcut_reduction", 0) * 100,
                        "message": res.message,
                    }
                )
        out = BENCH_ROOT / f"planning_{scenario}.csv"
        BENCH_ROOT.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        for name in PLANNERS:
            rs = [r for r in rows if r["planner"] == name]
            ok = [r for r in rs if r["success"]]
            summary[f"{scenario}/{name}"] = {
                "n": len(rs),
                "success_rate": len(ok) / len(rs) if rs else 0,
                "median_time_s": float(np.median([r["time_s"] for r in ok])) if ok else None,
                "p90_time_s": float(np.percentile([r["time_s"] for r in ok], 90)) if ok else None,
                "median_path_length": float(np.median([r["path_length_rad"] for r in ok])) if ok else None,
                "invalid_paths_reported_ok": sum(1 for r in ok if r["independently_validated"] is False),
            }
        log.info("scenario %s done: %s", scenario, json.dumps({k: v for k, v in summary.items() if k.startswith(scenario)}, indent=None))
    with open(BENCH_ROOT / "planning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_planning_benchmark(), indent=2))
