"""RRT* / Informed RRT* quality benchmark.

Protocol: open + tabletop scenes, 10 fixed seeds, 20 s budget, shortcutting
DISABLED so raw planner path quality is compared. Records per-trial final
raw cost + full best-cost convergence history; also probes the blocked
single_obs scene once per seed to report the known first-solution difficulty
of single-tree optimal planners in 7-D honestly.

Outputs: reports/benchmarks/rrt_star_quality.csv, rrt_star_convergence.csv,
rrt_star_summary.json.
"""

from __future__ import annotations

import csv
import json

import mujoco
import numpy as np

from robotstudio.collision.checker import CollisionChecker
from robotstudio.config import BENCH_ROOT, PandaSpec
from robotstudio.planning.base import PlannerConfig
from robotstudio.planning.rrt import RRTPlanner
from robotstudio.planning.rrt_star import InformedRRTStarPlanner, RRTStarPlanner
from robotstudio.robots.panda import PandaRobot
from robotstudio.simulation.scenes import preset_scenes, write_scene_file
from robotstudio.utils.logging import get_logger

log = get_logger("bench.rrt_star")

BUDGET = 15.0
SEEDS = list(range(101, 111))
PLANNERS = {
    "rrt": RRTPlanner,
    "rrt_star": RRTStarPlanner,
    "informed_rrt_star": InformedRRTStarPlanner,
}


def _goal(scenario: str, model, robot, checker, rng) -> np.ndarray | None:
    start = np.array(PandaSpec.HOME_QPOS)
    if scenario == "empty":
        for _ in range(20):
            q = rng.uniform(robot.jnt_range[:, 0] + 0.08, robot.jnt_range[:, 1] - 0.08)
            if checker.check_state(q).valid and np.linalg.norm(q - start) > 1.0:
                return q
        return None
    if scenario == "tabletop":
        q = np.clip(start + np.array([0.2, -0.35, 0.12, -0.08, 0.1, 0.12, 0.1]), *robot.jnt_range.T)
        return q if checker.check_state(q).valid else None
    if scenario == "single_obs":
        from robotstudio.kinematics.ik import IKSolver
        from robotstudio.kinematics.pose import Pose

        ik = IKSolver(model, mujoco.MjData(model), robot, seed=int(rng.integers(0, 9999)))
        res = ik.solve(
            Pose.make([0.35, -0.40, 0.55], [0.0, 1.0, 0.0, 0.0]),
            n_restarts=12,
            validate=lambda q_: checker.check_state(q_).valid,
        )
        return res.arm_q if res.success else None
    return None


def run_rrt_star_benchmark() -> dict:
    rows: list[dict] = []
    conv_rows: list[dict] = []
    for scenario in ("empty", "tabletop", "single_obs"):
        path = write_scene_file(preset_scenes()[scenario])
        model = mujoco.MjModel.from_xml_path(str(path))
        robot = PandaRobot(model)
        start = np.array(PandaSpec.HOME_QPOS)
        for seed in SEEDS:
            goal = _goal(scenario, model, robot, CollisionChecker(model, robot), np.random.default_rng(seed))
            if goal is None:
                log.warning("%s seed %d: no feasible goal generated", scenario, seed)
                continue
            for name, cls in PLANNERS.items():
                planner = cls(
                    CollisionChecker(model, robot),
                    PlannerConfig(seed=seed, max_planning_time=BUDGET, shortcut_tries=0),
                )
                res = planner.plan(start, goal)
                if not res.success:
                    raw_len = None
                elif name == "rrt":
                    raw_len = res.path_length  # shortcut disabled -> raw
                else:
                    raw_len = res.stats.get("raw_path_length", res.path_length)
                rows.append(
                    {
                        "scenario": scenario,
                        "planner": name,
                        "seed": seed,
                        "success": res.success,
                        "time_s": res.planning_time,
                        "raw_path_length": raw_len if raw_len is not None else "",
                        "iterations": res.stats.get("iterations", 0),
                    }
                )
                if name in ("rrt_star", "informed_rrt_star") and scenario == "tabletop" and seed == SEEDS[0]:
                    for it, cost in res.stats.get("cost_history", []):
                        conv_rows.append({"planner": name, "iteration": it, "best_cost": cost if np.isfinite(cost) else ""})
        log.info("scenario %s done", scenario)

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    with open(BENCH_ROOT / "rrt_star_quality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(BENCH_ROOT / "rrt_star_convergence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["planner", "iteration", "best_cost"])
        w.writeheader()
        w.writerows(conv_rows)

    summary = {}
    for scenario in ("empty", "tabletop", "single_obs"):
        for name in PLANNERS:
            rs = [r for r in rows if r["scenario"] == scenario and r["planner"] == name]
            if not rs:
                continue
            ok = [r for r in rs if r["success"]]
            costs = [float(r["raw_path_length"]) for r in ok if r["raw_path_length"] != ""]
            summary[f"{scenario}/{name}"] = {
                "n": len(rs),
                "success_rate": len(ok) / len(rs),
                "median_raw_cost": float(np.median(costs)) if costs else None,
                "median_time_s": float(np.median([r["time_s"] for r in ok])) if ok else None,
            }
    with open(BENCH_ROOT / "rrt_star_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("RRT* benchmark done: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_rrt_star_benchmark(), indent=2))
