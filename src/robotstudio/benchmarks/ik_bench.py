"""IK benchmark: success rate / errors / runtime on FK-generated reachable
poses, plus unreachable-target rejection. Writes JSON + CSV to reports/benchmarks/.
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
from robotstudio.kinematics.fk import forward_kinematics
from robotstudio.kinematics.ik import IKSolver
from robotstudio.kinematics.pose import Pose
from robotstudio.robots.panda import PandaRobot
from robotstudio.simulation.scenes import preset_scenes, write_scene_file
from robotstudio.utils.logging import get_logger

log = get_logger("bench.ik")


def run_ik_benchmark(n_reachable: int = 120, n_unreachable: int = 20, seed: int = 2028) -> dict:
    rng = np.random.default_rng(seed)
    path = write_scene_file(preset_scenes()["empty"])
    model = mujoco.MjModel.from_xml_path(str(path))
    robot = PandaRobot(model)
    checker = CollisionChecker(model, robot)
    solver = IKSolver(model, mujoco.MjData(model), robot, seed=seed)
    fk_data = mujoco.MjData(model)

    rows = []
    # ---- reachable poses (ground truth = FK of a collision-free config) ----
    # Targets whose ONLY realizations are self-colliding configurations are
    # not "valid reachable poses", so the ground-truth sample is filtered.
    fk_data = mujoco.MjData(model)
    n_sampled = 0
    while len(rows) < n_reachable:
        n_sampled += 1
        q_true = rng.uniform(robot.jnt_range[:, 0] + 0.03, robot.jnt_range[:, 1] - 0.03)
        if not checker.check_state(q_true).valid:
            continue  # ground truth itself collides -> not a valid sample
        target = forward_kinematics(model, fk_data, robot, q_true)
        i = len(rows)
        q_init = np.array(PandaSpec.HOME_QPOS) if i % 2 else rng.uniform(robot.jnt_range[:, 0], robot.jnt_range[:, 1])
        t0 = time.perf_counter()
        res = solver.solve(target, q_init=q_init, n_restarts=6, validate=lambda q: checker.check_state(q).valid)
        dt = time.perf_counter() - t0
        rows.append(
            {
                "kind": "reachable",
                "success": res.success,
                "pos_err_mm": res.pos_err * 1000,
                "rot_err_deg": float(np.degrees(res.rot_err)),
                "iterations": res.iterations,
                "time_ms": dt * 1000,
                "within_limits": bool(robot.within_limits(res.arm_q)) if res.success else "",
                "collision_free": bool(checker.check_state(res.arm_q).valid) if res.success else "",
            }
        )
    log.info("sampled %d configs to get %d valid reachable targets", n_sampled, n_reachable)

    # ---- unreachable targets (must be reported as failures) ----
    for _ in range(n_unreachable):
        ang = rng.uniform(0, 2 * np.pi)
        target = Pose.make([1.0 * np.cos(ang), 1.0 * np.sin(ang), 1.1])
        t0 = time.perf_counter()
        res = solver.solve(target, n_restarts=3)
        rows.append(
            {
                "kind": "unreachable",
                "success": res.success,
                "pos_err_mm": res.pos_err * 1000,
                "rot_err_deg": float(np.degrees(res.rot_err)),
                "iterations": res.iterations,
                "time_ms": (time.perf_counter() - t0) * 1000,
                "within_limits": "",
                "collision_free": "",
            }
        )

    # ---- summary ----
    reach = [r for r in rows if r["kind"] == "reachable"]
    unre = [r for r in rows if r["kind"] == "unreachable"]
    ok = [r for r in reach if r["success"]]
    summary = {
        "n_reachable": n_reachable,
        "n_unreachable": n_unreachable,
        "success_rate": len(ok) / n_reachable,
        "max_pos_err_mm": max((r["pos_err_mm"] for r in ok), default=0),
        "max_rot_err_deg": max((r["rot_err_deg"] for r in ok), default=0),
        "mean_time_ms": float(np.mean([r["time_ms"] for r in ok])) if ok else 0,
        "p95_time_ms": float(np.percentile([r["time_ms"] for r in ok], 95)) if ok else 0,
        "limit_violations": sum(1 for r in ok if r["within_limits"] is False),
        "collision_invalid_solutions": sum(1 for r in ok if r["collision_free"] is False),
        "false_success_on_unreachable": sum(1 for r in unre if r["success"]),
        "seed": seed,
    }

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    with open(BENCH_ROOT / "ik_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(BENCH_ROOT / "ik_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("IK benchmark done: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_ik_benchmark(), indent=2))
