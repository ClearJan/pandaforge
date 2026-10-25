"""Online replanning benchmark: guarded vs static-plan execution.

Protocol: moving_wall scene, home -> far-side goal across the sweep
corridor. Two sweep speeds (period 11 s ~0.18 m/s, period 9 s ~0.22 m/s),
5 trials each. GUARDED runs use GuardedExecutor (swept-volume guard +
park + online replanning); STATIC runs execute the initial plan with a
plain TrajectoryExecutor and no guard. Success = goal reached with ZERO
arm-obstacle contacts; the static baseline is expected to be hit - that
contrast is the point.
"""

from __future__ import annotations

import csv
import json
import time

import mujoco
import numpy as np

from robotstudio.collision.checker import CollisionChecker
from robotstudio.config import BENCH_ROOT, PandaSpec
from robotstudio.control.executor import TrajectoryExecutor
from robotstudio.control.monitor import GuardedExecutor
from robotstudio.kinematics.ik import IKSolver
from robotstudio.kinematics.pose import Pose
from robotstudio.planning.base import PlannerConfig
from robotstudio.planning.rrt import RRTConnectPlanner
from robotstudio.simulation.engine import SimulationEngine
from robotstudio.simulation.scenes import preset_scenes
from robotstudio.trajectory.timing import TrajectoryGenerator
from robotstudio.utils.logging import get_logger

log = get_logger("bench.replan")

SPEEDS = {"slow": 11.0, "fast": 9.0}  # sweep period (s)
TRIALS = 5


def _one_trial(period: float, guarded: bool, seed: int) -> dict:
    spec = preset_scenes()["moving_wall"]
    spec.moving_obstacle.period = period
    eng = SimulationEngine(scene_name="moving_wall", headless=True)
    eng._scene_spec = spec  # apply before start (engine thread not running)
    eng.start(with_viewer=False)
    eng.set_speed(3.0)
    try:
        time.sleep(0.3)
        checker = CollisionChecker(eng.model, eng.robot)
        ik = IKSolver(eng.model, mujoco.MjData(eng.model), eng.robot, seed=seed)
        res_ik = ik.solve(
            Pose.make([0.35, -0.40, 0.55], [0.0, 1.0, 0.0, 0.0]),
            n_restarts=12,
            validate=lambda q: checker.check_state(q).valid,
        )
        if not res_ik.success:
            return {"ok": False, "reason": "goal IK failed"}
        checker.pin_obstacle_time(eng.snapshot().sim_time)
        plan = RRTConnectPlanner(
            checker, PlannerConfig(seed=seed, max_planning_time=10.0)
        ).plan(np.array(PandaSpec.HOME_QPOS), res_ik.arm_q)
        if not plan.success:
            return {"ok": False, "reason": f"initial plan failed: {plan.message}"}
        traj = TrajectoryGenerator().generate(plan.path)

        # THE POINT: the plan is made while the corridor is OPEN (panel at
        # the far end), then execution starts as the panel sweeps IN - the
        # world changes between planning and execution. Without the guard
        # the static plan walks the arm into the panel.
        while eng.snapshot().sim_time < 1.7:
            time.sleep(0.05)

        contacts = 0
        if guarded:
            ctl = GuardedExecutor(eng, traj, seed=seed)
        else:
            ctl = TrajectoryExecutor(traj)
        eng.set_controller(ctl)

        # count arm-vs-obstacle contacts from live physics for BOTH modes
        obs_geoms = set()
        bid = int(eng.model.body("moving_obstacle").id)
        st0 = int(eng.model.body_geomadr[bid])
        obs_geoms.update(range(st0, st0 + int(eng.model.body_geomnum[bid])))
        arm_geoms = set()
        for name in ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"):
            b = int(eng.model.body(name).id)
            s = int(eng.model.body_geomadr[b])
            arm_geoms.update(range(s, s + int(eng.model.body_geomnum[b])))

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 150.0:
            time.sleep(0.1)
            d = eng.data
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in obs_geoms and g2 in arm_geoms) or (g2 in obs_geoms and g1 in arm_geoms):
                    contacts += 1
                    break
            done = getattr(ctl.stats, "completed", None)
            if done is None:
                done = ctl.stats.finished
            if done:
                break
        time.sleep(0.8)
        final_err = float(np.max(np.abs(eng.snapshot().qpos[eng.robot.arm_qposadr] - res_ik.arm_q)))
        replans = getattr(getattr(ctl, "stats", None), "replans", 0)
        wall = time.perf_counter() - t0
        return {
            "ok": True,
            "guarded": guarded,
            "contacts": contacts,
            "final_err_rad": final_err,
            "replans": int(replans),
            "wall_s": wall,
            "reached": final_err < 0.08,
        }
    finally:
        eng.stop()


def run_replan_benchmark() -> dict:
    rows = []
    for label, period in SPEEDS.items():
        for mode in (True, False):
            for i in range(TRIALS):
                r = _one_trial(period, mode, seed=2028 + i)
                r["speed"] = label
                r["period_s"] = period
                r["trial"] = i
                rows.append(r)
                log.info("%s/%s trial %d: %s", label, "guarded" if mode else "static", i, r)

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    with open(BENCH_ROOT / "replan_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for label in SPEEDS:
        for mode_name, mode in (("guarded", True), ("static", False)):
            rs = [r for r in rows if r["speed"] == label and r["guarded"] == mode and r["ok"]]
            if rs:
                summary[f"{label}/{mode_name}"] = {
                    "n": len(rs),
                    "zero_contact_rate": sum(1 for r in rs if r["contacts"] == 0) / len(rs),
                    "reached_rate": sum(1 for r in rs if r["reached"]) / len(rs),
                    "median_contacts": float(np.median([r["contacts"] for r in rs])),
                    "median_replans": float(np.median([r["replans"] for r in rs])),
                }
    with open(BENCH_ROOT / "replan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("replan benchmark done: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_replan_benchmark(), indent=2))
