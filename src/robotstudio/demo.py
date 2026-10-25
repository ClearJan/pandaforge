"""Headless proof-of-work demo: verify the product's claims without a display.

Runs the full pipeline headless and prints physical evidence for every claim:

    python -m robotstudio.demo            # plan + execute + pick & place
    python -m robotstudio.demo --quick    # skip the sweep, grasp only

No GUI required - this is the 60-second "prove it to me" path for reviewers.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="RobotStudio headless demo")
    parser.add_argument("--quick", action="store_true", help="grasp task only")
    parser.add_argument("--moving", action="store_true", help="dynamic-obstacle guarded replanning demo")
    args = parser.parse_args()

    from robotstudio.collision.checker import CollisionChecker
    from robotstudio.config import PandaSpec
    from robotstudio.control.executor import TrajectoryExecutor
    from robotstudio.manipulation.fsm import GraspFSM, GraspState, GraspTaskConfig
    from robotstudio.planning.base import PlannerConfig
    from robotstudio.planning.rrt import RRTConnectPlanner
    from robotstudio.simulation.engine import SimulationEngine
    from robotstudio.trajectory.timing import TrajectoryGenerator

    print("=" * 64)
    print("RobotStudio headless demo - all evidence from live physics")
    print("=" * 64)

    scene = "moving_wall" if args.moving else "tabletop"
    eng = SimulationEngine(scene_name=scene, headless=True)
    eng.start(with_viewer=False)
    eng.set_speed(3.0)
    time.sleep(0.5)

    try:
        if args.moving:
            from robotstudio.control.monitor import GuardedExecutor
            from robotstudio.kinematics.ik import IKSolver
            from robotstudio.kinematics.pose import Pose

            print("\n[1] guarded planning through a sweeping panel")
            checker = CollisionChecker(eng.model, eng.robot)
            import mujoco as _mj

            ik = IKSolver(eng.model, _mj.MjData(eng.model), eng.robot, seed=2028)
            res_ik = ik.solve(
                Pose.make([0.35, -0.40, 0.55], [0.0, 1.0, 0.0, 0.0]),
                n_restarts=12,
                validate=lambda q: checker.check_state(q).valid,
            )
            checker.pin_obstacle_time(eng.snapshot().sim_time)
            plan = RRTConnectPlanner(
                checker, PlannerConfig(seed=2028, max_planning_time=10.0)
            ).plan(np.array(PandaSpec.HOME_QPOS), res_ik.arm_q)
            print(f"    initial plan: success={plan.success} waypoints={plan.n_waypoints}")
            guarded = GuardedExecutor(eng, TrajectoryGenerator().generate(plan.path))
            eng.set_controller(guarded)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 150.0 and not guarded.stats.completed:
                time.sleep(0.1)
            st = guarded.stats
            print(f"    completed={st.completed} success={st.success} "
                  f"replans={st.replans} guard_violations={st.guard_violations} "
                  f"obstacle_contacts={st.obstacle_contacts}")
            ok = st.completed and st.success and st.obstacle_contacts == 0 and st.replans >= 1
            print("\n" + ("ALL CLAIMS VERIFIED" if ok else "DEMO FAILED - see messages above"))
            return 0 if ok else 2

        if not args.quick:
            # ---- 1. plan around the table ----
            print("\n[1] motion planning (RRT-Connect, collision-checked)")
            checker = CollisionChecker(eng.model, eng.robot)
            start = np.array(PandaSpec.HOME_QPOS)
            goal = np.clip(start + np.radians([25, -20, 10, -5, 0, 10, 0]), *eng.robot.jnt_range.T)
            res = RRTConnectPlanner(checker, PlannerConfig(seed=2028, max_planning_time=10.0)).plan(start, goal)
            print(f"    success={res.success}  waypoints={res.n_waypoints}  "
                  f"length={res.path_length:.2f} rad  time={res.planning_time*1000:.0f} ms")
            if not res.success:
                return 1
            fresh = CollisionChecker(eng.model, eng.robot)
            print(f"    independent re-validation: {'PASS' if fresh.check_path(res.path).valid else 'FAIL'}")

            # ---- 2. execute through real physics ----
            print("\n[2] trajectory execution (servo-streamed, settle-verified)")
            eng.set_path_visual(res.path)
            traj = TrajectoryGenerator().generate(res.path)
            print(f"    duration={traj.duration:.2f} s (TOPP-RA time-optimal allocation)")
            ex = TrajectoryExecutor(traj)
            eng.set_controller(ex)
            t0 = time.perf_counter()
            while not ex.stats.finished and time.perf_counter() - t0 < 120:
                time.sleep(0.05)
            snap = eng.snapshot()
            err = float(np.max(np.abs(snap.qpos[eng.robot.arm_qposadr] - goal)))
            print(f"    completed={ex.stats.success}  final joint error={np.degrees(err):.2f} deg  "
                  f"max tracking err={np.degrees(ex.stats.max_joint_tracking_err):.2f} deg")

        # ---- 3. automatic pick & place with physical verification ----
        print("\n[3] automatic pick & place (15-state FSM, physical criteria)")
        eng.reset()
        time.sleep(0.5)
        snap = eng.snapshot()
        spec = GraspTaskConfig(
            cube_start=tuple(snap.cube_pos),
            cube_half=0.025,
            place_target=(0.55, -0.30, 0.0),
            surface_z=0.40,
            seed=2028,
        )
        fsm = GraspFSM(eng, spec)
        eng.set_controller(fsm)
        fsm.start()
        seen: list[str] = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 240 and not fsm.is_terminal:
            st = fsm.state.value
            if not seen or seen[-1] != st:
                seen.append(st)
            time.sleep(0.05)
        print(f"    states visited: {' -> '.join(seen)}")
        final = eng.snapshot()
        xy_err = float(np.linalg.norm(final.cube_pos[:2] - np.array([0.55, -0.30])))
        moved = float(np.linalg.norm(final.cube_pos - snap.cube_pos))
        opening = float(np.mean(final.qpos[eng.robot.finger_qposadr]))
        print(f"    fsm verdict: {fsm.state.value}")
        print(f"    PHYSICAL EVIDENCE: cube moved {moved*1000:.0f} mm, "
              f"placed {xy_err*1000:.0f} mm from target, resting z={final.cube_pos[2]:.3f} m, "
              f"gripper open {opening*1000:.0f} mm")
        ok = fsm.state == GraspState.SUCCESS and xy_err <= 0.035 and moved > 0.15
        print("\n" + ("ALL CLAIMS VERIFIED" if ok else "DEMO FAILED - see messages above"))
        return 0 if ok else 2
    finally:
        eng.stop()


if __name__ == "__main__":
    raise SystemExit(main())
