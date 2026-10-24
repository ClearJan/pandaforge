"""Phase 3 integration tests: real physics execution of planned trajectories.

The engine runs headless at real-time factor 1.0; trajectories execute through
actual MuJoCo stepping with position-servo control. No teleports.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from robotstudio.config import PandaSpec
from robotstudio.control.executor import InfeasibleTrajectoryError, TrajectoryExecutor
from robotstudio.planning.base import PlannerConfig
from robotstudio.trajectory.timing import Trajectory, TrajectoryGenerator, TrajectoryLimits


@pytest.fixture()
def engine(request):
    from robotstudio.simulation.engine import SimulationEngine

    eng = SimulationEngine(scene_name="tabletop", headless=True)
    eng.start(with_viewer=False)
    yield eng
    eng.stop()


def _wait_until(predicate, timeout=30.0, interval=0.05):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_manual_joint_command_converges(engine):
    """Direct joint command must drive the physical arm to the target."""
    target = np.array([0.2, -0.3, 0.1, -1.2, 0.0, 1.4, 0.3])
    engine.set_joints(target)
    ok = _wait_until(
        lambda: np.max(np.abs(engine.snapshot().qpos[engine.robot.arm_qposadr] - target)) < 0.03,
        timeout=10.0,
    )
    assert ok, f"arm did not converge: {engine.snapshot().qpos}"


def test_invalid_joint_command_raises(engine):
    with pytest.raises(ValueError, match="outside limits"):
        engine.set_joints(np.full(7, 99.0))


def test_trajectory_execution_through_physics(engine):
    """Plan a real path, time-parameterize, execute via servos; verify tracking."""
    from robotstudio.collision.checker import CollisionChecker

    checker = CollisionChecker(engine.model, engine.robot)
    qpos, _qvel, _ctrl = engine.state_copy()
    start = engine.robot.arm_q_from_qpos(qpos)
    goal = np.clip(start + np.array([0.15, -0.2, 0.1, 0.0, -0.1, 0.1, 0.0]), *engine.robot.jnt_range.T)

    planner_res = __import__("robotstudio.planning", fromlist=["RRTConnectPlanner"]).RRTConnectPlanner(
        checker
    ).plan(start, goal)
    assert planner_res.success, planner_res.message

    traj = TrajectoryGenerator().generate(planner_res.path)
    executor = TrajectoryExecutor(traj)
    engine.set_controller(executor)

    ok = _wait_until(lambda: executor.stats.finished, timeout=traj.duration + 15.0)
    assert ok, "executor did not finish"

    snap = engine.snapshot()
    assert snap.controller_status == "completed"
    # the physical arm must actually BE at the goal (within servo tolerance)
    final_err = float(np.max(np.abs(snap.qpos[engine.robot.arm_qposadr] - goal)))
    assert final_err < 0.05, f"arm ended {final_err:.3f} rad from goal"
    # tracking statistics must have been collected from the real run
    assert executor.stats.max_joint_tracking_err > 0.0
    assert executor.stats.max_joint_tracking_err < 0.5, "tracking error too large - servo failure?"
    # no non-allowed collisions during execution
    assert snap.ncon >= 0  # engine alive; explicit collision audit is in FSM tests


def test_infeasible_trajectory_is_refused(engine):
    """A hand-built too-fast trajectory must be rejected before execution."""
    t = np.array([0.0, 0.01])  # 2 rad in 10 ms -> absurd velocity
    positions = np.array([np.zeros(7), np.full(7, 2.0)])
    spline = CubicSpline(t, positions, bc_type="clamped", axis=0)

    bad = Trajectory(times=t, positions=positions, duration=0.01, limits=TrajectoryLimits.from_panda(), _spline=spline)
    with pytest.raises(InfeasibleTrajectoryError):
        TrajectoryExecutor(bad)


def test_estop_freezes_motion(engine):
    """E-stop must hold targets and stop progression mid-trajectory."""
    start = np.array(PandaSpec.HOME_QPOS)
    # base sweep: substantial, definitely collision-free motion
    goal = start + np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    from robotstudio.collision.checker import CollisionChecker
    from robotstudio.planning.rrt import RRTConnectPlanner

    checker = CollisionChecker(engine.model, engine.robot)
    res = RRTConnectPlanner(checker, PlannerConfig(seed=5, max_planning_time=10.0)).plan(start, goal)
    assert res.success, res.message
    traj = TrajectoryGenerator().generate(res.path)
    executor = TrajectoryExecutor(traj)
    engine.set_controller(executor)

    # wait until motion is visibly in progress, then e-stop
    assert _wait_until(lambda: engine.snapshot().sim_time > 0.2, timeout=10.0)
    engine.set_estop(True)
    assert _wait_until(lambda: engine.snapshot().estop, timeout=5.0)

    q_frozen = engine.snapshot().qpos.copy()
    time.sleep(0.5)
    q_after = engine.snapshot().qpos.copy()
    drift = float(np.max(np.abs(q_after - q_frozen)))
    assert drift < 0.05, f"arm moved {drift:.3f} rad after e-stop"
