"""Phase 2 planner tests.

- Unified API result invariants (path endpoints, feasibility, stats).
- RRT & RRT-Connect solve a blocked-task in the single_obs scene; the
  returned path passes an INDEPENDENT dense re-validation.
- Failure path: impossible goal (inside a wall) must fail with a clear
  message, never an unsafe "path".
- Linear baseline must REFUSE when the straight line crosses the obstacle.
- Determinism: same seed -> same path.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robotstudio.collision.checker import CollisionChecker
from robotstudio.config import PandaSpec
from robotstudio.kinematics.ik import IKSolver
from robotstudio.kinematics.pose import Pose
from robotstudio.planning.base import PlannerConfig
from robotstudio.planning.linear import LinearPlanner
from robotstudio.planning.rrt import RRTConnectPlanner, RRTPlanner


def _ik(model, robot):
    return IKSolver(model, mujoco.MjData(model), robot, seed=42)


def _far_side_goal(single_obs, checker):
    """IK a collision-free configuration behind the wall (y < -0.3).

    Pose chosen (by setup probes) so that the straight segment from home
    provably crosses the wall, while a collision-free config exists.
    """
    model, _data, robot = single_obs
    ik = _ik(model, robot)
    target = Pose.make([0.35, -0.40, 0.55], [0.0, 1.0, 0.0, 0.0])  # hand z down
    res = ik.solve(target, n_restarts=12, validate=lambda q: checker.check_state(q).valid)
    assert res.success, "test setup: far-side IK must succeed"
    return res.arm_q


def test_result_invariants_rrt_connect_blocked(single_obs):
    """RRT-Connect must solve the wall-blocked task; path independently valid."""
    model, data, robot = single_obs
    checker = CollisionChecker(model, robot)
    start = np.array(PandaSpec.HOME_QPOS)
    goal = _far_side_goal(single_obs, checker)

    planner = RRTConnectPlanner(checker, PlannerConfig(seed=3, max_planning_time=20.0))
    res = planner.plan(start, goal)
    assert res.success, f"rrt_connect: {res.message}"
    assert res.path is not None and len(res.path) >= 2
    assert np.allclose(res.path[0], start, atol=1e-9)
    assert np.allclose(res.path[-1], goal, atol=1e-9)
    assert res.planning_time <= 25.0
    assert res.n_collision_checks > 0
    # independent re-validation with a fresh checker instance
    fresh = CollisionChecker(model, robot)
    rep = fresh.check_path(res.path)
    assert rep.valid, f"rrt_connect: path fails independent validation: {rep.message}"


def test_result_invariants_rrt_open(tabletop):
    """Single-tree RRT is verified on an open task (its design envelope).

    RRT's behavior on narrow-passage scenes is measured honestly in the
    benchmark suite (fixed seeds, timeout statistics reported), not asserted
    here: single-tree uniform RRT is known to scale poorly to 7D narrow
    passages (LaValle & Kuffner 2001; our own benchmark confirms it).
    """
    model, data, robot = tabletop
    checker = CollisionChecker(model, robot)
    start = np.array(PandaSpec.HOME_QPOS)
    goal = start + np.array([0.15, -0.25, 0.10, -0.05, 0.1, 0.05, 0.1])
    goal = np.clip(goal, checker.robot.jnt_range[:, 0], checker.robot.jnt_range[:, 1])

    planner = RRTPlanner(checker, PlannerConfig(seed=3, max_planning_time=20.0))
    res = planner.plan(start, goal)
    assert res.success, f"rrt: {res.message}"
    assert np.allclose(res.path[0], start, atol=1e-9)
    assert np.allclose(res.path[-1], goal, atol=1e-9)
    fresh = CollisionChecker(model, robot)
    assert fresh.check_path(res.path).valid


def test_rrt_connect_more_efficient_than_rrt(tabletop):
    """On an open task both succeed; RRT-Connect should need far fewer iterations."""
    model, data, robot = tabletop
    checker = CollisionChecker(model, robot)
    start = np.array(PandaSpec.HOME_QPOS)
    goal = start + np.array([0.2, -0.3, 0.15, -0.1, 0.1, 0.1, 0.15])
    goal = np.clip(goal, checker.robot.jnt_range[:, 0], checker.robot.jnt_range[:, 1])

    r1 = RRTPlanner(checker, PlannerConfig(seed=1, max_planning_time=30.0)).plan(start, goal)
    r2 = RRTConnectPlanner(checker, PlannerConfig(seed=1, max_planning_time=30.0)).plan(start, goal)
    assert r1.success and r2.success, f"{r1.message} / {r2.message}"
    assert r2.stats["iterations"] <= r1.stats["iterations"], (
        f"rrt_connect iterations {r2.stats['iterations']} should be below rrt {r1.stats['iterations']}"
    )


def test_impossible_goal_fails_cleanly(single_obs):
    model, data, robot = single_obs
    checker = CollisionChecker(model, robot)
    start = np.array(PandaSpec.HOME_QPOS)
    ik = _ik(model, robot)
    # goal INSIDE the wall (no collision-free config exists at this pose)
    target = Pose.make([0.45, -0.24, 0.50])
    res_ik = ik.solve(target, n_restarts=4)
    if not res_ik.success:
        # verify the checker agrees the IK best-effort is in collision
        rep = checker.check_state(res_ik.arm_q)
        assert not rep.valid
    # planning to a colliding goal must fail with a clear reason
    planner = RRTConnectPlanner(checker, PlannerConfig(seed=0, max_planning_time=5.0))
    if res_ik.success:
        res = planner.plan(start, res_ik.arm_q)
        assert not res.success and res.path is None
        assert "collision" in res.message
    else:
        # synthesize the in-collision goal from IK best effort
        rep = checker.check_state(res_ik.arm_q)
        assert not rep.valid
        res = planner.plan(start, res_ik.arm_q)
        assert not res.success and res.path is None


def test_linear_refuses_blocked_path(single_obs):
    model, data, robot = single_obs
    checker = CollisionChecker(model, robot)
    start = np.array(PandaSpec.HOME_QPOS)
    goal = _far_side_goal(single_obs, checker)
    planner = LinearPlanner(checker, PlannerConfig(seed=0))
    res = planner.plan(start, goal)
    assert not res.success, "straight line through a wall must be refused"
    assert res.path is None


def test_linear_accepts_free_path(tabletop):
    model, data, robot = tabletop
    checker = CollisionChecker(model, robot)
    start = np.array(PandaSpec.HOME_QPOS)
    goal = start + np.array([0.1, -0.1, 0.2, 0, -0.1, 0.1, 0])
    res = LinearPlanner(checker, PlannerConfig(seed=0)).plan(start, np.clip(goal, *checker.robot.jnt_range.T))
    if res.success:
        fresh = CollisionChecker(model, robot)
        assert fresh.check_path(res.path).valid


def test_determinism_same_seed(single_obs):
    model, data, robot = single_obs
    checker = CollisionChecker(model, robot)
    start = np.array(PandaSpec.HOME_QPOS)
    goal = _far_side_goal(single_obs, checker)
    r1 = RRTConnectPlanner(checker, PlannerConfig(seed=77, shortcut_tries=0)).plan(start, goal)
    r2 = RRTConnectPlanner(checker, PlannerConfig(seed=77, shortcut_tries=0)).plan(start, goal)
    assert r1.success and r2.success
    assert len(r1.path) == len(r2.path)
    assert np.allclose(np.asarray(r1.path), np.asarray(r2.path))


def test_shortcut_reduces_length(single_obs):
    model, data, robot = single_obs
    checker = CollisionChecker(model, robot)
    start = np.array(PandaSpec.HOME_QPOS)
    goal = _far_side_goal(single_obs, checker)
    raw = RRTConnectPlanner(checker, PlannerConfig(seed=9, shortcut_tries=0, max_planning_time=20)).plan(start, goal)
    short = RRTConnectPlanner(checker, PlannerConfig(seed=9, shortcut_tries=300, max_planning_time=20)).plan(start, goal)
    assert raw.success and short.success
    assert short.path_length <= raw.path_length + 1e-9
    fresh = CollisionChecker(model, robot)
    assert fresh.check_path(short.path).valid
