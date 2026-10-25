"""Phase 2 collision tests.

- Whitelist semantics: known-good state is valid; known-colliding state fails.
- Parity: checker's mj_kinematics+mj_collision finds the same contacts as
  mj_forward (guards against missing broadphase data).
- Mid-segment penetration: a segment passing through a wall must be caught
  even when BOTH endpoints are collision-free (the key discrete-check trap).
- Gripper-cube contacts are whitelist-allowed only when holding_cube=True.
- Joint-limit violation is reported.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robotstudio.collision.checker import CollisionChecker
from robotstudio.config import PandaSpec
from robotstudio.kinematics.ik import IKSolver
from robotstudio.kinematics.pose import Pose
from robotstudio.manipulation.grasp import grasp_pose, pregrasp_pose


def _make_checker(model, robot):
    return CollisionChecker(model, robot)


def test_home_pose_is_collision_free(checker):
    rep = checker.check_state(np.array(PandaSpec.HOME_QPOS))
    assert rep.valid, rep.message


def test_endpoint_collision_detected(single_obs):
    model, data, robot = single_obs
    checker = _make_checker(model, robot)
    # IK straight into the wall interior (center of the obstacle box)
    ik = IKSolver(model, mujoco.MjData(model), robot, seed=1)
    target = Pose.make([0.45, -0.24, 0.50])
    res = ik.solve(target, n_restarts=6)
    if res.success:  # IK found the (colliding) configuration -> checker must reject it
        rep = checker.check_state(res.arm_q)
        assert not rep.valid
        assert any("wall_obstacle" in a or "wall_obstacle" in b for a, b, _ in rep.violations)


def test_checker_matches_mj_forward(tabletop):
    model, data, robot = tabletop
    checker = _make_checker(model, robot)
    rng = np.random.default_rng(5)
    for _ in range(30):
        q = rng.uniform(robot.jnt_range[:, 0], robot.jnt_range[:, 1])
        # shadow pipeline
        rep = checker.check_state(q)
        # mj_forward pipeline on a fresh data
        d2 = mujoco.MjData(model)
        d2.qpos[robot.arm_qposadr] = q
        d2.qpos[robot.finger_qposadr] = 0.04
        mujoco.mj_forward(model, d2)
        # compare allowed-vs-violating classification of contact pairs
        def classify(dd):
            bad = 0
            for i in range(dd.ncon):
                c = dd.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if not checker.pair_is_allowed(g1, g2, holding_cube=False):
                    bad += 1
            return bad

        assert (rep.valid) == (classify(d2) == 0), f"parity mismatch for q={q}"


def test_mid_segment_penetration_caught(single_obs):
    """Both endpoints free, straight segment passes through the wall -> fail.

    Setup asserts (no early return): if the far-side IK cannot produce a
    collision-free goal, the test FAILS loudly rather than passing vacuously.
    """
    model, data, robot = single_obs
    checker = _make_checker(model, robot)
    home = np.array(PandaSpec.HOME_QPOS)
    assert checker.check_state(home).valid

    ik = IKSolver(model, mujoco.MjData(model), robot, seed=42)
    # goal beyond the wall (wall: x 0.25..0.65, y -0.27..-0.21, z 0.3..0.7);
    # pose chosen by setup probes so a collision-free config exists
    target = Pose.make([0.35, -0.40, 0.55], [0.0, 1.0, 0.0, 0.0])
    res = ik.solve(target, n_restarts=12, validate=lambda q: checker.check_state(q).valid)
    assert res.success, f"test setup must produce a collision-free far-side config: {res.message}"
    q_goal = res.arm_q
    assert checker.check_state(q_goal).valid
    rep = checker.check_segment(home, q_goal)
    # the straight line from home to a behind-the-wall goal must cross it
    assert not rep.valid, "discrete check missed a mid-segment collision"
    assert rep.first_bad_sample is not None and rep.first_bad_sample > 0


def test_gripper_cube_contact_whitelist(tabletop):
    model, data, robot = tabletop
    checker = _make_checker(model, robot)
    # Production grasp setup, mirroring the FSM: solve the pregrasp hover
    # first, then the grasp pose seeded from it (random multi-start alone
    # often lands on elbow-through-table branches for this low pose).
    ik = IKSolver(model, mujoco.MjData(model), robot, seed=11)
    cube_pos = data.body("cube").xpos
    pre = pregrasp_pose(cube_pos, 0.025)
    res_pre = ik.solve(pre, n_restarts=12, validate=lambda q: checker.check_state(q).valid)
    assert res_pre.success, f"test setup must solve the pregrasp pose: {res_pre.message}"
    grasp = grasp_pose(cube_pos, 0.025)
    res = ik.solve(grasp, q_init=res_pre.arm_q, n_restarts=12,
                   validate=lambda q: checker.check_state(q).valid)
    assert res.success, f"test setup must solve the grasp pose collision-free: {res.message}"

    def violation_pairs(rep):
        return {frozenset((a, b)) for a, b, _ in rep.violations}

    # Fingers squeezed inside the cube (finger_opening 5 mm << cube width).
    rep_free = checker.check_state(res.arm_q, finger_opening=0.005, holding_cube=False)
    rep_hold = checker.check_state(res.arm_q, finger_opening=0.005, holding_cube=True)
    grasp_pairs = {frozenset(("left_finger", "cube")), frozenset(("right_finger", "cube")), frozenset(("hand", "cube"))}
    # Free mode: finger-cube interpenetration MUST be flagged (a free path may
    # never touch the object).
    assert violation_pairs(rep_free) & grasp_pairs, "free mode missed finger-cube interpenetration"
    # Holding mode: the squeeze IS the grasp and must be whitelisted.
    assert not (violation_pairs(rep_hold) & grasp_pairs), "holding mode flagged the grasp itself"
    # Semantic split (see CollisionChecker._pair_allowed): in FREE mode the
    # cube resting on its support is physics -> NO cube-env violations; in
    # HOLD mode the carried cube must clear the environment -> cube-env
    # contacts ARE violations. Arm-vs-world (non-cube) violations must be
    # identical across modes (same configuration).
    def cube_env_pairs(pairs):
        gripper = {"left_finger", "right_finger", "hand"}
        return {p for p in pairs if "cube" in p and not (set(p) & gripper)}

    assert not (violation_pairs(rep_free) & cube_env_pairs(violation_pairs(rep_free))), (
        "free mode flagged the resting object's support contacts"
    )
    assert violation_pairs(rep_hold) & cube_env_pairs(violation_pairs(rep_hold)), (
        "holding mode should flag the carried cube touching the table"
    )
    arm_world_free = violation_pairs(rep_free) - grasp_pairs - cube_env_pairs(violation_pairs(rep_free))
    arm_world_hold = violation_pairs(rep_hold) - grasp_pairs - cube_env_pairs(violation_pairs(rep_hold))
    assert arm_world_free == arm_world_hold, "arm-vs-world violations inconsistent between modes"


def test_joint_limit_violation_reported(checker):
    bad = checker.robot.jnt_range[:, 1] + 0.1
    rep = checker.check_state(bad)
    assert not rep.valid
    assert rep.violations[0][0] == "joint_limits"


def test_check_path_validates_every_segment(tabletop):
    model, data, robot = tabletop
    checker = _make_checker(model, robot)
    home = np.array(PandaSpec.HOME_QPOS)
    # a short, definitely-free wiggle around home
    path = [home + 0.02 * i for i in range(4)]
    rep = checker.check_path(path)
    assert rep.valid
    assert rep.n_samples >= len(path)
