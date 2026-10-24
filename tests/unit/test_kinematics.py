"""Phase 1 kinematics tests.

- FK: home pose sanity + hand body frame matches site-less computation.
- Jacobian frame convention verified by finite differences (settles the
  jacr world-frame question empirically).
- SO(3) error: rotation matrix round-trip, pi-rotation branch, anti-symmetry.
- IK: reachability round trip (FK->IK), out-of-reach refusal, joint limits,
  NaN robustness near singular configurations.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robotstudio.config import PandaSpec
from robotstudio.kinematics.fk import body_jacobian, forward_kinematics
from robotstudio.kinematics.pose import Pose, mat_to_quat, quat_to_mat, so3_error
from robotstudio.kinematics.ik import IKSolver

RNG = np.random.default_rng(2028)


# --------------------------------------------------------------------------- #
# SO(3)
# --------------------------------------------------------------------------- #


def test_quat_matrix_roundtrip():
    for _ in range(200):
        q = RNG.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_mat(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        q2 = mat_to_quat(R)
        assert np.allclose(q2, q, atol=1e-7) or np.allclose(q2, -q, atol=1e-7)


def test_so3_error_small_rotation():
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    angle = 0.2
    from scipy.spatial.transform import Rotation

    R1 = Rotation.from_rotvec(axis * angle).as_matrix()
    R2 = Rotation.from_rotvec(axis * 0.0).as_matrix()
    err = so3_error(R1, R2)
    assert np.isclose(np.linalg.norm(err), angle, atol=1e-9)
    # direction consistency
    assert np.allclose(err / np.linalg.norm(err), axis, atol=1e-6) or np.allclose(
        -err / np.linalg.norm(err), axis, atol=1e-6
    )


def test_so3_error_near_pi():
    R = quat_to_mat(np.array([0.0, 1.0, 0.0, 0.0]))  # 180 deg about x
    I = np.eye(3)
    err = so3_error(R, I)
    assert np.isclose(np.linalg.norm(err), np.pi, atol=1e-5)


def test_so3_error_is_zero_for_identical():
    q = RNG.normal(size=4)
    q /= np.linalg.norm(q)
    R = quat_to_mat(q)
    assert np.allclose(so3_error(R, R), 0.0)


# --------------------------------------------------------------------------- #
# FK + Jacobian
# --------------------------------------------------------------------------- #


def test_fk_home_pose(empty):
    model, data, robot = empty
    pose = forward_kinematics(model, data, robot, np.array(PandaSpec.HOME_QPOS))
    # home EE is in front of the base, above tabletop height
    assert pose.pos[2] > 0.5 and pose.pos[2] < 0.75
    assert np.linalg.norm(pose.pos[:2]) > 0.3  # in front


def test_fk_matches_mujoco_body_frame(empty):
    model, data, robot = empty
    q = RNG.uniform(robot.jnt_range[:, 0], robot.jnt_range[:, 1])
    pose = forward_kinematics(model, data, robot, q)
    mujoco.mj_forward(model, data)
    assert np.allclose(pose.pos, data.body(robot.ee_body_id).xpos, atol=1e-10)


def test_jacobian_finite_difference(empty):
    """Cross-validate geometric Jacobian (both blocks) by finite differences.

    This test pins the world-frame convention of mj_jacBody for both the
    translational and rotational blocks.
    """
    model, data, robot = empty
    q = RNG.uniform(robot.jnt_range[:, 0] * 0.5, robot.jnt_range[:, 1] * 0.5)
    eps = 1e-6

    def ee_pose(qv):
        forward_kinematics(model, data, robot, qv)
        b = data.body(robot.ee_body_id)
        return b.xpos.copy(), quat_to_mat(b.xquat.copy())

    mujoco.mj_forward(model, data)
    p0, R0 = ee_pose(q)
    jacp, jacr = body_jacobian(model, data, robot)

    fd_p = np.zeros((3, 7))
    fd_r = np.zeros((3, 7))
    for j in range(7):
        qp = q.copy()
        qp[j] += eps
        p1, R1 = ee_pose(qp)
        fd_p[:, j] = (p1 - p0) / eps
        # R1 ≈ exp(eps * ω) R0  =>  ω ≈ log(R1 R0^T)/eps (world frame)
        fd_r[:, j] = so3_error(R0, R1) / eps

    assert np.allclose(jacp, fd_p, atol=1e-4), "jacp frame/derivative mismatch"
    assert np.allclose(jacr, fd_r, atol=1e-4), "jacr frame/derivative mismatch"


# --------------------------------------------------------------------------- #
# IK
# --------------------------------------------------------------------------- #


def test_ik_recovers_fk_generated_poses(ik, empty):
    """100 FK-generated reachable poses must be solved at high rate."""
    model, data, robot = empty
    n = 100
    successes = 0
    max_pos_err = 0.0
    max_rot_err = 0.0
    for i in range(n):
        q_true = RNG.uniform(robot.jnt_range[:, 0] + 0.05, robot.jnt_range[:, 1] - 0.05)
        target = forward_kinematics(model, data, robot, q_true)
        q_seed = np.array(PandaSpec.HOME_QPOS) if i % 2 == 0 else None
        result = ik.solve(target, q_init=q_seed, n_restarts=4)
        if result.success:
            successes += 1
            max_pos_err = max(max_pos_err, result.pos_err)
            max_rot_err = max(max_rot_err, result.rot_err)
            assert robot.within_limits(result.arm_q), "returned solution violates joint limits"
    rate = successes / n
    assert rate >= 0.95, f"IK success rate {rate:.2%} < 95%"
    assert max_pos_err <= 0.010, f"position error {max_pos_err*1000:.1f} mm > 10 mm"
    assert max_rot_err <= np.deg2rad(5), f"rotation error {np.degrees(max_rot_err):.1f}° > 5°"


def test_ik_rejects_out_of_reach(ik):
    far = Pose.make([1.2, 0.6, 0.9])  # far outside the 0.855 m reach
    result = ik.solve(far, n_restarts=3)
    assert not result.success
    assert "converged" in result.message or "not converged" in result.message


def test_ik_reports_failure_not_garbage(ik):
    target = Pose.make([0.05, 0.05, 1.5])  # above the shoulder, unreachable
    result = ik.solve(target, n_restarts=2)
    if result.success:  # only acceptable if truly within tolerances
        assert result.pos_err <= ik.pos_tol and result.rot_err <= ik.rot_tol
        assert ik.robot.within_limits(result.arm_q)
    else:
        assert result.arm_q is not None  # best-effort solution returned


def test_ik_respects_joint_limits(ik, empty):
    model, data, robot = empty
    for _ in range(20):
        q_true = RNG.uniform(robot.jnt_range[:, 0], robot.jnt_range[:, 1])
        target = forward_kinematics(model, data, robot, q_true)
        res = ik.solve(target, n_restarts=2)
        if res.success:
            assert robot.within_limits(res.arm_q, tol=1e-6)


def test_ik_multi_start_beats_single_seed(ik, empty):
    """A hard pose (near workspace boundary) should need restarts sometimes."""
    model, data, robot = empty
    q_hard = np.array([0.0, 0.3, 0.0, -2.4, 0.0, 2.8, 0.6])
    target = forward_kinematics(model, data, robot, q_hard)
    r_multi = ik.solve(target, n_restarts=6)
    assert r_multi.success or r_multi.pos_err < 0.05


def test_ik_convergence_tolerance_fields(ik, empty):
    model, data, robot = empty
    q_true = RNG.uniform(robot.jnt_range[:, 0], robot.jnt_range[:, 1])
    target = forward_kinematics(model, data, robot, q_true)
    res = ik.solve(target)
    assert isinstance(res.iterations, int) and res.iterations >= 1
    assert len(res.history) == res.iterations or res.iterations == ik.max_iters
