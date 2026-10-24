"""Damped Least Squares (DLS) inverse kinematics for the Panda arm.

Method (Siciliano et al., "Robotics: Modelling, Planning and Control", ch. 3
and 6; Buss, "Introduction to Inverse Kinematics with Jacobian
Transposition..." 2004):

    e    = [ p* - p ; log(R* R^T) ]      (world-frame pose error, 6-vector)
    J    = [ J_p ; J_r ]                 (geometric Jacobian, world axes)
    dq   = J^T (J J^T + lambda^2 I)^-1 e
    q    <- clip(q + dq + N q_null)

with adaptive damping ``lambda`` (increased when the error grows, decreased
on progress), per-iteration step clamping, joint-limit clamping and optional
nullspace posture bias toward the mid-range of each joint.

Convergence: ||p* - p|| <= pos_tol AND angle(log(R* R^T)) <= rot_tol.

Complexity: O(iters * (nv^3)) with nv=7 (6x6 solve dominates).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotstudio.config import IKDefaults, PandaSpec
from robotstudio.kinematics.fk import apply_arm_q, body_jacobian
from robotstudio.kinematics.pose import Pose, quat_to_mat, so3_error
from robotstudio.robots.panda import PandaRobot
from robotstudio.utils.logging import get_logger

log = get_logger("ik")


@dataclass
class IKResult:
    success: bool
    arm_q: np.ndarray | None  # (7,) radians, within joint limits when success
    pos_err: float  # m
    rot_err: float  # rad
    iterations: int
    message: str
    seed_used: int = 0
    history: list[float] = field(default_factory=list)  # error norm per iteration


class IKSolver:
    """DLS IK with adaptive damping, nullspace bias and multi-start."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: PandaRobot,
        pos_tol: float = IKDefaults.POS_TOL,
        rot_tol: float = IKDefaults.ROT_TOL,
        max_iters: int = IKDefaults.MAX_ITERS,
        damping: float = IKDefaults.DAMPING,
        nullspace_gain: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.data = data
        self.robot = robot
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.max_iters = max_iters
        self.damping = damping
        self.nullspace_gain = nullspace_gain
        self.rng = np.random.default_rng(seed)
        self._q_center = (robot.jnt_range[:, 0] + robot.jnt_range[:, 1]) / 2.0

    # ------------------------------------------------------------------ #
    def solve(
        self,
        target: Pose,
        q_init: np.ndarray | None = None,
        n_restarts: int = 8,
        validate: "callable | None" = None,
    ) -> IKResult:
        """Solve IK for ``target``.

        Parameters
        ----------
        target : Pose
            Desired end-effector pose (world frame).
        q_init : optional (7,) initial guess (radians).
        n_restarts : additional randomized seeds tried when earlier attempts fail.
        validate : optional callable (7,) -> bool (e.g. collision-free check);
            a candidate that fails validation is rejected and search continues.
        """
        seeds: list[np.ndarray] = []
        if q_init is not None:
            seeds.append(self.robot.clip_to_limits(q_init))
        seeds.append(np.array(PandaSpec.HOME_QPOS))
        while len(seeds) < 1 + n_restarts:
            seeds.append(self.robot.clip_to_limits(self._random_seed()))

        best: IKResult | None = None
        for i, seed in enumerate(seeds):
            result = self._solve_from(target, seed, validate)
            result.seed_used = i
            if result.success:
                return result
            if best is None or result.pos_err + result.rot_err < best.pos_err + best.rot_err:
                best = result
        return best  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    def _solve_from(self, target: Pose, q0: np.ndarray, validate) -> IKResult:
        m, d, robot = self.model, self.data, self.robot
        apply_arm_q(d, robot, q0)
        mujoco.mj_forward(m, d)
        q = robot.clip_to_limits(q0)
        lam = self.damping
        prev_err = np.inf
        history: list[float] = []
        target_R = quat_to_mat(target.quat_wxyz)

        for it in range(1, self.max_iters + 1):
            body = d.body(robot.ee_body_id)
            p_cur = body.xpos.copy()
            R_cur = quat_to_mat(body.xquat.copy())
            e_pos = target.pos - p_cur
            e_rot = so3_error(R_cur, target_R)
            pos_n = float(np.linalg.norm(e_pos))
            rot_n = float(np.linalg.norm(e_rot))
            history.append(pos_n)

            if pos_n <= self.pos_tol and rot_n <= self.rot_tol:
                if validate is not None and not validate(q):
                    return IKResult(False, q, pos_n, rot_n, it, "solution collides with environment (validation failed)", history=history)
                return IKResult(True, q, pos_n, rot_n, it, "converged", history=history)

            jacp, jacr = body_jacobian(m, d, robot)
            J = np.vstack([jacp, jacr])  # (6, 7)
            e = np.concatenate([e_pos, e_rot])

            if not np.all(np.isfinite(J)) or not np.all(np.isfinite(e)):
                return IKResult(False, q, pos_n, rot_n, it, "numerical issue: non-finite Jacobian or error", history=history)

            JJt = J @ J.T + (lam**2) * np.eye(6)
            try:
                dq = J.T @ np.linalg.solve(JJt, e)
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, 10.0)
                continue

            # nullspace bias toward mid-range (secondary task)
            J_pinv = J.T @ np.linalg.inv(JJt)
            dq_null = (np.eye(7) - J_pinv @ J) @ (self.nullspace_gain * (self._q_center - q))

            # step clamping: limit Cartesian progress per iteration
            scale = 1.0
            dp = np.linalg.norm(jacp @ dq)
            dr = np.linalg.norm(jacr @ dq)
            if dp > IKDefaults.MAX_TRANSLATION_STEP:
                scale = min(scale, IKDefaults.MAX_TRANSLATION_STEP / dp)
            if dr > IKDefaults.MAX_ROTATION_STEP:
                scale = min(scale, IKDefaults.MAX_ROTATION_STEP / dr)

            q_new = robot.clip_to_limits(q + scale * dq + dq_null)
            apply_arm_q(d, robot, q_new)
            mujoco.mj_forward(m, d)

            err_now = pos_n + rot_n
            if err_now > prev_err + 1e-9:  # no progress -> increase damping
                lam = min(lam * 4.0, 10.0)
            else:
                lam = max(lam * 0.7, self.damping)
            prev_err = err_now
            q = q_new

        body = d.body(robot.ee_body_id)
        pos_n = float(np.linalg.norm(target.pos - body.xpos))
        rot_n = float(np.linalg.norm(so3_error(quat_to_mat(body.xquat.copy()), target_R)))
        return IKResult(False, q, pos_n, rot_n, self.max_iters, "not converged within iteration limit", history=history)

    def _random_seed(self) -> np.ndarray:
        lo, hi = self.robot.jnt_range[:, 0], self.robot.jnt_range[:, 1]
        return self.rng.uniform(lo, hi)
