"""Trajectory execution controller: streams time-parameterized setpoints.

Runs inside the engine thread (Controller protocol). Each physics step it
samples the trajectory at (sim_time - start_time) and commands that joint
vector to the position servos, recording reference-vs-actual tracking error
statistics. It refuses to start on trajectories that violate joint
velocity/acceleration limits (validated again, independently of the
generator).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotstudio.simulation.engine import EngineContext
from robotstudio.trajectory.timing import Trajectory, TrajectoryLimits


class InfeasibleTrajectoryError(RuntimeError):
    pass


@dataclass
class ExecutionStats:
    finished: bool = False
    duration: float = 0.0
    executed_time: float = 0.0
    max_joint_tracking_err: float = 0.0  # rad, target vs actual
    rms_joint_tracking_err: float = 0.0
    max_ee_pos_err: float = 0.0  # m, reference EE vs actual EE
    success: bool = False


class TrajectoryExecutor:
    """Controller that follows a :class:`Trajectory` in real simulation time.

    Completion semantics: the trajectory is DONE only when (a) the time
    parameterization has played out AND (b) the physical arm has converged to
    within ``settle_tol`` radians of the final waypoint (checked every step,
    ``settle_timeout`` budget). Without (b), states downstream would hold the
    lagging actual configuration and freeze servo error mid-motion.
    """

    def __init__(
        self,
        trajectory: Trajectory,
        limits: TrajectoryLimits | None = None,
        settle_tol: float = 0.03,
        settle_timeout: float = 2.0,
    ) -> None:
        limits = limits or trajectory.limits
        v_ratio, a_ratio = trajectory.constraint_violation()
        if v_ratio > 1.0 or a_ratio > 1.0:
            raise InfeasibleTrajectoryError(
                f"trajectory violates limits: v_ratio={v_ratio:.2f}, a_ratio={a_ratio:.2f} - refusing to execute"
            )
        self.trajectory = trajectory
        self.start_time: float | None = None
        self.status_text = "trajectory ready"
        self.stats = ExecutionStats(duration=trajectory.duration)
        self.settle_tol = settle_tol
        self.settle_timeout = settle_timeout
        self._settle_elapsed = 0.0
        self._err_sq_sum = 0.0
        self._err_n = 0

    # ------------------------------------------------------------------ #
    @property
    def final_target(self) -> np.ndarray:
        """The trajectory's final waypoint (the static hold target)."""
        return self.trajectory.sample(self.trajectory.duration)

    def update(self, ctx: EngineContext, dt: float) -> np.ndarray:
        if self.start_time is None:
            self.start_time = ctx.time
            self.status_text = "executing"
        t = ctx.time - self.start_time

        if t >= self.trajectory.duration:
            q_final = self.trajectory.sample(self.trajectory.duration)
            self._record(ctx, q_final)
            settle_err = float(np.max(np.abs(q_final - ctx.arm_q)))
            self._settle_elapsed += dt
            if settle_err <= self.settle_tol:
                if not self.stats.finished:
                    self.stats.finished = True
                    self.stats.success = True
                    self.stats.executed_time = self.trajectory.duration
                    self.status_text = "completed"
            elif self._settle_elapsed > self.settle_timeout:
                if not self.stats.finished:
                    self.stats.finished = True
                    self.stats.success = False
                    self.status_text = f"did not settle (err {settle_err:.3f} rad)"
            else:
                self.status_text = f"settling ({settle_err:.3f} rad to go)"
            return q_final

        q_ref = self.trajectory.sample(t)
        self._record(ctx, q_ref)
        self.stats.executed_time = t
        self.status_text = f"executing {t:.2f}/{self.trajectory.duration:.2f}s"
        return q_ref

    def reference_joints(self, ctx: EngineContext) -> np.ndarray | None:
        """Reference joint vector for tracking-error metrics (engine thread)."""
        if self.start_time is None:
            return None
        t = np.clip(ctx.time - self.start_time, 0, self.trajectory.duration)
        return self.trajectory.sample(t)

    def _record(self, ctx: EngineContext, q_ref: np.ndarray) -> None:
        q_act = ctx.arm_q
        err = float(np.max(np.abs(q_ref - q_act)))
        self.stats.max_joint_tracking_err = max(self.stats.max_joint_tracking_err, err)
        self._err_sq_sum += float(np.sum((q_ref - q_act) ** 2))
        self._err_n += len(q_ref)
        self.stats.rms_joint_tracking_err = float(np.sqrt(self._err_sq_sum / max(self._err_n, 1)))
