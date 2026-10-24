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
    """Controller that follows a :class:`Trajectory` in real simulation time."""

    def __init__(self, trajectory: Trajectory, limits: TrajectoryLimits | None = None) -> None:
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
        self._err_sq_sum = 0.0
        self._err_n = 0

    # ------------------------------------------------------------------ #
    def update(self, ctx: EngineContext, dt: float) -> np.ndarray:
        if self.start_time is None:
            self.start_time = ctx.time
            self.status_text = "executing"
        t = ctx.time - self.start_time

        if t >= self.trajectory.duration:
            q_final = self.trajectory.sample(self.trajectory.duration)
            self._record(ctx, q_final)
            if not self.stats.finished:
                self.stats.finished = True
                self.stats.success = True
                self.stats.executed_time = self.trajectory.duration
                self.status_text = "completed"
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
