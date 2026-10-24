"""Trajectory time parameterization.

Pipeline (method: time-scaled cubic spline):

1. Allocate per-segment durations from per-joint trapezoidal lower bounds
   (displacement/vmax, plus vmax/amax ramp terms), synchronized to the
   slowest joint per segment - a conservative but complete bound.
2. Fit a clamped cubic spline (zero endpoint velocities) through the
   time-stamped waypoints.
3. Numerically verify joint velocity/acceleration limits at fine resolution.
   If violated, scale ALL durations by the feasibility ratio and refit
   (iterate until feasible or give up with a clear error).

The result is C2 within segments, C1 across waypoints, with genuinely
continuous velocity - not stop-and-go. Assumptions/limits: waypoints are
spatially dense enough for cubic interpolation between them to stay near the
planned straight segments (planner output after shortcutting satisfies
this); the numeric limit check is authoritative, not the visual smoothness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class TrajectoryLimits:
    vmax: np.ndarray  # (7,) rad/s
    amax: np.ndarray  # (7,) rad/s^2

    @staticmethod
    def from_panda() -> "TrajectoryLimits":
        from robotstudio.config import PandaSpec

        return TrajectoryLimits(
            vmax=np.asarray(PandaSpec.MAX_JOINT_VEL, dtype=float),
            amax=np.asarray(PandaSpec.MAX_JOINT_ACC, dtype=float),
        )


@dataclass
class Trajectory:
    times: np.ndarray  # (N,) cumulative waypoint times, s
    positions: np.ndarray  # (N, 7) radians
    duration: float
    limits: TrajectoryLimits
    _spline: CubicSpline

    def sample(self, t: float) -> np.ndarray:
        """Joint position at time t (clamped to [0, duration])."""
        t = float(np.clip(t, 0.0, self.duration))
        return np.clip(self._spline(t), *self._q_bounds())

    def sample_velocity(self, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, self.duration))
        return self._spline(t, 1)

    def _q_bounds(self):
        lo = np.minimum(self.positions[0], self.positions.min(axis=0)) - 1e-9
        hi = np.maximum(self.positions[0], self.positions.max(axis=0)) + 1e-9
        return lo, hi

    def constraint_violation(self, dt: float = 0.002) -> tuple[float, float]:
        """(max |v| / vmax, max |a| / amax) over a dense grid; 1.0 == at limit."""
        ts = np.arange(0.0, self.duration + dt / 2, dt)
        v = self._spline(ts, 1)
        a = self._spline(ts, 2)
        v_ratio = float(np.max(np.abs(v) / self.limits.vmax))
        a_ratio = float(np.max(np.abs(a) / self.limits.amax))
        return v_ratio, a_ratio

    def feasible(self, dt: float = 0.002, margin: float = 1.0) -> bool:
        v_ratio, a_ratio = self.constraint_violation(dt)
        return v_ratio <= margin and a_ratio <= margin


class TrajectoryGenerationError(RuntimeError):
    pass


def _segment_time(dq: np.ndarray, vmax: np.ndarray, amax: np.ndarray) -> float:
    """Synchronized trapezoidal lower bound for one segment (per joint, max)."""
    d = np.abs(dq)
    with np.errstate(divide="ignore", invalid="ignore"):
        ramp = vmax**2 / amax  # displacement covered during full accel+decel
        t_full = d / vmax + vmax / amax  # trapezoid if d >= ramp
        t_short = 2.0 * np.sqrt(d / amax)  # triangular if d < ramp
        t = np.where(d >= ramp, t_full, t_short)
    return float(np.max(t))


class TrajectoryGenerator:
    def __init__(self, limits: TrajectoryLimits | None = None, max_scale_iterations: int = 30) -> None:
        self.limits = limits or TrajectoryLimits.from_panda()
        self.max_scale_iterations = max_scale_iterations

    def generate(self, path: list[np.ndarray] | np.ndarray) -> Trajectory:
        """Time-parameterize a (N,7) waypoint path into a feasible Trajectory."""
        arr = np.asarray(path, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise TrajectoryGenerationError(f"path must be (N,7), got shape {arr.shape}")
        if arr.shape[0] < 2:
            raise TrajectoryGenerationError("path needs at least 2 waypoints")
        if not np.all(np.isfinite(arr)):
            raise TrajectoryGenerationError("path contains non-finite values")

        seg_dq = np.diff(arr, axis=0)
        seg_t = np.array([_segment_time(dq, self.limits.vmax, self.limits.amax) for dq in seg_dq])
        seg_t = np.maximum(seg_t, 1e-3)
        times = np.concatenate([[0.0], np.cumsum(seg_t)])

        scale = 1.0
        for _ in range(self.max_scale_iterations):
            spline = CubicSpline(times * scale, arr, bc_type="clamped", axis=0)
            traj = Trajectory(
                times=times * scale,
                positions=arr,
                duration=float(times[-1] * scale),
                limits=self.limits,
                _spline=spline,
            )
            v_ratio, a_ratio = traj.constraint_violation()
            if v_ratio <= 1.0 and a_ratio <= 1.0:
                return traj
            scale *= max(v_ratio, a_ratio) * 1.05  # grow until feasible

        raise TrajectoryGenerationError(
            f"cannot satisfy joint velocity/acceleration limits (v_ratio={v_ratio:.2f}, a_ratio={a_ratio:.2f})"
        )
