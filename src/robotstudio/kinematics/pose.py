"""Rigid-body pose math (SO(3) utilities).

Conventions
-----------
- MuJoCo quaternions are (w, x, y, z).
- All rotation errors use the matrix logarithm / axis-angle of
  ``R_err = R_target @ R_current.T`` (world-frame error), never per-axis
  Euler differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    """Rigid transform: position (m, world frame) + quaternion (w,x,y,z)."""

    pos: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)

    @staticmethod
    def make(pos, quat_wxyz=None) -> Pose:
        pos = np.asarray(pos, dtype=float).reshape(3)
        if quat_wxyz is None:
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        quat_wxyz = np.asarray(quat_wxyz, dtype=float).reshape(4)
        quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
        return Pose(pos=pos, quat_wxyz=quat_wxyz)

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose:
        m = np.asarray(matrix, dtype=float).reshape(4, 4)
        return Pose.make(m[:3, 3], mat_to_quat(m[:3, :3]))

    def as_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transform."""
        m = np.eye(4)
        m[:3, :3] = quat_to_mat(self.quat_wxyz)
        m[:3, 3] = self.pos
        return m

    def inverse(self) -> Pose:
        R = quat_to_mat(self.quat_wxyz)
        return Pose.make(-(R.T @ self.pos), mat_to_quat(R.T))


def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q_wxyz, dtype=float) / np.linalg.norm(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (w,x,y,z). Numerically robust branch picking."""
    m = np.asarray(R, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def so3_error(current_R: np.ndarray, target_R: np.ndarray) -> np.ndarray:
    """World-frame axis-angle error rotating ``current_R`` onto ``target_R``.

    Returns a 3-vector: direction is the rotation axis, magnitude is the angle
    in radians, always within [0, pi].
    """
    R_err = np.asarray(target_R) @ np.asarray(current_R).T
    angle = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    if angle > np.pi - 1e-6:  # near pi: axis from the quaternion (off-diagonal part degenerates)
        q = mat_to_quat(R_err)
        axis = q[1:] / max(np.linalg.norm(q[1:]), 1e-12)
        return axis * angle
    axis = np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
    axis = axis / (2.0 * np.sin(angle))
    return axis * angle


def so3_error_from_quat(current_quat_wxyz: np.ndarray, target_quat_wxyz: np.ndarray) -> np.ndarray:
    return so3_error(quat_to_mat(current_quat_wxyz), quat_to_mat(target_quat_wxyz))


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit quaternion (w,x,y,z)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q
