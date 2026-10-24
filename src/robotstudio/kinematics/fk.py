"""Forward kinematics helpers on top of MuJoCo.

All functions operate on a caller-owned (MjModel, MjData) pair so they can be
used both from the live engine thread and from planning worker threads with
their own shadow data.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robotstudio.robots.panda import PandaRobot
from robotstudio.kinematics.pose import Pose


def apply_arm_q(data: mujoco.MjData, robot: PandaRobot, arm_q: np.ndarray, finger_opening: float = 0.04) -> None:
    """Write a 7-DoF arm configuration (+ finger opening, m/finger) into qpos."""
    data.qpos[robot.arm_qposadr] = arm_q
    data.qpos[robot.finger_qposadr] = finger_opening


def forward_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: PandaRobot,
    arm_q: np.ndarray,
    finger_opening: float = 0.04,
) -> Pose:
    """End-effector (hand body) pose for an arm configuration.

    Runs ``mj_forward`` on ``data``; the caller must not share this MjData
    with a concurrently stepping engine.
    """
    apply_arm_q(data, robot, arm_q, finger_opening)
    mujoco.mj_forward(model, data)
    body = data.body(robot.ee_body_id)
    return Pose.make(body.xpos.copy(), body.xquat.copy())


def body_jacobian(model: mujoco.MjModel, data: mujoco.MjData, robot: PandaRobot) -> tuple[np.ndarray, np.ndarray]:
    """(jacp, jacr) of the EE body restricted to arm DoFs, each (3, 7).

    Both blocks are expressed in world axes (verified against finite
    differences in tests/unit/test_kinematics.py).
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, robot.ee_body_id)
    arm_dofs = robot.arm_dofadr
    return jacp[:, arm_dofs].copy(), jacr[:, arm_dofs].copy()
