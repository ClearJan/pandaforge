"""Franka Panda model abstraction over a loaded MuJoCo MjModel.

Resolves all indices by name so scene XMLs can freely reorder bodies.
Units: radians / meters / ctrl as defined in assets/robots/panda/panda.xml.
"""

from __future__ import annotations

import numpy as np
from mujoco import MjModel

from robotstudio.config import PandaSpec


class RobotModelError(RuntimeError):
    """Raised when the loaded model does not match the expected Panda layout."""


class PandaRobot:
    """Name-resolved view of a Panda scene model (arm + gripper)."""

    def __init__(self, model: MjModel, spec: type = PandaSpec) -> None:
        self.spec = spec
        self.model = model
        self._resolve()

    def _resolve(self) -> None:
        m = self.model
        # --- joints ---
        self.arm_joint_ids = []
        for name in self.spec.ARM_JOINT_NAMES:
            jid = m.joint(name).id
            if m.jnt_type[jid] != 3:  # mjJNT_HINGE
                raise RobotModelError(f"joint {name!r} is not a hinge")
            self.arm_joint_ids.append(jid)
        self.finger_joint_ids = [m.joint(n).id for n in self.spec.FINGER_JOINT_NAMES]

        self.arm_qposadr = np.array([m.jnt_qposadr[j] for j in self.arm_joint_ids], dtype=int)
        self.arm_dofadr = np.array([m.jnt_dofadr[j] for j in self.arm_joint_ids], dtype=int)
        self.finger_qposadr = np.array([m.jnt_qposadr[j] for j in self.finger_joint_ids], dtype=int)
        self.finger_dofadr = np.array([m.jnt_dofadr[j] for j in self.finger_joint_ids], dtype=int)

        if m.nu != self.spec.N_CTRL:
            raise RobotModelError(
                f"model layout mismatch: nu={m.nu}, expected {self.spec.N_CTRL} "
                "(scene objects may add qpos/nv dimensions, but actuators must be Panda-only)"
            )

        self.jnt_range = m.jnt_range[self.arm_joint_ids].copy()  # (7, 2) radians
        self.arm_ctrladr = np.arange(7, dtype=int)
        self.gripper_ctrladr = 7

        # --- end-effector body ---
        try:
            self.ee_body_id = m.body(self.spec.EE_BODY).id
        except KeyError as exc:
            raise RobotModelError(f"missing end-effector body {self.spec.EE_BODY!r}") from exc

        # --- fingers (collision pads) & gripper geoms ---
        self.finger_body_ids = [m.body("left_finger").id, m.body("right_finger").id]

        # Arm actuator ctrl ranges (actuators 1..7 map 1:1 to joints 1..7).
        self.ctrl_range = m.actuator_ctrlrange[:7].copy()

    # ------------------------------------------------------------------ #
    # conversions between the 7-DoF arm space and full model arrays
    # ------------------------------------------------------------------ #
    @property
    def n_arm(self) -> int:
        return self.spec.N_ARM_JOINTS

    def make_qpos(self, arm_q: np.ndarray, fingers: tuple[float, float] = (0.04, 0.04)) -> np.ndarray:
        """Full qpos vector (9,) from arm configuration (7,) and finger positions (m)."""
        qpos = np.zeros(self.model.nq)
        qpos[self.arm_qposadr] = arm_q
        qpos[self.finger_qposadr] = fingers
        return qpos

    def arm_q_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return np.asarray(qpos)[self.arm_qposadr].copy()

    def set_arm_ctrl(self, ctrl: np.ndarray, arm_q: np.ndarray) -> np.ndarray:
        """Write arm targets into a full ctrl vector (8,); gripper left untouched."""
        ctrl = np.array(ctrl, dtype=float)
        ctrl[self.arm_ctrladr] = arm_q
        return ctrl

    def within_limits(self, arm_q: np.ndarray, tol: float = 1e-6) -> bool:
        """Check a 7-DoF configuration against joint limits."""
        q = np.asarray(arm_q, dtype=float)
        return bool(np.all(q >= self.jnt_range[:, 0] - tol) and np.all(q <= self.jnt_range[:, 1] + tol))

    def clip_to_limits(self, arm_q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(arm_q, dtype=float), self.jnt_range[:, 0], self.jnt_range[:, 1])

    def gripper_ctrl_from_opening(self, opening_m: float) -> float:
        """Convert desired per-finger opening (m) to gripper actuator ctrl units."""
        return float(np.clip(opening_m / self.spec.GRIPPER_CTRL_TO_M, 0.0, 255.0))
