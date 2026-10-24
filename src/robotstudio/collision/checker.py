"""Collision checking on MuJoCo's real collision geometry.

The checker owns a private ``MjData`` ("shadow") so it can run from any
thread while the engine keeps stepping. Every reported contact pair is
matched against an EXPLICIT whitelist (body-name pairs). Anything outside
the whitelist is a violation - there is no implicit allowance.

Modes:
* ``holding_cube=False`` (default, path planning): any gripper-cube contact
  is a violation - the free path must not touch the object.
* ``holding_cube=True`` (grasp/transfer): finger-cube contacts are allowed
  (they ARE the grasp); cube-environment contacts are checked normally.

Path segments are checked with dense interpolation (default max 0.02 rad
between samples) so mid-segment penetrations are caught, not just endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotstudio.robots.panda import PandaRobot
from robotstudio.utils.logging import get_logger

log = get_logger("collision")

MAX_SEGMENT_STEP_RAD = 0.02


@dataclass
class CollisionReport:
    valid: bool
    n_samples: int = 0
    violations: list[tuple[str, str, float]] = field(default_factory=list)  # (body1, body2, dist) worst per pair
    first_bad_sample: int | None = None

    @property
    def message(self) -> str:
        if self.valid:
            return "collision-free"
        pairs = ", ".join(f"{a}<->{b} ({d:.4f}m)" for a, b, d in self.violations[:4])
        return f"collision at sample {self.first_bad_sample}: {pairs}"


class CollisionChecker:
    """Whitelist-based collision checker with its own shadow MjData."""

    def __init__(self, model: mujoco.MjModel, robot: PandaRobot) -> None:
        self.model = model
        self.robot = robot
        self.data = mujoco.MjData(model)
        # Explicit whitelist of allowed contact pairs, by BODY name, per mode.
        # - grasp pairs: the fingers/palm pressing the cube IS the grasp -> only
        #   legal while actually holding it (holding_cube=True).
        # - rest pairs: the cube resting on table/floor is a normal physical
        #   contact -> only legal while NOT holding it.
        self.grasp_body_pairs: set[frozenset[str]] = {
            frozenset(("left_finger", "cube")),
            frozenset(("right_finger", "cube")),
            frozenset(("hand", "cube")),
            frozenset(("left_finger", "right_finger")),
        }
        self.rest_body_pairs: set[frozenset[str]] = {
            frozenset(("cube", "table")),
            frozenset(("cube", "table_legs")),
            frozenset(("cube", "world")),  # floor plane lives in the world body
        }
        # Extra user-supplied allowances (tests / special scenes).
        self.extra_allowed_pairs: set[frozenset[str]] = set()
        # Cached geom ids grouped by body id.
        self._body_of_geom = np.asarray(model.geom_bodyid).copy()

    # ------------------------------------------------------------------ #
    def set_allowed(self, pairs: list[tuple[str, str]]) -> None:
        """Extend the whitelist (used by tests to build ground-truth scenes)."""
        self.extra_allowed_pairs |= {frozenset(p) for p in pairs}

    # ------------------------------------------------------------------ #
    def pair_is_allowed(self, g1: int, g2: int, holding_cube: bool) -> bool:
        """Public predicate (used by parity tests and diagnostics)."""
        return self._pair_allowed(g1, g2, holding_cube)

    def _body_name(self, gid: int) -> str:
        bid = int(self._body_of_geom[gid])
        return str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid))

    def _pair_allowed(self, g1: int, g2: int, holding_cube: bool) -> bool:
        b1, b2 = self._body_name(g1), self._body_name(g2)
        pair = frozenset((b1, b2))
        if pair in self.extra_allowed_pairs:
            return True
        if pair in self.grasp_body_pairs:
            return holding_cube
        if pair in self.rest_body_pairs:
            return not holding_cube
        return False

    def _check_single(self, arm_q: np.ndarray, finger_opening: float, holding_cube: bool) -> list[tuple[str, str, float]]:
        """Run kinematics+collision for one configuration; return violations."""
        m, d = self.model, self.data
        d.qpos[self.robot.arm_qposadr] = arm_q
        d.qpos[self.robot.finger_qposadr] = finger_opening
        d.qvel[:] = 0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        violations: list[tuple[str, str, float]] = []
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if self._pair_allowed(g1, g2, holding_cube):
                continue
            violations.append((self._body_name(g1), self._body_name(g2), float(con.dist)))
        return violations

    def check_state(
        self,
        arm_q: np.ndarray,
        finger_opening: float = 0.04,
        holding_cube: bool = False,
        check_limits: bool = True,
    ) -> CollisionReport:
        """Validate one arm configuration (7, radians)."""
        arm_q = np.asarray(arm_q, dtype=float)
        if check_limits and not self.robot.within_limits(arm_q):
            return CollisionReport(False, 1, [("joint_limits", "joint_limits", 0.0)], 0)
        violations = self._check_single(arm_q, finger_opening, holding_cube)
        return CollisionReport(valid=not violations, n_samples=1, violations=violations, first_bad_sample=0 if violations else None)

    def check_segment(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        finger_opening: float = 0.04,
        holding_cube: bool = False,
        max_step: float = MAX_SEGMENT_STEP_RAD,
    ) -> CollisionReport:
        """Densely interpolate a joint-space segment and validate every sample."""
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        n = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / max_step)) + 1)
        for i, t in enumerate(np.linspace(0.0, 1.0, n)):
            q = (1 - t) * q0 + t * q1
            rep = self.check_state(q, finger_opening, holding_cube)
            if not rep.valid:
                return CollisionReport(False, i + 1, rep.violations, i)
        return CollisionReport(True, n)

    def check_path(
        self,
        path: list[np.ndarray] | np.ndarray,
        finger_opening: float = 0.04,
        holding_cube: bool = False,
        max_step: float = MAX_SEGMENT_STEP_RAD,
    ) -> CollisionReport:
        """Validate a full waypoint path (N,7) including all sub-segments."""
        arr = np.asarray(path, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.robot.n_arm:
            raise ValueError(f"path must be (N,7), got {arr.shape}")
        if arr.shape[0] == 0:
            return CollisionReport(True, 0)
        total = 0
        rep = self.check_state(arr[0], finger_opening, holding_cube)
        total += 1
        if not rep.valid:
            return CollisionReport(False, total, rep.violations, 0)
        for i in range(1, arr.shape[0]):
            rep = self.check_segment(arr[i - 1], arr[i], finger_opening, holding_cube, max_step)
            total += rep.n_samples
            if not rep.valid:
                return CollisionReport(False, total, rep.violations, i)
        return CollisionReport(True, total)
