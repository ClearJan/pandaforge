"""Collision checking on MuJoCo's real collision geometry.

The checker owns a private ``MjData`` ("shadow") so it can run from any
thread while the engine keeps stepping. Every reported contact pair is
matched against an explicit, mode-dependent whitelist.

Modes:
* ``holding_cube=False`` (free paths): arm-vs-cube and arm-vs-environment are
  violations; the cube resting on ANY support is physics, not a fault
  (supports are arbitrary bodies and cannot be enumerated).
* ``holding_cube=True`` (transfer with the carried-object transform): the
  cube's pose follows the gripper, so cube-vs-environment means the carried
  object is sweeping into something - forbidden unless ``allow_cube_rest``
  (break-away); gripper-vs-cube is the grasp itself - allowed.

Moving obstacle: with ``moving_obstacle_fn`` set (sim time -> world pos),
every check validates the obstacle's SWEPT volume over
``[obstacle_time, obstacle_time + obstacle_swept_window]`` - planners and
the runtime guard therefore share one semantics (a validated path cannot be
immediately rejected by the guard, which was the source of replan thrash).

Path segments are checked with dense interpolation (default 0.02 rad).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotstudio.kinematics.pose import mat_to_quat
from robotstudio.robots.panda import PandaRobot
from robotstudio.utils.logging import get_logger

log = get_logger("collision")

MAX_SEGMENT_STEP_RAD = 0.02


@dataclass
class CollisionReport:
    valid: bool
    n_samples: int = 0
    violations: list[tuple[str, str, float]] = field(default_factory=list)
    first_bad_sample: int | None = None

    @property
    def message(self) -> str:
        if self.valid:
            return "collision-free"
        pairs = ", ".join(f"{a}<->{b} ({d:.4f}m)" for a, b, d in self.violations[:4])
        return f"collision at sample {self.first_bad_sample}: {pairs}"


class CollisionChecker:
    """Whitelist collision checker with its own shadow MjData."""

    def __init__(self, model: mujoco.MjModel, robot: PandaRobot) -> None:
        self.model = model
        self.robot = robot
        self.data = mujoco.MjData(model)
        # grasp pairs: the fingers/palm pressing the cube IS the grasp ->
        # only legal while holding it
        self.grasp_body_pairs: set[frozenset[str]] = {
            frozenset(("left_finger", "cube")),
            frozenset(("right_finger", "cube")),
            frozenset(("hand", "cube")),
            frozenset(("left_finger", "right_finger")),
        }
        # (rest pairs are handled semantically: cube-vs-environment is
        # allowed in free mode regardless of which body supports the object)
        self.rest_body_pairs: set[frozenset[str]] = set()
        # temporary relaxations (break-away during lift/retreat etc.)
        self.extra_allowed_pairs: set[frozenset[str]] = set()
        # carried-cube transform (grasp FSM): while set, every checked
        # configuration places the cube at (hand pose composed with offset)
        self.cube_carry_offset: tuple[np.ndarray, np.ndarray] | None = None
        # moving obstacle (scripted): sim time -> world position
        self.moving_obstacle_fn = None
        self.obstacle_time: float = 0.0
        self.obstacle_swept_window: float = 0.6
        # cached lookups
        self._body_of_geom = np.asarray(model.geom_bodyid).copy()
        self._cube_qposadr: int | None = None
        try:
            if int(model.joint("cube_joint").id) >= 0:
                self._cube_qposadr = int(model.jnt_qposadr[model.joint("cube_joint").id])
        except (KeyError, ValueError):
            self._cube_qposadr = None

    # ------------------------------------------------------------------ #
    # naming / whitelist
    # ------------------------------------------------------------------ #
    def _body_name(self, gid: int) -> str:
        bid = int(self._body_of_geom[gid])
        return str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid))

    def pair_is_allowed(self, g1: int, g2: int, holding_cube: bool, allow_cube_rest: bool = False) -> bool:
        return self._pair_allowed(g1, g2, holding_cube, allow_cube_rest)

    def _pair_allowed(self, g1: int, g2: int, holding_cube: bool, allow_cube_rest: bool = False) -> bool:
        b1, b2 = self._body_name(g1), self._body_name(g2)
        pair = frozenset((b1, b2))
        if pair in self.extra_allowed_pairs:
            return True
        gripper = {"left_finger", "right_finger", "hand"}
        is_grasp_pair = pair in self.grasp_body_pairs
        is_cube_env = "cube" in pair and not (set(pair) & gripper)
        if holding_cube:
            if is_grasp_pair:
                return True
            if is_cube_env:
                return allow_cube_rest
            return False
        if is_grasp_pair:
            return False  # a free path must not touch the object
        if is_cube_env:
            return True  # object resting on its support: not a planning fault
        return False

    # ------------------------------------------------------------------ #
    # single-state checking
    # ------------------------------------------------------------------ #
    def _check_once(self, arm_q, finger_opening, holding_cube, allow_cube_rest) -> list[tuple[str, str, float]]:
        """One kinematics+collision pass on the shadow data (obstacle fixed)."""
        m, d = self.model, self.data
        d.qpos[self.robot.arm_qposadr] = arm_q
        d.qpos[self.robot.finger_qposadr] = finger_opening
        d.qvel[:] = 0
        mujoco.mj_kinematics(m, d)
        if self.cube_carry_offset is not None and self._cube_qposadr is not None:
            pos_rel, R_rel = self.cube_carry_offset
            hand = d.body(self.robot.ee_body_id)
            R_hand = np.asarray(hand.xmat).reshape(3, 3)
            cube_R = R_hand @ R_rel
            cube_pos = np.asarray(hand.xpos) + R_hand @ pos_rel
            d.qpos[self._cube_qposadr : self._cube_qposadr + 3] = cube_pos
            d.qpos[self._cube_qposadr + 3 : self._cube_qposadr + 7] = mat_to_quat(cube_R)
            mujoco.mj_kinematics(m, d)  # refresh cube geoms
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        out: list[tuple[str, str, float]] = []
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if self._pair_allowed(g1, g2, holding_cube, allow_cube_rest):
                continue
            out.append((self._body_name(g1), self._body_name(g2), float(con.dist)))
        return out

    def _check_single(self, arm_q, finger_opening, holding_cube, allow_cube_rest=False):
        """SWEPT check when a moving obstacle is set, single pass otherwise."""
        if self.moving_obstacle_fn is None:
            return self._check_once(arm_q, finger_opening, holding_cube, allow_cube_rest)
        try:
            mid = int(self.model.body("moving_obstacle").mocapid[0])
        except (KeyError, ValueError):
            return self._check_once(arm_q, finger_opening, holding_cube, allow_cube_rest)
        for k in range(3):
            t = self.obstacle_time + self.obstacle_swept_window * k / 2.0
            self.data.mocap_pos[mid] = self.moving_obstacle_fn(t)
            violations = self._check_once(arm_q, finger_opening, holding_cube, allow_cube_rest)
            if violations:
                return violations
        return []

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def check_state_at(self, arm_q, t: float, finger_opening: float = 0.04, holding_cube: bool = False) -> CollisionReport:
        """Validate a configuration against the moving obstacle at time t."""
        self.obstacle_time = float(t)
        return self.check_state(arm_q, finger_opening, holding_cube)

    def pin_obstacle_time(self, t: float) -> None:
        """Pin the moving obstacle at time t for a subsequent planning run."""
        self.obstacle_time = float(t)

    def check_state(
        self,
        arm_q: np.ndarray,
        finger_opening: float = 0.04,
        holding_cube: bool = False,
        check_limits: bool = True,
        allow_cube_rest: bool = False,
    ) -> CollisionReport:
        """Validate one arm configuration (7, radians)."""
        arm_q = np.asarray(arm_q, dtype=float)
        if check_limits and not self.robot.within_limits(arm_q):
            return CollisionReport(False, 1, [("joint_limits", "joint_limits", 0.0)], 0)
        violations = self._check_single(arm_q, finger_opening, holding_cube, allow_cube_rest)
        return CollisionReport(
            valid=not violations, n_samples=1, violations=violations, first_bad_sample=0 if violations else None
        )

    def check_segment(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        finger_opening: float = 0.04,
        holding_cube: bool = False,
        max_step: float = MAX_SEGMENT_STEP_RAD,
        allow_cube_rest: bool = False,
    ) -> CollisionReport:
        """Densely interpolate a joint-space segment and validate every sample."""
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        n = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / max_step)) + 1)
        for i, t in enumerate(np.linspace(0.0, 1.0, n)):
            q = (1 - t) * q0 + t * q1
            rep = self.check_state(q, finger_opening, holding_cube, allow_cube_rest=allow_cube_rest)
            if not rep.valid:
                return CollisionReport(False, i + 1, rep.violations, i)
        return CollisionReport(True, n)

    def check_path(
        self,
        path,
        finger_opening: float = 0.04,
        holding_cube: bool = False,
        max_step: float = MAX_SEGMENT_STEP_RAD,
        allow_cube_rest: bool = False,
    ) -> CollisionReport:
        """Validate a full waypoint path (N,7) including all sub-segments."""
        arr = np.asarray(path, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.robot.n_arm:
            raise ValueError(f"path must be (N,7), got {arr.shape}")
        if arr.shape[0] == 0:
            return CollisionReport(True, 0)
        total = 0
        rep = self.check_state(arr[0], finger_opening, holding_cube, allow_cube_rest=allow_cube_rest)
        total += 1
        if not rep.valid:
            return CollisionReport(False, total, rep.violations, 0)
        for i in range(1, arr.shape[0]):
            rep = self.check_segment(arr[i - 1], arr[i], finger_opening, holding_cube, max_step, allow_cube_rest)
            total += rep.n_samples
            if not rep.valid:
                return CollisionReport(False, total, rep.violations, i)
        return CollisionReport(True, total)

    # ------------------------------------------------------------------ #
    # helpers used by the grasp FSM / benchmarks
    # ------------------------------------------------------------------ #
    def cube_contact_pairs(self, arm_q: np.ndarray, finger_opening: float) -> set[frozenset[str]]:
        """Body-name pairs currently contacting the shadow cube at ``arm_q``."""
        violations = self._check_once(arm_q, finger_opening, False, False)
        return {frozenset((a, b)) for a, b, _ in violations if "cube" in (a, b)}

    def set_shadow_cube_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        """Pin the shadow-data cube at a known world pose."""
        if self._cube_qposadr is None:
            return
        self.data.qpos[self._cube_qposadr : self._cube_qposadr + 3] = pos
        self.data.qpos[self._cube_qposadr + 3 : self._cube_qposadr + 7] = quat_wxyz
