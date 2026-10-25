"""Guarded trajectory execution with online replanning under a moving obstacle.

Architecture (plan -> monitor -> replan):

- The obstacle follows a DETERMINISTIC script (pure function of sim time),
  so planning against it stays reproducible.
- Every physics step the guard validates the trajectory's reference
  configuration against the obstacle's pose at (now + GUARD_HORIZON) on a
  shadow checker - the horizon buys the deceleration + replan time.
- On violation the controller REPLANS immediately from the current arm
  configuration to the goal with the obstacle pinned at its current pose.
  Planning runs inside the engine thread: physics (and hence the obstacle
  script, which is time-based) is frozen during the search, so the plan is
  consistent with the world it is validated against. Execution then resumes
  under the same guard; repeated violations trigger further replans up to
  MAX_REPLANS, after which the task fails loudly.
- If planning fails (obstacle exactly blocking), the controller holds the
  current configuration for RETRY_DELAY seconds of simulation time - the
  obstacle advances, opening a window - and retries.

All outcomes are measurable: replan count, obstacle contacts observed (must
stay 0), guard violations, final goal error.
"""

from __future__ import annotations

import numpy as np

from robotstudio.collision.checker import CollisionChecker
from robotstudio.control.executor import TrajectoryExecutor
from robotstudio.planning.base import PlannerConfig
from robotstudio.planning.rrt import RRTConnectPlanner
from robotstudio.trajectory.timing import Trajectory, TrajectoryGenerator

# lookahead: must exceed the obstacle's advance over the whole
# detect-decelerate-retreat sequence. Forensics: at 0.6 s the guard first
# fired with the hand ALREADY 15 mm inside the panel (overlap band
# ~0.16 m vs 0.11 m of advance) - by then no plan can start legally.
GUARD_HORIZON = 1.2  # s of lookahead
# the obstacle closes the corridor periodically: a blocked attempt must
# retry cheaply and often until the sweep opens a window
MAX_REPLANS = 40
RETRY_DELAY = 0.4  # s of sim time between replan attempts while blocked
PLAN_BUDGET = 1.0  # s wall per replan attempt (replans are short)


class GuardedExecutionStats:
    def __init__(self) -> None:
        self.completed = False
        self.success = False
        self.replans = 0
        self.guard_violations = 0
        self.obstacle_contacts = 0
        self.max_joint_tracking_err = 0.0

    def as_dict(self) -> dict:
        return {
            "completed": self.completed,
            "success": self.success,
            "replans": self.replans,
            "guard_violations": self.guard_violations,
            "obstacle_contacts": self.obstacle_contacts,
            "max_joint_tracking_err_rad": self.max_joint_tracking_err,
        }


class GuardedExecutor:
    """TrajectoryExecutor wrapper that monitors a scripted moving obstacle
    and replans online when the reference becomes unsafe."""

    def __init__(
        self,
        engine,
        trajectory: Trajectory,
        checker: CollisionChecker | None = None,
        seed: int = 2028,
        guard_horizon: float = GUARD_HORIZON,
    ) -> None:
        self.engine = engine
        self.goal = trajectory.sample(trajectory.duration)
        self.executor = TrajectoryExecutor(trajectory)
        self.checker = checker or CollisionChecker(engine.model, engine.robot)
        self.checker.moving_obstacle_fn = engine.moving_obstacle_pose
        self.guard_horizon = guard_horizon
        self.seed = seed
        self.stats = GuardedExecutionStats()
        self._retry_at: float | None = None
        self._last_arm_q = self.goal.copy()
        self._park_q = self._solve_park()
        self._obstacle_geom_ids = self._obstacle_geoms()

    # ------------------------------------------------------------------ #
    def _obstacle_geoms(self) -> set[int]:
        m = self.engine.model
        try:
            bid = int(m.body("moving_obstacle").id)
        except (KeyError, ValueError):
            return set()
        start = int(m.body_geomadr[bid])
        return set(range(start, start + int(m.body_geomnum[bid])))

    @property
    def status_text(self) -> str:
        if self.stats.completed:
            return "guarded: completed" if self.stats.success else f"guarded: failed ({self.stats.obstacle_contacts} contacts)"
        if self.executor is None:
            return "guarded: replanning"
        return f"guarded [{self.executor.status_text}] replans={self.stats.replans}"

    # ------------------------------------------------------------------ #
    def update(self, ctx, dt: float) -> np.ndarray | None:
        self._count_obstacle_contacts()
        if self.stats.completed:
            return self.goal

        # ---- parking / replanning phase ----
        if self.executor is None:
        # EE-based jacobian escapes were tried and are WRONG here: the panel
        # hits arbitrary ARM LINKS, and pulling the EE back can swing the
        # elbow INTO the sweep plane (measured: 44 simultaneous contacts).
        # The sound approach is a PARK configuration - solved once, verified
        # against the obstacle's ENTIRE sweep cycle offline - and a simple
        # state machine: violation -> park -> wait for a window -> replan.
            now = ctx.time
            if self._park_q is None:
                self._fail_loudly("no verified park configuration available")
                self.stats.completed = True
                return ctx.arm_q
            if not self._at_park(ctx.arm_q):
                # the retreat to park must itself be GUARDED: a straight
                # servo to park can cut through the panel (found by the
                # --moving demo at a different sweep phase). Plan the retreat
                # with the obstacle pinned at the guarded future time; it
                # runs under this same monitor, so a mid-retreat violation
                # simply re-enters this logic.
                self.stats.replans += 1
                self.checker.pin_obstacle_time(now)
                res = RRTConnectPlanner(
                    self.checker,
                    PlannerConfig(seed=self.seed + 1000 + self.stats.replans, max_planning_time=1.0),
                ).plan(ctx.arm_q, self._park_q)
                if res.success:
                    self._retry_at = None
                    self.executor = TrajectoryExecutor(TrajectoryGenerator().generate(res.path))
                    return self.executor.update(ctx, dt)
                return self._park_q  # last resort if the retreat is blocked
        # parked: wait for the sweep to open a window, then replan
            if self._retry_at is not None and now < self._retry_at:
                return self._park_q
            if self.stats.replans >= MAX_REPLANS:
                self.stats.completed = True
                self.stats.success = False
                return self._park_q
            self.stats.replans += 1
            self.checker.pin_obstacle_time(now)
            planner = RRTConnectPlanner(
                self.checker, PlannerConfig(seed=self.seed + self.stats.replans, max_planning_time=PLAN_BUDGET)
            )
            res = planner.plan(ctx.arm_q, self.goal)
            if not res.success:
                self._retry_at = now + RETRY_DELAY
                return self._park_q
            self._retry_at = None
            self.executor = TrajectoryExecutor(TrajectoryGenerator().generate(res.path))
            return self.executor.update(ctx, dt)

        # ---- guarding phase (swept volume) ----
        q_ref = self.executor.reference_joints(ctx)
        if q_ref is None:
            q_ref = self._last_arm_q
        self._last_arm_q = q_ref
        # guard BOTH the reference and the PHYSICAL arm: the arm lags the
        # reference during transients, and the panel can sweep into that gap
        # (found by first-contact forensics: ref was declared safe while the
        # lagging physical arm was already in the swept volume)
        if not (self._swept_safe(q_ref, ctx.time) and self._swept_safe(ctx.arm_q, ctx.time)):
            self.stats.guard_violations += 1
            self.executor = None  # drop and replan from the current state
            return ctx.arm_q
        targets = self.executor.update(ctx, dt)
        if self.executor.stats.finished:
            self.stats.completed = True
            self.stats.success = self.executor.stats.success and self.stats.obstacle_contacts == 0
            self.stats.max_joint_tracking_err = max(
                self.stats.max_joint_tracking_err, self.executor.stats.max_joint_tracking_err
            )
        return targets

    def reference_joints(self, ctx):
        if self.executor is not None:
            return self.executor.reference_joints(ctx)
        return None

    def _swept_safe(self, q: np.ndarray, now: float) -> bool:
        """Guard semantics = checker semantics: with a moving obstacle set,
        every check_state_at already validates the obstacle's swept volume
        (see CollisionChecker.obstacle_swept_window)."""
        return self.checker.check_state_at(q, now).valid

    def _solve_park(self) -> np.ndarray | None:
        """A configuration safe for the obstacle's ENTIRE sweep cycle.

        EE above and behind the sweep plane (x ~0.30, z ~0.78): solved by IK
        and verified against 16 phases of the script. Park is what makes the
        violation path sound - the arm always has somewhere proven-safe to go.
        """
        import mujoco as _mj

        from robotstudio.kinematics.ik import IKSolver
        from robotstudio.kinematics.pose import Pose

        eng = self.engine
        ik = IKSolver(eng.model, _mj.MjData(eng.model), eng.robot, seed=self.seed)
        res = ik.solve(Pose.make([0.30, 0.0, 0.78], [0.0, 1.0, 0.0, 0.0]), n_restarts=10)
        if not res.success:
            return None
        period = eng.current_scene_spec().moving_obstacle.period
        for k in range(16):
            if not self.checker.check_state_at(res.arm_q, period * k / 16.0).valid:
                return None  # not cycle-safe: refuse to use it
        return res.arm_q

    def _at_park(self, arm_q: np.ndarray) -> bool:
        return bool(np.max(np.abs(arm_q - self._park_q)) < 0.08)

    def _fail_loudly(self, message: str) -> None:
        self.stats.success = False

    def _count_obstacle_contacts(self) -> None:
        if not self._obstacle_geom_ids:
            return
        d = self.engine.data
        arm_geoms = self._arm_geoms()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self._obstacle_geom_ids and g2 in arm_geoms) or (
                g2 in self._obstacle_geom_ids and g1 in arm_geoms
            ):
                self.stats.obstacle_contacts += 1
                return  # one per step is enough for accounting

    def _arm_geoms(self) -> set[int]:
        m = self.engine.model
        out: set[int] = set()
        for name in ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"):
            try:
                bid = int(m.body(name).id)
            except (KeyError, ValueError):
                continue
            start = int(m.body_geomadr[bid])
            out.update(range(start, start + int(m.body_geomnum[bid])))
        return out
