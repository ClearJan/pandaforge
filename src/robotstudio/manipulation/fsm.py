"""Grasp-and-place finite state machine.

State chain (each state has entry conditions, an action, a success exit,
a timeout and a failure class - recoverable or fatal):

    IDLE -> PREGRASP -> APPROACH -> CLOSE -> VERIFY_GRASP -> LIFT
         -> TRANSFER -> PLACE -> RELEASE -> VERIFY_PLACE -> RETREAT
         -> SUCCESS

Terminal failure states: FAILED (fatal, with reason), TIMEOUT, ABORTED.

The FSM is a Controller: it runs INSIDE the engine thread (the single owner
of physics state) and drives nested TrajectoryExecutors for the motion
phases. All success judgements are physical:

* VERIFY_GRASP  - >=2 fingertip-cube contacts reported by MuJoCo.
* LIFT          - the cube's centre actually rises with the gripper.
* VERIFY_PLACE  - cube rests inside the target tolerance, gripper open,
                  cube nearly at rest; gripper released.

No object teleporting, no scripted animations: every motion phase streams
joint targets through the position servos, and the cube moves only through
contact forces.

Planning during the FSM blocks the engine thread for the (short) planning
duration - documented trade-off for state-ownership simplicity; planning
budgets are bounded (see GraspTaskConfig).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotstudio.collision.checker import CollisionChecker
from robotstudio.config import PandaSpec
from robotstudio.control.executor import TrajectoryExecutor
from robotstudio.kinematics.ik import IKSolver
from robotstudio.kinematics.pose import Pose, quat_to_mat
from robotstudio.manipulation.grasp import (
    GRASP_QUAT_WXYZ,
    grasp_pose,
    gripper_opening_for,
    place_pose,
    place_pregrasp_pose,
    pregrasp_pose,
)
from robotstudio.planning.base import PlannerConfig
from robotstudio.planning.rrt import RRTConnectPlanner
from robotstudio.trajectory.timing import TrajectoryGenerator
from robotstudio.utils.logging import get_logger

log = get_logger("fsm")


class GraspState(enum.Enum):
    IDLE = "IDLE"
    PREGRASP = "PREGRASP"
    APPROACH = "APPROACH"
    CLOSE = "CLOSE"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT = "LIFT"
    RETRY_DECISION = "RETRY_DECISION"  # internal: routes retry pull-out vs transfer
    TRANSFER = "TRANSFER"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    VERIFY_PLACE = "VERIFY_PLACE"
    RETREAT = "RETREAT"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    ABORTED = "ABORTED"


TERMINAL_OK = {GraspState.SUCCESS}
TERMINAL_FAIL = {GraspState.FAILED, GraspState.TIMEOUT, GraspState.ABORTED}


@dataclass
class GraspTaskConfig:
    cube_start: tuple[float, float, float]  # cube centre at task start, m
    cube_half: float = 0.025
    place_target: tuple[float, float, float] = (0.55, -0.3, 0.0)  # xy used; z = surface
    surface_z: float = 0.40  # table surface height (0.0 for empty scene)
    lift_height: float = 0.08  # m
    seed: int = 0
    plan_budget: float = 8.0  # s per planning call
    state_timeouts: dict = field(
        default_factory=lambda: {
            GraspState.PREGRASP: 30.0,
            GraspState.APPROACH: 20.0,
            GraspState.CLOSE: 4.0,
            GraspState.VERIFY_GRASP: 3.0,
            GraspState.LIFT: 15.0,
            GraspState.TRANSFER: 40.0,
            GraspState.PLACE: 20.0,
            GraspState.RELEASE: 4.0,
            GraspState.VERIFY_PLACE: 4.0,
            GraspState.RETREAT: 25.0,
        }
    )
    place_xy_tolerance: float = 0.035  # m
    max_grasp_retries: int = 1


@dataclass
class PhaseRecord:
    state: GraspState
    entered_at: float  # sim time
    exited_at: float | None = None
    outcome: str = ""  # "ok" / failure reason


class GraspFSM:
    """Drives a full pick-and-place task against the live engine."""

    def __init__(self, engine, spec: GraspTaskConfig) -> None:
        self.engine = engine
        self.spec = spec
        model = engine.model
        self.checker = CollisionChecker(model, engine.robot)
        self.ik = IKSolver(model, mujoco.MjData(model), engine.robot, seed=spec.seed)
        self.traj_gen = TrajectoryGenerator()
        self._planner = RRTConnectPlanner(
            self.checker,
            PlannerConfig(seed=spec.seed, max_planning_time=spec.plan_budget, shortcut_tries=150),
        )

        self.state = GraspState.IDLE
        self.history: list[PhaseRecord] = []
        self.error: str | None = None
        self._phase_entered_at = 0.0
        self._executor: TrajectoryExecutor | None = None
        self._pending_plan: str | None = None  # next state to plan for
        self._wait_until_t: float | None = None
        self._grasp_retries = 0
        self._cube_start = np.asarray(spec.cube_start, dtype=float)
        self._cube_dofadr = int(model.jnt_dofadr[model.joint("cube_joint").id])
        self._carry_offset: tuple[np.ndarray, np.ndarray] | None = None
        self._cube_z_at_grasp = float("nan")
        self._grasp_verified = False
        self._retrying = False
        self._planning_note = ""
        engine.on_estop(self.abort)

    # ------------------------------------------------------------------ #
    # public API (any thread)
    # ------------------------------------------------------------------ #
    @property
    def state_name(self) -> str:
        return self.state.value

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_OK | TERMINAL_FAIL

    @property
    def status_text(self) -> str:
        base = f"{self.state.value}"
        if self._planning_note:
            return f"{base} ({self._planning_note})"
        if self._executor is not None:
            return f"{base} [{self._executor.status_text}]"
        return base

    def start(self) -> None:
        """Begin the task (engine thread picks it up on the next update)."""
        if self.state not in (GraspState.IDLE,) and not self.is_terminal:
            raise RuntimeError(f"cannot start from state {self.state}")
        self.history.clear()
        self.error = None
        self._grasp_retries = 0
        self._carry_offset = None
        self.checker.cube_carry_offset = None
        self._transition(GraspState.PREGRASP)

    def abort(self) -> None:
        if self.is_terminal or self.state == GraspState.IDLE:
            return
        self._finish(GraspState.ABORTED, "aborted by user or e-stop")

    # ------------------------------------------------------------------ #
    # Controller protocol (engine thread only)
    # ------------------------------------------------------------------ #
    def update(self, ctx, dt: float) -> np.ndarray | None:
        if self.is_terminal or self.state == GraspState.IDLE:
            return None
        now = ctx.time
        if self._check_timeout(now):
            return None

        handler = getattr(self, f"_tick_{self.state.value.lower()}", None)
        if handler is None:
            self._fail(f"no handler for state {self.state}")
            return None
        return handler(ctx, now)

    def reference_joints(self, ctx):
        if self._executor is not None and self._executor.start_time is not None:
            return self._executor.reference_joints(ctx)
        return None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _transition(self, state: GraspState) -> None:
        now = self.engine.data.time
        if self.history and self.history[-1].exited_at is None:
            self.history[-1].exited_at = now
            if not self.history[-1].outcome:
                self.history[-1].outcome = "ok"
        self.history.append(PhaseRecord(state=state, entered_at=now))
        self.state = state
        self._phase_entered_at = now
        self._executor = None
        self._pending_plan = None
        self._wait_until_t = None
        self._planning_note = ""
        log.info("FSM -> %s (t=%.2f)", state.value, now)

    def _finish(self, state: GraspState, message: str) -> None:
        now = self.engine.data.time
        if self.history and self.history[-1].exited_at is None:
            self.history[-1].exited_at = now
            self.history[-1].outcome = message
        self.state = state
        self.error = message if state is not GraspState.SUCCESS else None
        self._executor = None
        self.checker.cube_carry_offset = None
        log.info("FSM terminal: %s (%s)", state.value, message)

    def _fail(self, reason: str) -> None:
        log.warning("FSM FAILED: %s", reason)
        self._finish(GraspState.FAILED, reason)

    def _timeout(self, reason: str) -> None:
        log.warning("FSM TIMEOUT: %s", reason)
        self._finish(GraspState.TIMEOUT, reason)

    def _check_timeout(self, now: float) -> bool:
        limit = self.spec.state_timeouts.get(self.state)
        if limit is not None and now - self._phase_entered_at > limit:
            self._timeout(f"state {self.state.value} exceeded {limit:.0f}s")
            return True
        return False

    def _plan_and_execute(self, ctx, target_q: np.ndarray, holding: bool, from_q: np.ndarray | None = None):
        """Plan a free (or carrying) path from the current config to target_q.

        Returns True when an executor is running; sets _executor. On planner
        failure, fails the task (planner already refuses unsafe output).
        """
        start = ctx.arm_q if from_q is None else from_q
        res = self._planner.plan(start, target_q, holding_cube=holding)
        if not res.success:
            self._fail(f"{self.state.value}: planning failed - {res.message}")
            return False
        try:
            traj = self.traj_gen.generate(res.path)
        except Exception as exc:
            self._fail(f"{self.state.value}: trajectory generation failed - {exc}")
            return False
        self._executor = TrajectoryExecutor(traj)
        return True

    def _ik_move(self, ctx, target_pose: Pose, holding: bool, finger_opening: float):
        """Plan a short move to an IK-solved configuration for target_pose."""
        res = self.ik.solve(
            target_pose,
            q_init=ctx.arm_q,
            n_restarts=6,
            validate=lambda q: self.checker.check_state(q, finger_opening, holding_cube=holding).valid,
        )
        if not res.success:
            self._fail(f"{self.state.value}: IK failed (pos_err={res.pos_err:.3f} m): {res.message}")
            return False
        rep = self.checker.check_segment(ctx.arm_q, res.arm_q, finger_opening, holding_cube=holding)
        if not rep.valid:
            self._fail(f"{self.state.value}: direct approach segment not collision-free: {rep.message}")
            return False
        try:
            traj = self.traj_gen.generate([ctx.arm_q, res.arm_q])
        except Exception as exc:
            self._fail(f"{self.state.value}: trajectory generation failed - {exc}")
            return False
        self._executor = TrajectoryExecutor(traj)
        return True

    def _run_executor(self, ctx, next_state: GraspState, holding: bool):
        """Step the active executor; transition when it completes."""
        if self._executor is None:
            self._fail(f"{self.state.value}: no executor")
            return None
        targets = self._executor.update(ctx, self.engine.model.opt.timestep)
        if self._executor.stats.finished:
            if self._executor.stats.success:
                self._transition(next_state)
            else:
                self._fail(f"{self.state.value}: execution failed")
        return targets

    def _cube_pos(self, ctx) -> np.ndarray:
        p = ctx.cube_pos
        if p is None:
            self._fail("scene has no cube")
            return np.zeros(3)
        return p

    # ------------------------------------------------------------------ #
    # state handlers
    # ------------------------------------------------------------------ #
    def _tick_pregrasp(self, ctx, now):
        """Plan+execute a collision-free move to the pregrasp hover."""
        cube = self._cube_pos(ctx)
        if self._pending_plan is None:
            self._pending_plan = "planning"
            self._planning_note = "planning"
            pose = pregrasp_pose(cube, self.spec.cube_half)
            res = self.ik.solve(
                pose,
                q_init=ctx.arm_q,
                n_restarts=8,
                validate=lambda q: self.checker.check_state(q, 0.04, holding_cube=False).valid,
            )
            if not res.success:
                self._fail(f"PREGRASP: no collision-free IK for pregrasp pose (pos_err={res.pos_err:.3f} m)")
                return None
            if not self._plan_and_execute(ctx, res.arm_q, holding=False):
                return None
            self._planning_note = ""
        return self._run_executor(ctx, GraspState.APPROACH, holding=False)

    def _tick_approach(self, ctx, now):
        """Vertical descent to the grasp pose (open fingers)."""
        cube = self._cube_pos(ctx)
        if self._pending_plan is None:
            self._pending_plan = "planning"
            self._planning_note = "IK descent"
            self.engine.set_gripper(PandaSpec.FINGER_OPEN)
            if not self._ik_move(ctx, grasp_pose(cube, self.spec.cube_half), holding=False, finger_opening=0.04):
                return None
            self._planning_note = ""
        return self._run_executor(ctx, GraspState.CLOSE, holding=False)

    def _tick_close(self, ctx, now):
        """Close the gripper onto the cube with a force-limited squeeze."""
        if self._wait_until_t is None:
            self._wait_until_t = now + 1.2  # squeeze settle time
            self.engine.set_gripper(gripper_opening_for(self.spec.cube_half))
        if now >= self._wait_until_t:
            self._wait_until_t = None
            self._transition(GraspState.VERIFY_GRASP)
        return ctx.arm_q  # hold position while closing

    def _tick_verify_grasp(self, ctx, now):
        """Physical evidence: fingertip-cube contacts must exist."""
        contacts = self.engine._count_grasp_contacts()
        if contacts >= 2:
            # capture hand->cube carry transform for transfer planning
            hand = self.engine.data.body(self.engine.robot.ee_body_id)
            R_hand = quat_to_mat(np.asarray(hand.xquat).copy())
            cube_quat = np.asarray(ctx.cube_quat).copy()
            R_cube = quat_to_mat(cube_quat)
            self._carry_offset = (np.asarray(hand.xpos).copy(), R_hand.T @ R_cube)
            # position offset needs hand translation too:
            cube_pos = self._cube_pos(ctx)
            self._carry_offset = (R_hand.T @ (cube_pos - np.asarray(hand.xpos)), R_hand.T @ R_cube)
            self.checker.cube_carry_offset = self._carry_offset
            self._cube_z_at_grasp = float(cube_pos[2])
            self._grasp_verified = True
            self._transition(GraspState.LIFT)
            return ctx.arm_q
        self._grasp_retries += 1
        if self._grasp_retries <= self.spec.max_grasp_retries:
            log.warning("no grasp contacts (%d); reopening, pulling out and re-approaching", contacts)
            self.engine.set_gripper(PandaSpec.FINGER_OPEN)
            self._retrying = True
            self._transition(GraspState.LIFT)  # pull-out validates with holding-mode whitelist
            return ctx.arm_q
        self._fail(f"VERIFY_GRASP: no stable fingertip contacts after retries (contacts={contacts})")
        return None

    def _tick_lift(self, ctx, now):
        """Rise vertically; the cube must actually come with the gripper."""
        if self._pending_plan is None:
            self._pending_plan = "planning"
            self._planning_note = "lift"
            cur = ctx.arm_q
            lift_q = self.ik.solve(
                Pose.make(ctx_ee(ctx) + [0, 0, self.spec.lift_height], GRASP_QUAT_WXYZ),
                q_init=cur,
                n_restarts=6,
                validate=lambda q: self.checker.check_state(q, 0.02, holding_cube=True).valid,
            )
            if not lift_q.success:
                self._fail(f"LIFT: IK failed for lift pose: {lift_q.message}")
                return None
            target = lift_q.arm_q
            if not self.checker.check_segment(cur, target, 0.02, holding_cube=True, allow_cube_rest=True).valid:
                self._fail("LIFT: vertical segment not collision-free while carrying")
                return None
            try:
                self._executor = TrajectoryExecutor(self.traj_gen.generate([cur, target]))
            except Exception as exc:
                self._fail(f"LIFT: trajectory generation failed - {exc}")
                return None
            self._planning_note = ""
        targets = self._run_executor(ctx, GraspState.RETRY_DECISION, holding=True)
        if self.state == GraspState.RETRY_DECISION:
            if self._retrying and not self._grasp_verified:
                # pull-out after a failed verify: go back for a clean attempt
                self._retrying = False
                self._transition(GraspState.PREGRASP)
            else:
                # physical check: cube must have risen with the gripper
                rise = float(self._cube_pos(ctx)[2] - self._cube_z_at_grasp)
                if rise < 0.5 * self.spec.lift_height:
                    self._fail(f"LIFT: cube did not follow gripper (rise={rise*1000:.0f} mm) - grasp lost")
                    return None
                self._transition(GraspState.TRANSFER)
        return targets

    def _tick_transfer(self, ctx, now):
        """Plan+execute the carrying move to the place hover."""
        if self._pending_plan is None:
            self._pending_plan = "planning"
            self._planning_note = "planning transfer"
            pose = place_pregrasp_pose(self.spec.place_target, self.spec.cube_half, self.spec.surface_z)
            res = self.ik.solve(
                pose,
                q_init=ctx.arm_q,
                n_restarts=8,
                validate=lambda q: self.checker.check_state(q, 0.02, holding_cube=True).valid,
            )
            if not res.success:
                self._fail(f"TRANSFER: no collision-free IK for place hover (pos_err={res.pos_err:.3f} m)")
                return None
            if not self._plan_and_execute(ctx, res.arm_q, holding=True):
                return None
            self._planning_note = ""
        return self._run_executor(ctx, GraspState.PLACE, holding=True)

    def _tick_place(self, ctx, now):
        """Descend to the release pose (cube bottom lands on the surface)."""
        if self._pending_plan is None:
            self._pending_plan = "planning"
            self._planning_note = "place descent"
            pose = place_pose(self.spec.place_target, self.spec.cube_half, self.spec.surface_z)
            # descend until the cube bottom rests on the surface
            res = self.ik.solve(
                pose,
                q_init=ctx.arm_q,
                n_restarts=6,
                validate=lambda q: self.checker.check_state(q, 0.02, holding_cube=True).valid,
            )
            if not res.success:
                self._fail(f"PLACE: IK failed for place pose (pos_err={res.pos_err:.3f} m)")
                return None
            rep = self.checker.check_segment(ctx.arm_q, res.arm_q, 0.02, holding_cube=True)
            if not rep.valid:
                self._fail(f"PLACE: descent segment not collision-free: {rep.message}")
                return None
            try:
                self._executor = TrajectoryExecutor(self.traj_gen.generate([ctx.arm_q, res.arm_q]))
            except Exception as exc:
                self._fail(f"PLACE: trajectory generation failed - {exc}")
                return None
            self._planning_note = ""
        return self._run_executor(ctx, GraspState.RELEASE, holding=True)

    def _tick_release(self, ctx, now):
        """Open the gripper and let the cube settle."""
        if self._wait_until_t is None:
            self._wait_until_t = now + 1.0
            self.engine.set_gripper(PandaSpec.FINGER_OPEN)
            self.checker.cube_carry_offset = None
            self._carry_offset = None
        if now >= self._wait_until_t:
            self._wait_until_t = None
            self._transition(GraspState.VERIFY_PLACE)
        return ctx.arm_q

    def _tick_verify_place(self, ctx, now):
        """Independent physical criteria for a completed placement."""
        cube = self._cube_pos(ctx)
        target = np.asarray(self.spec.place_target[:2])
        xy_err = float(np.linalg.norm(cube[:2] - target))
        surface = self.spec.surface_z
        z_ok = (surface - 0.01) <= cube[2] <= (surface + self.spec.cube_half + 0.03)
        opening = ctx.gripper_opening
        cube_vel = self.engine.data.qvel[self._cube_dofadr : self._cube_dofadr + 3]
        settled = bool(np.all(np.abs(cube_vel) < 0.05))
        if xy_err <= self.spec.place_xy_tolerance and z_ok and opening > 0.035 and settled:
            self._transition(GraspState.RETREAT)
            return ctx.arm_q
        if now - self._phase_entered_at > 2.5 and not z_ok:
            self._fail(f"VERIFY_PLACE: cube not placed (xy_err={xy_err:.3f} m, z={cube[2]:.3f})")
            return None
        return ctx.arm_q  # wait for settling within the timeout window

    def _tick_retreat(self, ctx, now):
        """Back off to a raised hover, then declare success.

        The direct vertical pull-out is tried first (fast); near table edges
        it can sweep the arm through the table, so the fallback is a proper
        RRT-Connect free path to the same hover pose. The just-released cube
        is modelled at its real resting pose, and grasp-pair grazing is
        tolerated (break-away), while arm-vs-table stays strictly checked.
        """
        if self._pending_plan is None:
            self._pending_plan = "planning"
            self._planning_note = "retreat"
            cur = ctx.arm_q
            up = self.ik.solve(
                Pose.make(ctx_ee(ctx) + [0, 0, self.spec.lift_height], GRASP_QUAT_WXYZ),
                q_init=cur,
                n_restarts=4,
                validate=lambda q: self.checker.check_state(q, 0.04, holding_cube=True, allow_cube_rest=True).valid,
            )
            if not up.success:
                self._fail(f"RETREAT: IK failed: {up.message}")
                return None
            if self.checker.check_segment(cur, up.arm_q, 0.04, holding_cube=True, allow_cube_rest=True).valid:
                waypoints = [cur, up.arm_q]
            else:
                # vertical pull-out unsafe (e.g. table edge) -> plan a free path
                cube = self._cube_pos(ctx)
                self.checker.set_shadow_cube_pose(cube, np.asarray(ctx.cube_quat))
                res = self._planner.plan(cur, up.arm_q, holding_cube=True)
                if not res.success:
                    self._fail(f"RETREAT: no collision-free retreat path - {res.message}")
                    return None
                waypoints = res.path
            try:
                self._executor = TrajectoryExecutor(self.traj_gen.generate(waypoints))
            except Exception as exc:
                self._fail(f"RETREAT: trajectory generation failed - {exc}")
                return None
            self._planning_note = ""
        targets = self._run_executor(ctx, GraspState.SUCCESS, holding=False)
        return targets


def ctx_ee(ctx) -> np.ndarray:
    """Current EE position from the engine context."""
    return ctx.engine.data.body(ctx.engine.robot.ee_body_id).xpos.copy()
