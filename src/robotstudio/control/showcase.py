"""One-click showcase sequencer: a scripted demo controller.

Drives a full product story inside the live simulation:
  1. switch to the obstacle scene (resets to home),
  2. settle,
  3. plan + physically execute a collision-free sweep to the far-side target,
  4. switch to the tabletop scene,
  5. run the complete automatic pick & place,
  6. report SUCCESS / the failing stage.

Runs as a Controller inside the engine thread; every motion is still real
physics (the sequencer only chains the existing, verified components).
"""

from __future__ import annotations

import enum
import time

import mujoco
import numpy as np

from robotstudio.collision.checker import CollisionChecker
from robotstudio.config import PandaSpec
from robotstudio.control.executor import TrajectoryExecutor
from robotstudio.kinematics.ik import IKSolver
from robotstudio.kinematics.pose import Pose
from robotstudio.manipulation.fsm import GraspFSM, GraspState, GraspTaskConfig
from robotstudio.planning.base import PlannerConfig
from robotstudio.planning.rrt import RRTConnectPlanner
from robotstudio.simulation.scenes import preset_scenes
from robotstudio.trajectory.timing import TrajectoryGenerator
from robotstudio.utils.logging import get_logger

log = get_logger("showcase")


class Stage(enum.Enum):
    SWITCH_TO_OBSTACLE = "switch to obstacle scene"
    SETTLE_HOME = "settle at home"
    SWEEP_MOVE = "planned sweep around the obstacle"
    SWITCH_TO_TABLE = "switch to tabletop scene"
    PICK_AND_PLACE = "automatic pick & place"
    DONE = "done"


class ShowcaseSequencer:
    """Controller chaining the flagship demo; started by the UI toolbar."""

    #: keep running across the scene switches the sequencer itself submits
    keep_across_scene_switch = True

    def __init__(self, engine) -> None:
        self.engine = engine
        self.status_text = "showcase: starting"
        self.stage = Stage.SWITCH_TO_OBSTACLE
        self._estop_hit = False
        engine.on_estop(self._on_estop)
        self.error: str | None = None
        self._stage_t0 = time.perf_counter()
        self._executor: TrajectoryExecutor | None = None
        self._fsm: GraspFSM | None = None
        self._table_switched = False

    def _on_estop(self) -> None:
        """E-stop detaches controllers -> detect via callback, not update()."""
        self._estop_hit = True
        self.error = "e-stop"
        self.status_text = "showcase: aborted (e-stop)"
        self.stage = Stage.DONE

    # ------------------------------------------------------------------ #
    def update(self, ctx, dt: float) -> np.ndarray | None:
        if self.engine.estop or self._estop_hit:
            self.status_text = "showcase: aborted (e-stop)"
            self.error = "e-stop"
            return ctx.arm_q
        elapsed = time.perf_counter() - self._stage_t0
        if self.stage is Stage.SWITCH_TO_OBSTACLE:
            self.engine.apply_scene_spec(preset_scenes()["single_obs"])
            self._advance(Stage.SETTLE_HOME)
            return np.array(PandaSpec.HOME_QPOS)
        if self.stage is Stage.SETTLE_HOME:
            self.status_text = "showcase: settling at home"
            if elapsed > 1.5:
                self._plan_sweep(ctx)
            return np.array(PandaSpec.HOME_QPOS)
        if self.stage is Stage.SWEEP_MOVE:
            return self._tick_executor(ctx, Stage.SWITCH_TO_TABLE)
        if self.stage is Stage.SWITCH_TO_TABLE:
            if not self._table_switched:  # first entry: switch scene
                self.engine.apply_scene_spec(preset_scenes()["tabletop"])
                self._table_switched = True
                self._executor = None
                self._fsm = None
                return ctx.arm_q
            if elapsed > 1.2:
                self._start_pick_place(ctx)
            return np.array(PandaSpec.HOME_QPOS)
        if self.stage is Stage.PICK_AND_PLACE:
            assert self._fsm is not None
            targets = self._fsm.update(ctx, dt)
            self.status_text = f"showcase: {self._fsm.status_text}"
            if self._fsm.is_terminal:
                if self._fsm.state == GraspState.SUCCESS:
                    self.status_text = "showcase COMPLETE (sweep + pick & place)"
                    self.stage = Stage.DONE
                else:
                    self.error = f"pick & place failed: {self._fsm.state.value}: {self._fsm.error}"
                    self.status_text = f"showcase FAILED: {self.error}"
                    self.stage = Stage.DONE
            return targets
        return None  # DONE

    def reference_joints(self, ctx):
        if self._executor is not None:
            return self._executor.reference_joints(ctx)
        if self._fsm is not None:
            return self._fsm.reference_joints(ctx)
        return None

    # ------------------------------------------------------------------ #
    def _advance(self, stage: Stage) -> None:
        self.stage = stage
        self._stage_t0 = time.perf_counter()
        log.info("showcase stage -> %s", stage.value)

    def _plan_sweep(self, ctx) -> None:
        self.status_text = "showcase: planning sweep"
        checker = CollisionChecker(self.engine.model, self.engine.robot)
        ik = IKSolver(self.engine.model, mujoco.MjData(self.engine.model), self.engine.robot, seed=2028)
        res_ik = ik.solve(
            Pose.make([0.35, -0.40, 0.55], [0.0, 1.0, 0.0, 0.0]),
            q_init=ctx.arm_q,
            n_restarts=12,
            validate=lambda q: checker.check_state(q).valid,
        )
        if not res_ik.success:
            self._fail(f"showcase sweep IK failed: {res_ik.message}")
            return
        res = RRTConnectPlanner(
            checker, PlannerConfig(seed=2028, max_planning_time=10.0)
        ).plan(np.array(PandaSpec.HOME_QPOS), res_ik.arm_q)
        if not res.success:
            self._fail(f"showcase sweep planning failed: {res.message}")
            return
        try:
            traj = TrajectoryGenerator().generate(res.path)
        except Exception as exc:
            self._fail(f"showcase trajectory failed: {exc}")
            return
        self.engine.set_path_visual(res.path)
        self._executor = TrajectoryExecutor(traj)
        self._advance(Stage.SWEEP_MOVE)

    def _tick_executor(self, ctx, next_stage: Stage):
        if self._executor is None:
            self._fail("showcase: no executor")
            return None
        self.status_text = f"showcase: executing sweep [{self._executor.status_text}]"
        targets = self._executor.update(ctx, self.engine.model.opt.timestep)
        if self._executor.stats.finished:
            if self._executor.stats.success:
                self._executor = None
                self._advance(next_stage)
                return targets
            self._fail(f"showcase sweep execution: {self._executor.status_text}")
        return targets

    def _start_pick_place(self, ctx) -> None:
        snap_cube = ctx.cube_pos
        spec = GraspTaskConfig(
            cube_start=tuple(np.asarray(snap_cube).copy()),
            cube_half=0.025,
            place_target=(0.55, -0.30, 0.0),
            surface_z=0.40,
            seed=2028,
        )
        fsm = GraspFSM(self.engine, spec)
        self._fsm = fsm
        fsm.start()
        self._advance(Stage.PICK_AND_PLACE)

    def _fail(self, message: str) -> None:
        self.error = message
        self.status_text = f"showcase FAILED: {message}"
        self.stage = Stage.DONE
        log.warning("%s", message)
