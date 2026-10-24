"""Physics simulation engine.

Architecture
------------
The engine is the SINGLE OWNER of MuJoCo model/data state. All interaction
from other threads (UI, planners, grasp FSM) goes through:

* command submission (``submit``) for actions that mutate physics, and
* ``snapshot()`` for consistent state reads (copies under a short lock).

The engine thread runs at ``SimDefaults.REALTIME_HZ``: it drains the command
queue, updates the active :class:`~robotstudio.control.base.Controller` at
every physics step, calls ``mj_step`` for the requested realtime factor and
syncs the viewer.

Scene switching does not reload the model: all preset obstacles, the table,
the grasp cube and the place marker exist in every generated scene file;
``apply_scene_spec`` toggles them in-place (static body poses / geom
``contype``/free-joint ``qpos``), so the viewer window stays alive.

Units: meters, radians, seconds. Gripper ctrl in actuator units 0..255.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import mujoco
import numpy as np

from robotstudio.config import PandaSpec, SimDefaults
from robotstudio.robots.panda import PandaRobot
from robotstudio.simulation.scenes import (
    SceneSpec,
    preset_scenes,
    write_scene_file,
)
from robotstudio.simulation.viewer import ViewerHandle
from robotstudio.utils.logging import get_logger

log = get_logger("engine")


# --------------------------------------------------------------------------- #
# controllers
# --------------------------------------------------------------------------- #


class Controller(Protocol):
    """A controller produces arm joint targets (7,) every physics step."""

    def update(self, ctx: "EngineContext", dt: float) -> np.ndarray | None:
        """Return arm ctrl targets, or None to keep the last targets."""
        ...

    def on_finish(self, ctx: "EngineContext") -> None: ...


@dataclass
class EngineContext:
    """Read/write facade handed to controllers each step (engine thread only)."""

    engine: "SimulationEngine"

    @property
    def time(self) -> float:
        return self.engine.data.time

    @property
    def arm_q(self) -> np.ndarray:
        return self.engine.robot.arm_q_from_qpos(self.engine.data.qpos)

    @property
    def arm_qvel(self) -> np.ndarray:
        return self.engine.data.qvel[self.engine.robot.arm_dofadr].copy()

    @property
    def gripper_opening(self) -> float:
        """Current finger separation midpoint (m per finger)."""
        d = self.engine.data
        return float(np.mean(d.qpos[self.engine.robot.finger_qposadr]))

    @property
    def cube_pos(self) -> np.ndarray | None:
        eng = self.engine
        if "cube" not in eng._cube_body_name_map:
            return None
        bid = eng._cube_body_name_map["cube"]
        return eng.data.body(bid).xpos.copy()

    @property
    def cube_quat(self) -> np.ndarray | None:
        eng = self.engine
        if "cube" not in eng._cube_body_name_map:
            return None
        bid = eng._cube_body_name_map["cube"]
        return eng.data.body(bid).xquat.copy()


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


@dataclass
class Command:
    pass


@dataclass
class SetJointsCmd(Command):
    arm_q: np.ndarray
    gripper_opening: float | None = None  # meters per finger


@dataclass
class SetGripperCmd(Command):
    opening: float  # meters per finger


@dataclass
class SetControllerCmd(Command):
    controller: Controller | None  # None detaches


@dataclass
class PauseCmd(Command):
    paused: bool


@dataclass
class EStopCmd(Command):
    active: bool = True


@dataclass
class ResetCmd(Command):
    pass


@dataclass
class ApplySceneCmd(Command):
    spec: SceneSpec


@dataclass
class SetSpeedCmd(Command):
    realtime_factor: float


# --------------------------------------------------------------------------- #
# snapshot for other threads
# --------------------------------------------------------------------------- #


@dataclass
class SimSnapshot:
    sim_time: float
    wall_fps: float
    realtime_factor: float
    paused: bool
    estop: bool
    qpos: np.ndarray  # (9,)
    qvel: np.ndarray  # (9,)
    ctrl: np.ndarray  # (8,)
    ee_pos: np.ndarray  # (3,)
    ee_quat: np.ndarray  # (4,) wxyz
    joint_targets: np.ndarray  # (7,) current ctrl targets
    joint_tracking_err: np.ndarray  # (7,) rad
    ee_tracking_pos_err: float  # m (vs trajectory reference, 0 when idle)
    ee_tracking_rot_err: float  # rad
    ncon: int
    scene_name: str
    controller_status: str
    cube_pos: np.ndarray | None
    cube_quat: np.ndarray | None
    grasp_contacts: int
    history: dict[str, np.ndarray] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #


class SimulationEngine:
    """Single owner of the MuJoCo physics state (engine thread)."""

    def __init__(self, scene_name: str = "tabletop", headless: bool = False) -> None:
        self._lock = threading.RLock()
        self._cmd_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._headless = headless

        self.scene_name = scene_name
        self._scene_spec = preset_scenes()[scene_name]
        self._load_model(self._scene_spec)

        self.paused = False
        self.estop = False
        self.realtime_factor = 1.0
        self._controller: Controller | None = None
        self._controller_status = "idle"
        self._joint_targets = np.array(PandaSpec.HOME_QPOS)
        self._gripper_target = PandaSpec.GRIPPER_CTRL_OPEN
        self._history: dict[str, list] = {k: [] for k in ("t", "ee_pos_err", "ee_rot_err", "jtrack", "arm_q")}
        self._history_cap = 4000
        self._on_estop: list[Callable[[], None]] = []
        self._on_error: list[Callable[[str], None]] = []
        self._last_error: str | None = None
        self._last_wall = time.perf_counter()
        self._fps_ema = 0.0

    # ------------------------------------------------------------------ #
    # model / scene management
    # ------------------------------------------------------------------ #
    def _load_model(self, spec: SceneSpec) -> None:
        path = write_scene_file(spec)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._fk_data = mujoco.MjData(self.model)
        self.robot = PandaRobot(self.model)
        self._stash_original_collision_flags()
        self._cube_body_name_map = {"cube": self.model.body("cube").id}
        self._apply_scene_to_model(spec)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # home keyframe
        mujoco.mj_forward(self.model, self.data)
        self._scene_spec = spec
        self.scene_name = spec.name
        log.info("scene %r loaded (nq=%d nu=%d nbody=%d)", spec.name, self.model.nq, self.model.nu, self.model.nbody)

    def _apply_scene_to_model(self, spec: SceneSpec) -> None:
        """Toggle table/obstacles in-place so the viewer never reloads."""
        m, d = self.model, self.data
        HIDDEN_Z = -5.0

        # --- table ---
        table_body = m.body("table")
        if spec.table.enabled:
            top_z = spec.table.top_height
            plate_z = top_z - spec.table.thickness / 2.0
            table_body.pos[:] = [spec.table.center[0], spec.table.center[1], plate_z]
            self._set_geom_enabled_by_body(table_body.id, True)
            legs = m.body("table_legs")
            legs.pos[:] = [spec.table.center[0], spec.table.center[1], (top_z - spec.table.thickness) / 2.0]
            self._set_geom_enabled_by_body(legs.id, True)
        else:
            table_body.pos[:] = [0, 0, HIDDEN_Z]
            self._set_geom_enabled_by_body(table_body.id, False)
            legs = m.body("table_legs")
            legs.pos[:] = [0, 0, HIDDEN_Z]
            self._set_geom_enabled_by_body(legs.id, False)

        # --- obstacles: names present in the model but missing from spec are hidden ---
        spec_obs = {o.name: o for o in spec.obstacles if o.enabled}
        for body_id in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name is None or not name.startswith(("wall_", "shelf_", "post_", "obs_")):
                continue
            if name in spec_obs:
                o = spec_obs[name]
                m.body(name).pos[:] = o.pos
                m.body(name).quat[:] = o.quat
                self._set_geom_enabled_by_body(body_id, True)
            else:
                m.body(name).pos[:] = [0, 0, HIDDEN_Z]
                self._set_geom_enabled_by_body(body_id, False)

        # --- cube ---
        cube_bid = m.body("cube").id
        cube_jadr = m.jnt_qposadr[m.joint("cube_joint").id]
        cube_vadr = m.jnt_dofadr[m.joint("cube_joint").id]
        if spec.cube_enabled:
            x, y, _ = spec.cube_pos
            surf = spec.table.top_height if spec.table.enabled else 0.0
            z = surf + spec.cube_size
            d.qpos[cube_jadr : cube_jadr + 3] = [x, y, z]
            d.qpos[cube_jadr + 3 : cube_jadr + 7] = [1, 0, 0, 0]
            d.qvel[cube_vadr : cube_vadr + 6] = 0
            self._set_geom_enabled_by_body(cube_bid, True)
        else:
            d.qpos[cube_jadr : cube_jadr + 3] = [0, 0, HIDDEN_Z]
            self._set_geom_enabled_by_body(cube_bid, False)

        # --- place target (mocap) ---
        try:
            mocap_id = m.body("place_target").mocapid[0]
            if spec.place_enabled and mocap_id >= 0:
                x, y, _ = spec.place_pos
                surf = spec.table.top_height if spec.table.enabled else 0.0
                d.mocap_pos[mocap_id] = [x, y, surf]
            elif mocap_id >= 0:
                d.mocap_pos[mocap_id] = [0, 0, HIDDEN_Z]
        except (KeyError, IndexError):
            pass

        mujoco.mj_forward(m, d)

    def _set_geom_enabled_by_body(self, body_id: int, enabled: bool) -> None:
        m = self.model
        start = int(m.body_geomadr[body_id])
        count = int(m.body_geomnum[body_id])
        for gid in range(start, start + count):
            if gid < 0:
                continue
            if enabled:
                m.geom_contype[gid] = self._original_contype[gid]
                m.geom_conaffinity[gid] = self._original_conaffinity[gid]
            else:
                m.geom_contype[gid] = 0
                m.geom_conaffinity[gid] = 0

    def _stash_original_collision_flags(self) -> None:
        m = self.model
        self._original_contype = m.geom_contype.copy()
        self._original_conaffinity = m.geom_conaffinity.copy()

    def apply_scene_spec(self, spec: SceneSpec) -> None:
        """Submit a scene change; the viewer window stays open."""
        self.submit(ApplySceneCmd(spec=spec))

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self, with_viewer: bool = True) -> None:
        if self._running:
            return
        self._running = True
        self._with_viewer = with_viewer and not self._headless
        self._thread = threading.Thread(target=self._run, name="sim-engine", daemon=False)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        viewer = ViewerHandle(self.model, self.data, show=self._with_viewer)
        self._viewer_ref = viewer
        if viewer.show:
            viewer.set_camera(lookat=(0.4, 0.0, 0.35), distance=2.2, azimuth=135, elevation=-25)
        frame_budget = 1.0 / SimDefaults.REALTIME_HZ
        next_frame = time.perf_counter()
        while self._running:
            t0 = time.perf_counter()
            try:
                self._process_commands()
                if not self.paused and not self.estop:
                    self._step_frame()
                elif self.estop:
                    self._estop_hold()
                viewer.sync()
            except Exception as exc:  # keep the loop alive, but surface the error
                log.exception("engine loop error")
                self._report_error(f"{type(exc).__name__}: {exc}")
            # frame pacing
            next_frame += frame_budget
            now = time.perf_counter()
            sleep_for = next_frame - now
            if sleep_for < -0.05:  # fell behind: resync
                next_frame = now + frame_budget
            elif sleep_for > 0:
                time.sleep(sleep_for)
            dt_wall = time.perf_counter() - t0
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * (1.0 / max(dt_wall, 1e-6))
        viewer.close()

    def _step_frame(self) -> None:
        m, d = self.model, self.data
        steps = max(1, int(round(self.realtime_factor * (1.0 / SimDefaults.REALTIME_HZ) / m.opt.timestep)))
        steps = min(steps, int(SimDefaults.MAX_REALTIME_FACTOR * (1.0 / SimDefaults.REALTIME_HZ) / m.opt.timestep))
        ctx = EngineContext(engine=self)
        with self._lock:
            for _ in range(steps):
                if self._controller is not None:
                    targets = self._controller.update(ctx, m.opt.timestep)
                    if targets is not None:
                        self._joint_targets = np.clip(targets, self.robot.jnt_range[:, 0], self.robot.jnt_range[:, 1])
                d.ctrl[self.robot.arm_ctrladr] = self._joint_targets
                d.ctrl[self.robot.gripper_ctrladr] = self._gripper_target
                mujoco.mj_step(m, d)
            self._record_history(steps)

    def _estop_hold(self) -> None:
        """Emergency stop: freeze targets at the last commanded values."""
        with self._lock:
            self._controller = None
            self._controller_status = "ESTOP (targets held)"

    def _record_history(self, steps: int) -> None:
        h = self._history
        h["t"].append(float(self.data.time))
        h["arm_q"].append(self.robot.arm_q_from_qpos(self.data.qpos).copy())
        h["jtrack"].append(np.abs(self.data.ctrl[:7] - self.data.qpos[self.robot.arm_qposadr]))
        ee_pos = self.data.body(self.robot.ee_body_id).xpos.copy()
        ref_pos = self._reference_ee_pos()
        h["ee_pos_err"].append(float(np.linalg.norm(ee_pos - ref_pos)) if ref_pos is not None else 0.0)
        h["ee_rot_err"].append(0.0)
        for k in h:
            if len(h[k]) > self._history_cap:
                del h[k][0 : len(h[k]) - self._history_cap]

    def _reference_ee_pos(self) -> np.ndarray | None:
        """FK of the controller's reference joints (50 Hz, cheap kinematics-only)."""
        if self._controller is None or not hasattr(self._controller, "reference_joints"):
            return None
        ctx = EngineContext(engine=self)
        q_ref = self._controller.reference_joints(ctx)
        if q_ref is None:
            return None
        fk = self._fk_data
        fk.qpos[self.robot.arm_qposadr] = q_ref
        fk.qpos[self.robot.finger_qposadr] = 0.04
        mujoco.mj_kinematics(self.model, fk)
        return fk.body(self.robot.ee_body_id).xpos.copy()

    # ------------------------------------------------------------------ #
    # command processing
    # ------------------------------------------------------------------ #
    def submit(self, cmd: Command) -> None:
        self._cmd_queue.put(cmd)

    def _process_commands(self) -> None:
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                self._apply_command(cmd)

    def _apply_command(self, cmd: Command) -> None:
        if isinstance(cmd, SetJointsCmd):
            if not self.robot.within_limits(cmd.arm_q):
                bad = [
                    (i, float(q), tuple(r))
                    for i, (q, r) in enumerate(zip(cmd.arm_q, self.robot.jnt_range))
                    if not (r[0] <= q <= r[1])
                ]
                raise ValueError(f"joint targets outside limits: {bad}")
            self._joint_targets = np.asarray(cmd.arm_q, dtype=float).copy()
            if cmd.gripper_opening is not None:
                self._gripper_target = self.robot.gripper_ctrl_from_opening(cmd.gripper_opening)
            self._controller = None
            self._controller_status = "manual joint control"
        elif isinstance(cmd, SetGripperCmd):
            self._gripper_target = self.robot.gripper_ctrl_from_opening(cmd.opening)
        elif isinstance(cmd, SetControllerCmd):
            self._controller = cmd.controller
            self._controller_status = "controller active" if cmd.controller else "idle"
            self.estop = False
        elif isinstance(cmd, PauseCmd):
            self.paused = cmd.paused
            self._controller_status = ("paused" if cmd.paused else "running")
        elif isinstance(cmd, EStopCmd):
            self.estop = cmd.active
            if cmd.active:
                for cb in self._on_estop:
                    try:
                        cb()
                    except Exception:
                        log.exception("estop callback failed")
                self._controller_status = "ESTOP (targets held)"
            else:
                self._controller_status = "idle"
        elif isinstance(cmd, ResetCmd):
            self._reset_to_home()
        elif isinstance(cmd, ApplySceneCmd):
            self._controller = None
            self._controller_status = "idle"
            self._scene_spec = cmd.spec
            self.scene_name = cmd.spec.name
            self._apply_scene_to_model(cmd.spec)
            self._reset_to_home()
            if getattr(self, "_viewer_ref", None) is not None:
                self._viewer_ref.sync()
        elif isinstance(cmd, SetSpeedCmd):
            self.realtime_factor = float(np.clip(cmd.realtime_factor, 0.05, SimDefaults.MAX_REALTIME_FACTOR))

    def _reset_to_home(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._joint_targets = np.array(PandaSpec.HOME_QPOS)
        self._gripper_target = PandaSpec.GRIPPER_CTRL_OPEN
        self._controller = None
        self._controller_status = "home"
        self._history = {k: [] for k in self._history}

    # ------------------------------------------------------------------ #
    # external API (any thread)
    # ------------------------------------------------------------------ #
    def set_joints(self, arm_q, gripper_opening: float | None = None) -> None:
        arm_q = np.asarray(arm_q, dtype=float)
        if arm_q.shape != (self.robot.n_arm,):
            raise ValueError(f"arm_q must have shape ({self.robot.n_arm},), got {arm_q.shape}")
        if not self.robot.within_limits(arm_q):
            bad = [
                (i + 1, float(q), (float(lo), float(hi)))
                for i, (q, (lo, hi)) in enumerate(zip(arm_q, self.robot.jnt_range))
                if not (lo <= q <= hi)
            ]
            raise ValueError(f"joint targets outside limits (units: radians): {bad}")
        self.submit(SetJointsCmd(arm_q=arm_q, gripper_opening=gripper_opening))

    def set_gripper(self, opening: float) -> None:
        self.submit(SetGripperCmd(opening=float(opening)))

    def set_controller(self, controller: Controller | None) -> None:
        self.submit(SetControllerCmd(controller=controller))

    def set_paused(self, paused: bool) -> None:
        self.submit(PauseCmd(paused=paused))

    def set_estop(self, active: bool = True) -> None:
        self.submit(EStopCmd(active=active))

    def reset(self) -> None:
        self.submit(ResetCmd())

    def set_speed(self, realtime_factor: float) -> None:
        self.submit(SetSpeedCmd(realtime_factor=realtime_factor))

    def on_estop(self, callback: Callable[[], None]) -> None:
        self._on_estop.append(callback)

    def on_error(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked (from the engine thread) on runtime errors."""
        self._on_error.append(callback)

    def _report_error(self, message: str) -> None:
        self._last_error = message
        for cb in self._on_error:
            try:
                cb(message)
            except Exception:
                log.exception("error callback failed")

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def current_scene_spec(self) -> SceneSpec:
        with self._lock:
            return self._scene_spec

    def state_copy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Atomic (qpos, qvel, ctrl) copy for planning snapshots."""
        with self._lock:
            return self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy()

    def snapshot(self) -> SimSnapshot:
        with self._lock:
            d = self.data
            # live controller status (changes every step during execution)
            if self._controller is not None and hasattr(self._controller, "status_text"):
                self._controller_status = str(self._controller.status_text)
            qpos = d.qpos.copy()
            ctrl = d.ctrl.copy()
            arm_q = self.robot.arm_q_from_qpos(qpos)
            ee_pos = d.body(self.robot.ee_body_id).xpos.copy()
            ee_quat = d.body(self.robot.ee_body_id).xquat.copy()
            cube_bid = self._cube_body_name_map.get("cube")
            cube_pos = d.body(cube_bid).xpos.copy() if cube_bid is not None else None
            cube_quat = d.body(cube_bid).xquat.copy() if cube_bid is not None else None
            grasp_contacts = self._count_grasp_contacts()
            hist = {k: np.asarray(v) for k, v in self._history.items()}
            return SimSnapshot(
                sim_time=float(d.time),
                wall_fps=float(self._fps_ema),
                realtime_factor=float(self.realtime_factor),
                paused=self.paused,
                estop=self.estop,
                qpos=qpos,
                qvel=d.qvel.copy(),
                ctrl=ctrl,
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                joint_targets=ctrl[:7].copy(),
                joint_tracking_err=np.abs(ctrl[:7] - qpos[self.robot.arm_qposadr]),
                ee_tracking_pos_err=float(hist["ee_pos_err"][-1]) if len(hist["ee_pos_err"]) else 0.0,
                ee_tracking_rot_err=float(hist["ee_rot_err"][-1]) if len(hist["ee_rot_err"]) else 0.0,
                ncon=int(d.ncon),
                scene_name=self.scene_name,
                controller_status=self._controller_status,
                cube_pos=cube_pos,
                cube_quat=cube_quat,
                grasp_contacts=grasp_contacts,
                history=hist,
            )

    def _count_grasp_contacts(self) -> int:
        """Contacts between finger geoms and the cube (grasp evidence)."""
        m, d = self.model, self.data
        try:
            cube_gid = int(m.geom("cube_geom").id)
        except KeyError:
            return 0
        finger_gids: set[int] = set()
        for bid in self.robot.finger_body_ids:
            start = int(m.body_geomadr[bid])
            for gid in range(start, start + int(m.body_geomnum[bid])):
                finger_gids.add(gid)
        n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == cube_gid and g2 in finger_gids) or (g2 == cube_gid and g1 in finger_gids):
                n += 1
        return n

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())
