"""RobotStudio main window: docks, wiring, and lifecycle management."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTabWidget,
    QWidget,
)

from robotstudio.config import PandaSpec
from robotstudio.control.executor import TrajectoryExecutor
from robotstudio.kinematics.pose import Pose
from robotstudio.manipulation.fsm import GraspFSM, GraspState, GraspTaskConfig
from robotstudio.planning.base import PlannerConfig
from robotstudio.planning.linear import LinearPlanner
from robotstudio.planning.rrt import RRTConnectPlanner, RRTPlanner
from robotstudio.simulation.scenes import load_scene_json, preset_scenes, save_scene_json
from robotstudio.ui.panels import (
    GraspPanel,
    IKPanel,
    JointControlPanel,
    MetricsPanel,
    PlanningPanel,
    ScenePanel,
)
from robotstudio.ui.style import DARK_QSS
from robotstudio.ui.workers import IKWorker, PlanWorker, TrajectoryWorker
from robotstudio.utils.logging import get_logger

log = get_logger("ui")

PLANNER_FACTORIES = {
    "rrt_connect": lambda checker, cfg: RRTConnectPlanner(checker, cfg),
    "rrt": lambda checker, cfg: RRTPlanner(checker, cfg),
    "linear": lambda checker, cfg: LinearPlanner(checker, cfg),
}


class QtLogHandler:
    """Forwards robotstudio log records into the UI log dock (thread-safe)."""

    def __init__(self, append_fn) -> None:
        self.append_fn = append_fn
        import logging

        self._logger = logging.getLogger("robotstudio")
        self._logger.addCallback = None  # attribute guard (unused)

    def install(self) -> None:
        import logging

        handler = logging.Handler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"))

        def emit(record):
            try:
                self.append_fn(handler.format(record))
            except Exception:
                pass

        handler.emit = emit
        self._logger.addHandler(handler)


class MainWindow(QMainWindow):
    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        self.robot = engine.robot
        self._plan_result = None
        self._fsm: GraspFSM | None = None
        self._ik_worker: IKWorker | None = None
        self._plan_worker: PlanWorker | None = None
        self._traj_worker: TrajectoryWorker | None = None
        import queue as _queue

        self._engine_errors: _queue.Queue = _queue.Queue()

        self.setWindowTitle("RobotStudio — Interactive Robot Motion Planning & Manipulation Studio")
        self.resize(1280, 860)
        self.setStyleSheet(DARK_QSS)
        self._build_ui()
        self._wire()

        # UI refresh timer (engine thread stays independent)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        self.engine.on_error(self._on_engine_error)
        # sync the scene panel with the engine's actual starting scene
        current = self.engine.current_scene_spec()
        idx = list(preset_scenes().keys()).index(current.name) if current.name in preset_scenes() else 0
        self.scene_panel.scene_combo.setCurrentIndex(idx)
        self.scene_panel.show_spec(current)
        self._refresh()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        # --- center: tabs with IK / Planning / Scene panels ---
        tabs = QTabWidget()
        self.ik_panel = IKPanel()
        self.plan_panel = PlanningPanel()
        self.scene_panel = ScenePanel(preset_scenes())
        tabs.addTab(self.ik_panel, "Kinematics / IK")
        tabs.addTab(self.plan_panel, "Motion Planning")
        tabs.addTab(self.scene_panel, "Scene")
        self.setCentralWidget(tabs)

        # --- left dock: robot control ---
        self.joint_panel = JointControlPanel(self.robot)
        self._add_dock("Robot Control", self.joint_panel, Qt.LeftDockWidgetArea)

        # --- right dock: metrics ---
        self.metrics_panel = MetricsPanel()
        self._add_dock("Metrics", self.metrics_panel, Qt.RightDockWidgetArea)

        # --- right dock below metrics: log ---
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(160)
        self._add_dock("Log", self.log_view, Qt.RightDockWidgetArea)
        QtLogHandler(self._append_log).install()

        # --- bottom dock: grasp task ---
        self.grasp_panel = GraspPanel()
        self._add_dock("Automatic Pick && Place", self.grasp_panel, Qt.BottomDockWidgetArea)

        # --- status bar ---
        self.status_label = QLabel("engine starting…")
        self.statusBar().addWidget(self.status_label, 1)

    def _add_dock(self, title: str, widget: QWidget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, dock)
        return dock

    # ------------------------------------------------------------------ #
    # wiring
    # ------------------------------------------------------------------ #
    def _wire(self) -> None:
        jp = self.joint_panel
        jp.joints_commanded.connect(self._on_joints_commanded)
        jp.home_requested.connect(self._on_home)
        jp.reset_requested.connect(self.engine.reset)

        ik = self.ik_panel
        ik.solve_requested.connect(self._on_solve_ik)
        ik.read_pose_requested.connect(self._on_read_ee)
        ik.move_to_solution.connect(self._on_move_to_solution)

        pp = self.plan_panel
        pp.plan_requested.connect(self._on_plan)
        pp.execute_requested.connect(self._on_execute)
        pp.cancel_requested.connect(self._on_cancel)
        pp.clear_path_requested.connect(lambda: self.plan_panel.set_execution_state("no plan yet", "warn"))

        gp = self.grasp_panel
        gp.grasp_requested.connect(self._on_grasp)
        gp.stop_requested.connect(self._on_grasp_stop)

        sp = self.scene_panel
        sp.scene_requested.connect(self._on_scene)
        sp.apply_scene_spec.connect(self._on_apply_spec)
        sp.save_requested.connect(self._on_save_scene)
        sp.load_requested.connect(self._on_load_scene)

    # ------------------------------------------------------------------ #
    # handlers
    # ------------------------------------------------------------------ #
    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        self.log_view.moveCursor(QTextCursor.End)

    def _on_engine_error(self, message: str) -> None:
        # called from the engine thread: enqueue, drained by the GUI refresh timer
        self._engine_errors.put(message)

    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"error: {message}")
        self._append_log(f"[error] {message}")

    def _on_joints_commanded(self, arm_q: np.ndarray, opening: float) -> None:
        try:
            self.engine.set_joints(arm_q, opening)
            self.status_label.setText("manual joint command")
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid joint target", str(exc))

    def _on_home(self) -> None:
        self.engine.reset()
        for j, q in enumerate(PandaSpec.HOME_QPOS):
            self.joint_panel.set_joint_target(j, float(q))
        self.joint_panel.set_gripper_slider(100)

    def _on_read_ee(self) -> None:
        snap = self.engine.snapshot()
        self.ik_panel.read_current(snap.ee_pos)

    def _on_solve_ik(self, pos: np.ndarray, orient_mode: int) -> None:
        snap = self.engine.snapshot()
        if orient_mode == IKPanel.TOP_DOWN:
            quat = (0.0, 1.0, 0.0, 0.0)
        elif orient_mode == IKPanel.KEEP_CURRENT:
            quat = tuple(snap.ee_quat)
        else:
            quat = None
        target = Pose.make(pos, quat)
        self.ik_panel.solve_btn.setEnabled(False)
        self._ik_worker = IKWorker(self.engine.model, self.robot, target, snap.qpos[self.robot.arm_qposadr].copy())
        self._ik_worker.finished_ok.connect(lambda r: self._on_ik_done(r))
        self._ik_worker.failed.connect(lambda m: self._on_worker_failed(self.ik_panel, m))
        self._ik_worker.start()

    def _on_ik_done(self, result) -> None:
        self.ik_panel.solve_btn.setEnabled(True)
        self.ik_panel.set_result(result)
        if result.success:
            self.plan_panel.set_goal(result.arm_q)
            self.status_label.setText("IK solved - goal loaded into planning panel")

    def _on_move_to_solution(self, arm_q: np.ndarray) -> None:
        self.engine.set_joints(arm_q)
        for j, q in enumerate(arm_q):
            self.joint_panel.set_joint_target(j, float(q))

    def _on_plan(self, algo: str, seed: int, budget: float, goal_q: np.ndarray) -> None:
        snap = self.engine.snapshot()
        start = snap.qpos[self.robot.arm_qposadr].copy()
        goal = self.robot.clip_to_limits(np.asarray(goal_q, dtype=float))
        checker = self._fresh_checker()
        cfg = PlannerConfig(seed=seed, max_planning_time=budget)
        planner = PLANNER_FACTORIES[algo](checker, cfg)
        self.plan_panel.plan_btn.setEnabled(False)
        self.plan_panel.set_execution_state("planning…", "warn")
        self._plan_worker = PlanWorker(self.engine.model, self.robot, planner, start, goal)
        self._plan_worker.finished_ok.connect(self._on_plan_done)
        self._plan_worker.failed.connect(lambda m: self._on_worker_failed(self.plan_panel, m))
        self._plan_worker.start()

    def _on_plan_done(self, result) -> None:
        self.plan_panel.plan_btn.setEnabled(True)
        self.plan_panel.set_result(result)
        self._plan_result = result
        self.plan_panel.execute_btn.setEnabled(result.success)
        if result.success:
            self.status_label.setText(
                f"plan found ({result.algorithm}, {result.planning_time:.2f}s) - ready to execute"
            )
        else:
            self.status_label.setText(f"planning failed: {result.message}")

    def _on_execute(self) -> None:
        if self._plan_result is None or not self._plan_result.success:
            return
        path = self._plan_result.path
        self.plan_panel.set_execution_state("generating trajectory…", "warn")
        self._traj_worker = TrajectoryWorker(path)
        self._traj_worker.finished_ok.connect(self._on_traj_ready)
        self._traj_worker.failed.connect(lambda m: self._on_worker_failed(self.plan_panel, m))
        self._traj_worker.start()

    def _on_traj_ready(self, traj) -> None:
        try:
            executor = TrajectoryExecutor(traj)
        except Exception as exc:
            self.plan_panel.set_execution_state(f"execution refused: {exc}", "err")
            return
        self.engine.set_controller(executor)
        self.plan_panel.set_execution_state(
            f"executing trajectory ({traj.duration:.2f} s)…", "warn"
        )
        self._active_executor = executor

    def _on_cancel(self) -> None:
        self.engine.set_estop(True)
        self.engine.set_estop(False)
        self.engine.set_controller(None)
        self.engine.set_joints(self.engine.snapshot().qpos[self.robot.arm_qposadr].copy())
        self.plan_panel.set_execution_state("cancelled", "warn")
        self.status_label.setText("motion cancelled")

    def _on_grasp(self, obj_xyz: tuple, place_xyz: tuple) -> None:
        if self._fsm is not None and not self._fsm.is_terminal:
            QMessageBox.information(self, "Task running", "A pick&place task is already running. Stop it first.")
            return
        spec = GraspTaskConfig(
            cube_start=obj_xyz,
            cube_half=0.025,
            place_target=place_xyz,
            surface_z=self._surface_z(),
            seed=2028,
        )
        fsm = GraspFSM(self.engine, spec)
        self._fsm = fsm
        self.grasp_panel.bind_fsm(fsm)
        self.grasp_panel.reset_view()
        self.engine.set_controller(fsm)
        fsm.start()
        self.status_label.setText("pick&place task started")

    def _on_grasp_stop(self) -> None:
        if self._fsm is not None and not self._fsm.is_terminal:
            self._fsm.abort()
            self.status_label.setText("pick&place task aborted")

    def _on_scene(self, name: str) -> None:
        spec = preset_scenes()[name]
        self.engine.apply_scene_spec(spec)
        self.scene_panel.show_spec(spec)
        self._fsm = None
        self.grasp_panel.reset_view()
        self.grasp_panel.bind_fsm(None)
        self.status_label.setText(f"scene switched to {name!r}")

    def _on_apply_spec(self, spec) -> None:
        self.engine.apply_scene_spec(spec)
        self.status_label.setText(f"scene {spec.name!r} updated")

    def _on_save_scene(self, path: str) -> None:
        spec = self.engine.current_scene_spec()
        full = save_scene_json(spec, path)
        self.status_label.setText(f"scene saved: {full}")

    def _on_load_scene(self, path: str) -> None:
        try:
            spec = load_scene_json(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load scene", f"could not load {path}:\n{exc}")
            return
        self.engine.apply_scene_spec(spec)
        self.scene_panel.show_spec(spec)
        self.status_label.setText(f"scene loaded from {path}")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def app_process_events(self) -> None:
        """Pump the Qt loop once (also used by tests)."""
        QApplication.processEvents()

    def _fresh_checker(self):
        from robotstudio.collision.checker import CollisionChecker

        return CollisionChecker(self.engine.model, self.robot)

    def _surface_z(self) -> float:
        spec = self.engine.current_scene_spec()
        return spec.table.top_height if spec.table.enabled else 0.0

    def _on_worker_failed(self, panel, message: str) -> None:
        panel.plan_btn.setEnabled(True)
        panel.solve_btn.setEnabled(True)
        try:
            panel.set_execution_state(message, "err")
        except AttributeError:
            pass
        self.status_label.setText(f"worker failed: {message}")

    # ------------------------------------------------------------------ #
    # periodic refresh
    # ------------------------------------------------------------------ #
    def _refresh(self) -> None:
        if not self.engine.is_running:
            return
        while not self._engine_errors.empty():
            try:
                self._show_error(self._engine_errors.get_nowait())
            except Exception:
                break
        snap = self.engine.snapshot()
        self.metrics_panel.update_state(snap)
        self.joint_panel.update_state(
            snap.qpos[self.robot.arm_qposadr],
            snap.joint_tracking_err,
            float(np.mean(snap.qpos[self.robot.finger_qposadr])),
        )
        self.status_label.setText(
            f"t={snap.sim_time:.1f}s  {snap.controller_status}"
            + ("  [E-STOP]" if snap.estop else "")
            + ("  [paused]" if snap.paused else "")
        )
        if self._fsm is not None:
            self.grasp_panel.update_fsm()
            if self._fsm.is_terminal:
                fsm = self._fsm
                self._fsm = None
                if fsm.state == GraspState.SUCCESS:
                    self.status_label.setText("pick&place SUCCESS")
                else:
                    self.status_label.setText(f"pick&place {fsm.state.value}: {fsm.error}")

    def closeEvent(self, event) -> None:
        """Stop workers, engine thread, and viewer cleanly."""
        reply = QMessageBox.question(
            self,
            "Quit RobotStudio",
            "Stop the simulation and quit?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            event.ignore()
            return
        self._timer.stop()
        for w in (self._ik_worker, self._plan_worker, self._traj_worker):
            if w is not None and w.isRunning():
                w.wait(3000)
        self.engine.stop()
        event.accept()
