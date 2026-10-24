"""UI panels: robot control, IK, planning, grasp task, scene editing.

Panels emit signals only; MainWindow wires them to engine commands. All
numeric displays carry explicit units.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from robotstudio.config import PandaSpec


def _deg_spin(lo, hi, val, step=1.0):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setDecimals(1)
    s.setSingleStep(step)
    s.setSuffix("°")
    return s


def _m_spin(lo, hi, val, step=0.01):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setDecimals(3)
    s.setSingleStep(step)
    s.setSuffix(" m")
    return s


# --------------------------------------------------------------------------- #


class JointControlPanel(QWidget):
    """Seven joint sliders + gripper, with live state readout and limits."""

    joints_commanded = Signal(np.ndarray, float)  # arm_q (rad), gripper opening (m/finger)
    home_requested = Signal()
    reset_requested = Signal()

    def __init__(self, robot, parent=None) -> None:
        super().__init__(parent)
        self.robot = robot
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.sliders: list[QSlider] = []
        self.spins: list[QDoubleSpinBox] = []
        self._actual_labels: list[QLabel] = []
        self._err_labels: list[QLabel] = []
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        for col, h in enumerate(["Joint", "Target", "Command (°)", "Actual (°)", "Error (°)"]):
            grid.addWidget(QLabel(f"<b>{h}</b>"), 0, col)
        for j in range(robot.n_arm):
            lo, hi = np.degrees(robot.jnt_range[j])
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(lo * 10), int(hi * 10))
            spin = _deg_spin(lo, hi, 0.0, step=1.0)
            actual = QLabel("0.0")
            err = QLabel("0.0")
            slider.valueChanged.connect(self._on_slider)
            spin.valueChanged.connect(self._on_spin)
            grid.addWidget(QLabel(f"J{j + 1}"), j + 1, 0)
            grid.addWidget(slider, j + 1, 1)
            grid.addWidget(spin, j + 1, 2)
            grid.addWidget(actual, j + 1, 3)
            grid.addWidget(err, j + 1, 4)
            self.sliders.append(slider)
            self.spins.append(spin)
            self._actual_labels.append(actual)
            self._err_labels.append(err)

        grip_row = QHBoxLayout()
        grip_row.addWidget(QLabel("Gripper"))
        self.grip_slider = QSlider(Qt.Horizontal)
        self.grip_slider.setRange(0, 100)  # percent open
        self.grip_slider.setValue(100)
        self.grip_slider.valueChanged.connect(self._on_slider)
        self.grip_value = QLabel("100 % (0.040 m)")
        grip_row.addWidget(self.grip_slider, 1)
        grip_row.addWidget(self.grip_value)
        grid.addLayout(grip_row, robot.n_arm + 1, 0, 1, 5)
        layout.addLayout(grid)

        btn_row = QHBoxLayout()
        home_btn = QPushButton("Home")
        home_btn.clicked.connect(self.home_requested.emit)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_requested.emit)
        btn_row.addWidget(home_btn)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        for j, q in enumerate(PandaSpec.HOME_QPOS):
            self.set_joint_target(j, float(q))

    # ------------------------------------------------------------------ #
    def _gripper_opening(self) -> float:
        return self.grip_slider.value() / 100.0 * PandaSpec.FINGER_OPEN

    def _on_slider(self) -> None:
        if self._updating:
            return
        self._updating = True
        for sl, sp in zip(self.sliders, self.spins):
            sp.setValue(sl.value() / 10.0)
        self._updating = False
        self.joints_commanded.emit(self.target_q(), self._gripper_opening())

    def _on_spin(self) -> None:
        if self._updating:
            return
        self._updating = True
        for sl, sp in zip(self.sliders, self.spins):
            sl.setValue(int(sp.value() * 10))
        self._updating = False
        self.joints_commanded.emit(self.target_q(), self._gripper_opening())

    def target_q(self) -> np.ndarray:
        return np.radians([sp.value() for sp in self.spins])

    def set_joint_target(self, j: int, q_rad: float) -> None:
        self._updating = True
        self.sliders[j].setValue(int(np.degrees(q_rad) * 10))
        self.spins[j].setValue(np.degrees(q_rad))
        self._updating = False

    def set_gripper_slider(self, percent: int) -> None:
        self._updating = True
        self.grip_slider.setValue(percent)
        self._updating = False

    def update_state(self, arm_q: np.ndarray, tracking: np.ndarray, opening: float) -> None:
        for j in range(self.robot.n_arm):
            self._actual_labels[j].setText(f"{np.degrees(arm_q[j]):.1f}")
            self._err_labels[j].setText(f"{np.degrees(tracking[j]):.2f}")
        self.grip_value.setText(f"{self.grip_slider.value()} % ({opening:.3f} m)")


# --------------------------------------------------------------------------- #


class IKPanel(QWidget):
    """End-effector target input, IK solve, and result application."""

    solve_requested = Signal(np.ndarray, int)  # target position (3,), orientation mode
    move_to_solution = Signal(np.ndarray)  # arm q (7,)
    read_pose_requested = Signal()

    TOP_DOWN, KEEP_CURRENT, FREE = 0, 1, 2

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        self.pos_spins = [_m_spin(-1.0, 1.2, v) for v in (0.45, 0.0, 0.55)]
        form.addRow("X", self.pos_spins[0])
        form.addRow("Y", self.pos_spins[1])
        form.addRow("Z", self.pos_spins[2])

        self.orient_combo = QComboBox()
        self.orient_combo.addItems(["Top-down (hand z down)", "Keep current orientation", "Free orientation"])
        form.addRow("Orientation", self.orient_combo)

        btn_row = QHBoxLayout()
        self.solve_btn = QPushButton("Solve IK")
        self.solve_btn.setObjectName("primaryButton")
        self.solve_btn.clicked.connect(self._emit_solve)
        read_btn = QPushButton("Read current EE")
        read_btn.clicked.connect(self.read_pose_requested.emit)
        self.apply_btn = QPushButton("Move to solution")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._emit_apply)
        btn_row.addWidget(self.solve_btn)
        btn_row.addWidget(read_btn)
        btn_row.addWidget(self.apply_btn)
        form.addRow(btn_row)

        self.result_label = QLabel("idle")
        self.result_label.setWordWrap(True)
        form.addRow("Result", self.result_label)
        self.solution: np.ndarray | None = None

    def _emit_solve(self) -> None:
        pos = np.array([sp.value() for sp in self.pos_spins])
        self.solve_requested.emit(pos, self.orient_combo.currentIndex())

    def _emit_apply(self) -> None:
        if self.solution is not None:
            self.move_to_solution.emit(self.solution)

    def set_result(self, ik_result) -> None:
        if ik_result is None:
            return
        if ik_result.success:
            self.solution = ik_result.arm_q
            self.apply_btn.setEnabled(True)
            state, text = "ok", (
                f"solved in {ik_result.iterations} iters | pos err {ik_result.pos_err * 1000:.1f} mm | "
                f"rot err {np.degrees(ik_result.rot_err):.2f}°"
            )
        else:
            self.solution = None
            self.apply_btn.setEnabled(False)
            state, text = "err", (
                f"FAILED: {ik_result.message} (pos err {ik_result.pos_err * 1000:.1f} mm, "
                f"rot err {np.degrees(ik_result.rot_err):.2f}°)"
            )
        self._set_state_label(self.result_label, state, text)

    def read_current(self, ee_pos: np.ndarray) -> None:
        for sp, v in zip(self.pos_spins, ee_pos):
            sp.setValue(float(v))

    @staticmethod
    def _set_state_label(label: QLabel, state: str, text: str) -> None:
        label.setProperty("state", state)
        label.setText(text)
        label.style().unpolish(label)
        label.style().polish(label)


# --------------------------------------------------------------------------- #


class PlanningPanel(QWidget):
    """Algorithm selection, planning, execution, and result metrics."""

    plan_requested = Signal(str, int, float, np.ndarray)  # algo key, seed, budget, goal q
    execute_requested = Signal()
    cancel_requested = Signal()
    clear_path_requested = Signal()

    ALGOS = [("RRT-Connect", "rrt_connect"), ("RRT", "rrt"), ("Linear (baseline)", "linear")]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        self.algo_combo = QComboBox()
        for label, _key in self.ALGOS:
            self.algo_combo.addItem(label)
        form.addRow("Algorithm", self.algo_combo)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        form.addRow("Random seed", self.seed_spin)
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(1.0, 120.0)
        self.time_spin.setValue(10.0)
        self.time_spin.setSuffix(" s")
        form.addRow("Time budget", self.time_spin)

        self.goal_spins = [_deg_spin(-180, 180, 0.0) for _ in range(7)]
        goal_grid = QGridLayout()
        for j, sp in enumerate(self.goal_spins):
            goal_grid.addWidget(QLabel(f"J{j + 1}"), 0, j)
            goal_grid.addWidget(sp, 1, j)
        form.addRow(QLabel("Goal joint targets (°):"))
        form.addRow(goal_grid)

        btn_row = QHBoxLayout()
        self.plan_btn = QPushButton("Plan")
        self.plan_btn.setObjectName("primaryButton")
        self.execute_btn = QPushButton("Execute")
        self.execute_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("dangerButton")
        clear_btn = QPushButton("Clear Path")
        self.plan_btn.clicked.connect(self._emit_plan)
        self.execute_btn.clicked.connect(self.execute_requested.emit)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        clear_btn.clicked.connect(self.clear_path_requested.emit)
        for b in (self.plan_btn, self.execute_btn, self.cancel_btn, clear_btn):
            btn_row.addWidget(b)
        form.addRow(btn_row)

        self.result_label = QLabel("no plan yet")
        self.result_label.setWordWrap(True)
        form.addRow("Plan result", self.result_label)

    def _emit_plan(self) -> None:
        key = self.ALGOS[self.algo_combo.currentIndex()][1]
        goal = np.radians([sp.value() for sp in self.goal_spins])
        self.plan_requested.emit(key, self.seed_spin.value(), self.time_spin.value(), goal)

    def set_goal(self, q_rad: np.ndarray) -> None:
        for sp, q in zip(self.goal_spins, q_rad):
            sp.setValue(float(np.degrees(q)))

    def set_result(self, res) -> None:
        if res.success:
            state = "ok"
            text = (
                f"OK: {res.n_waypoints} waypoints, {res.path_length:.2f} rad, "
                f"{res.planning_time:.2f} s, {res.n_nodes} nodes, {res.n_collision_checks} checks, "
                f"shortcut -{res.stats.get('shortcut_reduction', 0) * 100:.0f}%"
            )
        else:
            state = "err"
            text = f"FAILED: {res.message} ({res.planning_time:.2f} s, {res.n_nodes} nodes)"
        IKPanel._set_state_label(self.result_label, state, text)

    def set_execution_state(self, text: str, state: str = "warn") -> None:
        IKPanel._set_state_label(self.result_label, state, text)


# --------------------------------------------------------------------------- #


class GraspPanel(QWidget):
    """Automatic pick-and-place task control."""

    grasp_requested = Signal(tuple, tuple)  # object xyz, place xyz
    stop_requested = Signal()

    PHASES = [
        "IDLE", "PREGRASP", "APPROACH", "CLOSE", "VERIFY_GRASP", "LIFT",
        "RETRY_DECISION", "TRANSFER", "PLACE", "RELEASE", "VERIFY_PLACE",
        "RETREAT", "SUCCESS", "FAILED", "TIMEOUT", "ABORTED",
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        self.obj_spins = [_m_spin(-1.0, 1.2, v) for v in (0.55, 0.0, 0.425)]
        self.place_spins = [_m_spin(-1.0, 1.2, v) for v in (0.55, -0.3, 0.425)]
        form.addRow("Object x/y/z", self._row(self.obj_spins))
        form.addRow("Place x/y/z", self._row(self.place_spins))

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Pick && Place")
        self.start_btn.setObjectName("primaryButton")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerButton")
        self.start_btn.clicked.connect(self._emit_start)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        form.addRow(btn_row)

        self.phase_list = QListWidget()
        self.phase_list.setFixedHeight(170)
        for p in self.PHASES:
            self.phase_list.addItem(p)
        form.addRow("State machine", self.phase_list)

        self.error_label = QLabel("no active task")
        self.error_label.setWordWrap(True)
        form.addRow("Status", self.error_label)
        self._fsm = None

    @staticmethod
    def _row(spins) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for sp in spins:
            lay.addWidget(sp)
        return w

    def _emit_start(self) -> None:
        obj = tuple(sp.value() for sp in self.obj_spins)
        place = tuple(sp.value() for sp in self.place_spins)
        self.grasp_requested.emit(obj, place)

    def read_object_pos(self, pos: np.ndarray) -> None:
        for sp, v in zip(self.obj_spins[:2], pos[:2]):
            sp.setValue(float(v))

    def bind_fsm(self, fsm) -> None:
        self._fsm = fsm

    def update_fsm(self) -> None:
        fsm = self._fsm
        if fsm is None:
            return
        current = fsm.state.value
        visited = {r.state.value for r in fsm.history}
        for i, name in enumerate(self.PHASES):
            item = self.phase_list.item(i)
            if name == current:
                item.setBackground(Qt.darkBlue)
                item.setForeground(Qt.white)
            elif name in visited:
                item.setForeground(Qt.green)
            else:
                item.setForeground(Qt.gray)
        err = fsm.error
        if fsm.is_terminal:
            if fsm.state.name == "SUCCESS":
                IKPanel._set_state_label(self.error_label, "ok", "task SUCCESS - cube placed and verified")
            else:
                IKPanel._set_state_label(self.error_label, "err", f"{fsm.state.value}: {err}")
        else:
            IKPanel._set_state_label(self.error_label, "warn", fsm.status_text)

    def reset_view(self) -> None:
        for i in range(self.phase_list.count()):
            self.phase_list.item(i).setBackground(Qt.transparent)
            self.phase_list.item(i).setForeground(Qt.lightGray)
        IKPanel._set_state_label(self.error_label, "warn", "no active task")


# --------------------------------------------------------------------------- #


class ScenePanel(QWidget):
    """Preset scene selection, obstacle editing, save/load."""

    scene_requested = Signal(str)
    apply_scene_spec = Signal(object)  # SceneSpec
    save_requested = Signal(str)
    load_requested = Signal(str)

    def __init__(self, presets: dict, parent=None) -> None:
        super().__init__(parent)
        self._presets = presets
        self._current_spec = None
        form = QFormLayout(self)

        self.scene_combo = QComboBox()
        for name in presets:
            self.scene_combo.addItem(name)
        load_btn = QPushButton("Load preset")
        load_btn.clicked.connect(self._emit_preset)
        row = QHBoxLayout()
        row.addWidget(self.scene_combo, 1)
        row.addWidget(load_btn)
        form.addRow("Preset", row)

        self.obstacle_table = QTableWidget(0, 7)
        self.obstacle_table.setHorizontalHeaderLabels(
            ["name", "type", "x", "y", "z", "size", "enabled"]
        )
        self.obstacle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.obstacle_table.setFixedHeight(160)
        form.addRow("Obstacles", self.obstacle_table)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply changes")
        apply_btn.setObjectName("primaryButton")
        save_btn = QPushButton("Save scene…")
        open_btn = QPushButton("Open scene…")
        apply_btn.clicked.connect(self._emit_apply)
        save_btn.clicked.connect(lambda: self.save_requested.emit("scenes/user_scene.json"))
        open_btn.clicked.connect(lambda: self.load_requested.emit("scenes/user_scene.json"))
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(open_btn)
        form.addRow(btn_row)

    def _emit_preset(self) -> None:
        name = self.scene_combo.currentText()
        self.scene_requested.emit(name)

    def show_spec(self, spec) -> None:
        self._current_spec = spec
        obs = [o for o in spec.obstacles if o.enabled]
        self.obstacle_table.setRowCount(len(obs))
        for i, o in enumerate(obs):
            if o.kind == "box":
                size_str = "x".join(f"{v:.3f}" for v in o.size)
            elif o.kind == "sphere":
                size_str = f"r={o.radius:.3f}"
            else:
                size_str = f"r={o.radius:.3f},h={o.half_length:.3f}"
            vals = [o.name, o.kind, f"{o.pos[0]:.3f}", f"{o.pos[1]:.3f}", f"{o.pos[2]:.3f}", size_str]
            for c, v in enumerate(vals):
                self.obstacle_table.setItem(i, c, QTableWidgetItem(str(v)))
            chk = QTableWidgetItem()
            chk.setCheckState(Qt.Checked if o.enabled else Qt.Unchecked)
            self.obstacle_table.setItem(i, 6, chk)

    def _emit_apply(self) -> None:
        if self._current_spec is None:
            return
        # NOTE: obstacle table editing is applied to a copy of the spec


        spec = self._current_spec
        for i in range(self.obstacle_table.rowCount()):
            if i >= len(spec.obstacles):
                break
            o = spec.obstacles[i]
            try:
                x = float(self.obstacle_table.item(i, 2).text())
                y = float(self.obstacle_table.item(i, 3).text())
                z = float(self.obstacle_table.item(i, 4).text())
            except (TypeError, ValueError):
                continue
            o.pos = (x, y, z)
            enabled = self.obstacle_table.item(i, 6).checkState() == Qt.Checked
            o.enabled = bool(enabled)
        self.apply_scene_spec.emit(spec)


class MetricsPanel(QWidget):
    """Live curves (pyqtgraph) + numeric indicators."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        pg.setConfigOptions(antialias=True, background="#14171b", foreground="#d6dbe1")
        layout = QVBoxLayout(self)
        self.status_labels: dict[str, QLabel] = {}
        grid = QGridLayout()
        for i, key in enumerate(["sim_time", "rtf", "ee_err", "jtrack", "contacts", "scene"]):
            lab = QLabel("-")
            grid.addWidget(QLabel(key), i // 2, (i % 2) * 2)
            grid.addWidget(lab, i // 2, (i % 2) * 2 + 1)
            self.status_labels[key] = lab
        layout.addLayout(grid)

        self.joint_plot = pg.PlotWidget(title="Joint angles [rad]")
        self.joint_plot.addLegend(offset=(8, 8), labelTextSize="7pt")
        self.joint_curves = [
            self.joint_plot.plot(pen=(i, 7), name=f"J{i + 1}") for i in range(7)
        ]
        self.joint_plot.setLabel("left", "angle [rad]")
        self.joint_plot.setLabel("bottom", "sim time [s]")
        layout.addWidget(self.joint_plot, 2)

        self.err_plot = pg.PlotWidget(title="EE position tracking error [mm]")
        self.err_curve = self.err_plot.plot(pen="#e6c07b")
        self.err_plot.setLabel("left", "error [mm]")
        self.err_plot.setLabel("bottom", "sim time [s]")
        layout.addWidget(self.err_plot, 1)

    def update_state(self, snap) -> None:
        self.status_labels["sim_time"].setText(f"{snap.sim_time:.2f} s")
        self.status_labels["rtf"].setText(f"{snap.realtime_factor:.2f}x ({snap.wall_fps:.0f} fps)")
        self.status_labels["ee_err"].setText(f"{snap.ee_tracking_pos_err * 1000:.2f} mm")
        self.status_labels["jtrack"].setText(f"{np.degrees(np.max(snap.joint_tracking_err)):.2f}°")
        self.status_labels["contacts"].setText(str(snap.ncon))
        self.status_labels["scene"].setText(snap.scene_name)

        hist = snap.history
        if len(hist.get("t", [])) > 1:
            t = hist["t"]
            arm_q = np.asarray(hist["arm_q"])
            if arm_q.ndim == 2:
                for i, curve in enumerate(self.joint_curves):
                    curve.setData(t, arm_q[:, i])
            ee = np.asarray(hist["ee_pos_err"]) * 1000.0
            self.err_curve.setData(t, ee)
