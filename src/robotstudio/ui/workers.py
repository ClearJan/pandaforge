"""Background workers for UI: planning / IK / trajectory run off the UI thread.

Workers are plain QThread subclasses that emit Qt signals with results; the
UI thread applies them. Workers never touch the engine directly.
"""

from __future__ import annotations

import mujoco
from PySide6.QtCore import QThread, Signal

from robotstudio.kinematics.ik import IKSolver
from robotstudio.kinematics.pose import Pose
from robotstudio.planning.base import JointSpacePlanner
from robotstudio.trajectory.timing import TrajectoryGenerator
from robotstudio.utils.logging import get_logger

log = get_logger("ui.worker")


class IKWorker(QThread):
    finished_ok = Signal(object)  # IKResult
    failed = Signal(str)

    def __init__(self, model, robot, target: Pose, q_init, parent=None) -> None:
        super().__init__(parent)
        self._model = model
        self._robot = robot
        self._target = target
        self._q_init = q_init

    def run(self) -> None:
        try:
            solver = IKSolver(self._model, mujoco.MjData(self._model), self._robot)
            result = solver.solve(self._target, q_init=self._q_init, n_restarts=8)
            self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("IK worker failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class PlanWorker(QThread):
    finished_ok = Signal(object)  # PlanningResult
    failed = Signal(str)

    def __init__(self, model, robot, planner: JointSpacePlanner, start, goal, holding=False, parent=None) -> None:
        super().__init__(parent)
        self._model = model
        self._robot = robot
        self._planner = planner
        self._start = start
        self._goal = goal
        self._holding = holding

    def run(self) -> None:
        try:
            result = self._planner.plan(self._start, self._goal, holding_cube=self._holding)
            self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("planning worker failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class TrajectoryWorker(QThread):
    finished_ok = Signal(object)  # Trajectory
    failed = Signal(str)

    def __init__(self, path, parent=None) -> None:
        super().__init__(parent)
        self._path = path

    def run(self) -> None:
        try:
            traj = TrajectoryGenerator().generate(self._path)
            self.finished_ok.emit(traj)
        except Exception as exc:
            self.failed.emit(f"trajectory generation failed: {exc}")
