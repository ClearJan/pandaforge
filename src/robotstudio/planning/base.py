"""Unified planner API.

Every planner consumes the same inputs (start/goal configurations, the
collision checker, seed, time budget) and produces the same
:class:`PlanningResult`. Planners are pure computation - no Qt, no engine
thread, no global state - so they run on worker threads.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from robotstudio.collision.checker import CollisionChecker


@dataclass
class PlannerConfig:
    seed: int = 0
    max_planning_time: float = 5.0  # seconds
    max_iterations: int = 20000
    goal_bias: float = 0.1  # RRT: probability of sampling the goal directly
    step_size: float = 0.3  # rad, extension step
    shortcut_tries: int = 200  # post-processing attempts; 0 disables shortcutting


@dataclass
class PlanningResult:
    success: bool
    algorithm: str
    path: list[np.ndarray] | None  # list of (7,) waypoints, None on failure
    n_nodes: int  # nodes explored
    n_collision_checks: int
    planning_time: float  # seconds
    message: str
    path_length: float = 0.0  # rad, joint-space arc length
    n_waypoints: int = 0
    shortcut_reduction: float = 0.0  # fraction of length removed by shortcutting
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path is not None and self.n_waypoints == 0:
            self.n_waypoints = len(self.path)


def path_length(path: list[np.ndarray]) -> float:
    """Joint-space arc length of a waypoint list."""
    arr = np.asarray(path)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


class JointSpacePlanner(ABC):
    """Base class for 7-DoF joint-space planners."""

    algorithm = "base"

    def __init__(self, checker: CollisionChecker, config: PlannerConfig | None = None) -> None:
        self.checker = checker
        self.config = config or PlannerConfig()
        self._n_collision_checks = 0

    def _time_left(self, t0: float) -> float:
        return self.config.max_planning_time - (time.perf_counter() - t0)

    def _state_valid(self, q: np.ndarray, holding_cube: bool) -> bool:
        self._n_collision_checks += 1
        return self.checker.check_state(q, holding_cube=holding_cube).valid

    def _segment_valid(self, q0: np.ndarray, q1: np.ndarray, holding_cube: bool) -> bool:
        self._n_collision_checks += 1
        return self.checker.check_segment(q0, q1, holding_cube=holding_cube).valid

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        finger_opening: float = 0.04,
        holding_cube: bool = False,
    ) -> PlanningResult:
        """Plan from ``start`` to ``goal`` (both (7,) radians)."""
        t0 = time.perf_counter()
        self._n_collision_checks = 0
        rng = np.random.default_rng(self.config.seed)

        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if not self.checker.robot.within_limits(start) or not self.checker.robot.within_limits(goal):
            return self._result(False, None, 0, t0, "start or goal violates joint limits")

        if not self._state_valid(start, holding_cube):
            return self._result(False, None, 1, t0, "start configuration is in collision")
        if not self._state_valid(goal, holding_cube):
            return self._result(False, None, 2, t0, "goal configuration is in collision")

        if np.linalg.norm(goal - start) < 1e-9:
            return self._result(True, [start.copy()], 2, t0, "start equals goal")

        result = self._plan_impl(start, goal, rng, finger_opening, holding_cube, t0)
        return result

    @abstractmethod
    def _plan_impl(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        rng: np.random.Generator,
        finger_opening: float,
        holding_cube: bool,
        t0: float,
    ) -> PlanningResult: ...

    def _result(self, success: bool, path, n_nodes: int, t0: float, message: str, stats: dict | None = None) -> PlanningResult:
        elapsed = time.perf_counter() - t0
        return PlanningResult(
            success=success,
            algorithm=self.algorithm,
            path=path,
            n_nodes=n_nodes,
            n_collision_checks=self._n_collision_checks,
            planning_time=elapsed,
            message=message,
            path_length=path_length(path) if path else 0.0,
            n_waypoints=len(path) if path else 0,
            stats=stats or {},
        )
