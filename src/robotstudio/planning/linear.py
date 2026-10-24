"""Straight-line joint-space interpolation baseline.

Deliberately naive: one waypoint per configured step, then a single dense
validation. Used as the reference baseline in benchmarks and as the planner
of first resort when the straight line happens to be collision-free.
"""

from __future__ import annotations

import numpy as np

from robotstudio.planning.base import JointSpacePlanner, PlanningResult


class LinearPlanner(JointSpacePlanner):
    algorithm = "linear"

    def _plan_impl(self, start, goal, rng, finger_opening, holding_cube, t0) -> PlanningResult:
        step = self.config.step_size
        n = max(2, int(np.ceil(np.max(np.abs(goal - start)) / step)) + 1)
        path = [(1 - t) * start + t * goal for t in np.linspace(0.0, 1.0, n)]
        rep = self.checker.check_path(path, finger_opening, holding_cube)
        self._n_collision_checks += rep.n_samples
        if not rep.valid:
            return self._result(False, None, 2, t0, f"straight-line path in collision: {rep.message}", {"n_samples": rep.n_samples})
        return self._result(True, path, 2, t0, "straight line is collision-free", {"n_samples": rep.n_samples})
