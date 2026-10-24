"""Sampling-based planners: RRT and RRT-Connect (independent implementations).

References: LaValle & Kuffner, "Randomized Kinodynamic Planning" (2001);
Kuffner & LaValle, "RRT-Connect: An Efficient Approach to Single-Query Path
Planning" (ICRA 2000). Implemented here in joint space with the shared
collision interface; no third-party planning code is used.

RRT: single tree, goal-biased uniform sampling, bounded extension step.
RRT-Connect: two trees growing toward each other with CONNECT (repeat EXTEND
until obstacle or goal reached); typically 1-2 orders of magnitude fewer
iterations in uncluttered scenes.

Performance note (profiled 2026-10-25): nearest-neighbour is vectorized over
a preallocated (N,7) array - a pure-Python per-node scan was the dominant
cost (O(n) Python iterations per sample), not collision checking.
"""

from __future__ import annotations

import numpy as np

from robotstudio.planning.base import JointSpacePlanner, PlanningResult
from robotstudio.planning.shortcut import shortcut

_TREE_CAPACITY = 50000


class _Tree:
    """Flat preallocated tree with vectorized nearest-neighbour queries."""

    __slots__ = ("points", "parents", "n")

    def __init__(self, root: np.ndarray) -> None:
        self.points = np.empty((_TREE_CAPACITY, root.shape[0]))
        self.points[0] = root
        self.parents = np.full(_TREE_CAPACITY, -1, dtype=np.int64)
        self.n = 1

    def add(self, q: np.ndarray, parent: int) -> int:
        if self.n >= _TREE_CAPACITY:
            raise RuntimeError(f"planner tree capacity ({_TREE_CAPACITY}) exceeded")
        idx = self.n
        self.points[idx] = q
        self.parents[idx] = parent
        self.n += 1
        return idx

    def nearest(self, target: np.ndarray) -> int:
        d = self.points[: self.n] - target
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def path_to(self, idx: int) -> list[np.ndarray]:
        out = []
        while idx >= 0:
            out.append(self.points[idx].copy())
            idx = int(self.parents[idx])
        out.reverse()
        return out


class RRTPlanner(JointSpacePlanner):
    algorithm = "rrt"

    def _plan_impl(self, start, goal, rng, finger_opening, holding_cube, t0) -> PlanningResult:
        cfg = self.config
        lo, hi = self.checker.robot.jnt_range[:, 0], self.checker.robot.jnt_range[:, 1]

        tree = _Tree(start)
        goal_reached_idx = -1
        it = 0
        while it < cfg.max_iterations:
            if self._time_left(t0) <= 0:
                return self._result(False, None, tree.n, t0, "planning timeout", {"iterations": it})
            it += 1
            if rng.random() < cfg.goal_bias:
                target = goal
            else:
                target = rng.uniform(lo, hi)

            i_near = tree.nearest(target)
            q_new = self._steer(tree.points[i_near], target, cfg.step_size)

            if not self.robot_limits_ok(q_new):
                continue
            if not self._segment_valid(tree.points[i_near], q_new, holding_cube):
                continue

            idx_new = tree.add(q_new, i_near)

            if np.linalg.norm(q_new - goal) <= cfg.step_size and self._segment_valid(q_new, goal, holding_cube):
                goal_reached_idx = tree.add(goal.copy(), idx_new)
                break

        if goal_reached_idx < 0:
            return self._result(False, None, tree.n, t0, "max iterations reached without finding the goal", {"iterations": it})

        path = tree.path_to(goal_reached_idx)
        stats = {"iterations": it, "tree_size": tree.n}
        path = self._maybe_shortcut(path, rng, finger_opening, holding_cube, stats)
        return self._result(True, path, tree.n, t0, "path found", stats)

    def robot_limits_ok(self, q: np.ndarray) -> bool:
        return self.checker.robot.within_limits(q)

    @staticmethod
    def _steer(q_near: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
        delta = target - q_near
        dist = np.linalg.norm(delta)
        if dist <= step:
            return target.copy()
        return q_near + delta / dist * step

    def _maybe_shortcut(self, path, rng, finger_opening, holding_cube, stats) -> list[np.ndarray]:
        if self.config.shortcut_tries <= 0:
            return path
        before = float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)))
        path = shortcut(self, path, rng, finger_opening, holding_cube, self.config.shortcut_tries, t0_budget=None)
        after = float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)))
        stats["shortcut_reduction"] = 1.0 - after / before if before > 0 else 0.0
        return path


class RRTConnectPlanner(RRTPlanner):
    algorithm = "rrt_connect"

    def _plan_impl(self, start, goal, rng, finger_opening, holding_cube, t0) -> PlanningResult:
        cfg = self.config
        lo, hi = self.checker.robot.jnt_range[:, 0], self.checker.robot.jnt_range[:, 1]

        t1 = _Tree(start)  # rooted at start
        t2 = _Tree(goal)  # rooted at goal
        t1_is_start = True
        connected = False
        it = 0
        meet1 = meet2 = -1

        while not connected:
            if it >= cfg.max_iterations:
                return self._result(False, None, t1.n + t2.n, t0, "max iterations reached", {"iterations": it})
            if self._time_left(t0) <= 0:
                return self._result(False, None, t1.n + t2.n, t0, "planning timeout", {"iterations": it})
            it += 1

            target = rng.uniform(lo, hi)
            status1, idx1 = self._extend(t1, target, cfg.step_size, holding_cube)
            if status1 != "trapped":
                q_target = t1.points[idx1]
                while True:
                    status2, idx2 = self._extend(t2, q_target, cfg.step_size, holding_cube)
                    if status2 == "reached":
                        connected = True
                        meet1, meet2 = idx1, idx2
                        break
                    if status2 == "trapped":
                        break
            if not connected:
                t1, t2 = t2, t1
                t1_is_start = not t1_is_start

        path1 = t1.path_to(meet1)
        path2 = t2.path_to(meet2)
        full = path1 + path2[::-1][1:] if t1_is_start else path2 + path1[::-1][1:]

        stats = {"iterations": it, "tree_size_a": t1.n, "tree_size_b": t2.n}
        full = self._maybe_shortcut(full, rng, finger_opening, holding_cube, stats)
        return self._result(True, full, t1.n + t2.n, t0, "path found", stats)

    def _extend(self, tree: _Tree, target, step, holding_cube):
        """One EXTEND step. Returns (status, new_index or -1)."""
        i_near = tree.nearest(target)
        q_new = self._steer(tree.points[i_near], target, step)
        if not self.robot_limits_ok(q_new):
            return "trapped", -1
        if not self._segment_valid(tree.points[i_near], q_new, holding_cube):
            return "trapped", -1
        idx = tree.add(q_new, i_near)
        if np.array_equal(q_new, target):
            return "reached", idx
        return "advanced", idx
