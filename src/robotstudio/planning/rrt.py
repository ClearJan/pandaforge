"""Sampling-based planners: RRT and RRT-Connect (independent implementations).

References: LaValle & Kuffner, "Randomized Kinodynamic Planning" (2001);
Kuffner & LaValle, "RRT-Connect: An Efficient Approach to Single-Query Path
Planning" (ICRA 2000). Implemented here in joint space with the shared
collision interface; no third-party planning code is used.

RRT: single tree, goal-biased uniform sampling, bounded extension step.
RRT-Connect: two trees growing toward each other with CONNECT (repeat EXTEND
until obstacle or goal reached); typically 1-2 orders of magnitude fewer
iterations in uncluttered scenes.
"""

from __future__ import annotations

import numpy as np

from robotstudio.planning.base import JointSpacePlanner, PlanningResult
from robotstudio.planning.shortcut import shortcut


class RRTPlanner(JointSpacePlanner):
    algorithm = "rrt"

    def _plan_impl(self, start, goal, rng, finger_opening, holding_cube, t0) -> PlanningResult:
        cfg = self.config
        lo, hi = self.checker.robot.jnt_range[:, 0], self.checker.robot.jnt_range[:, 1]

        # Tree as parallel lists for speed.
        nodes = [start.copy()]
        parents = [-1]

        goal_reached_idx = -1
        it = 0
        while it < cfg.max_iterations:
            if self._time_left(t0) <= 0:
                return self._result(False, None, len(nodes), t0, "planning timeout", {"iterations": it})
            it += 1
            if rng.random() < cfg.goal_bias:
                target = goal
            else:
                target = rng.uniform(lo, hi)

            i_near = self._nearest(nodes, target)
            q_new = self._steer(nodes[i_near], target, cfg.step_size)

            if not self.robot_limits_ok(q_new):
                continue
            if not self._segment_valid(nodes[i_near], q_new, holding_cube):
                continue

            nodes.append(q_new)
            parents.append(i_near)
            idx_new = len(nodes) - 1

            if np.linalg.norm(q_new - goal) <= cfg.step_size and self._segment_valid(q_new, goal, holding_cube):
                nodes.append(goal.copy())
                parents.append(idx_new)
                goal_reached_idx = len(nodes) - 1
                break

        if goal_reached_idx < 0:
            return self._result(False, None, len(nodes), t0, "max iterations reached without finding the goal", {"iterations": it})

        path = self._backtrack(nodes, parents, goal_reached_idx)
        path = self._densify_endpoints(path, goal)
        stats = {"iterations": it, "tree_size": len(nodes)}
        path = self._maybe_shortcut(path, rng, finger_opening, holding_cube, stats)
        return self._result(True, path, len(nodes), t0, "path found", stats)

    def robot_limits_ok(self, q: np.ndarray) -> bool:
        return self.checker.robot.within_limits(q)

    @staticmethod
    def _nearest(nodes: list[np.ndarray], target: np.ndarray) -> int:
        dists = [np.linalg.norm(n - target) for n in nodes]
        return int(np.argmin(dists))

    @staticmethod
    def _steer(q_near: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
        delta = target - q_near
        dist = np.linalg.norm(delta)
        if dist <= step:
            return target.copy()
        return q_near + delta / dist * step

    @staticmethod
    def _backtrack(nodes, parents, idx) -> list[np.ndarray]:
        path = []
        while idx >= 0:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()
        return path

    @staticmethod
    def _densify_endpoints(path: list[np.ndarray], goal: np.ndarray) -> list[np.ndarray]:
        if np.linalg.norm(path[-1] - goal) > 1e-9:
            path.append(goal.copy())
        return path

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

        t1_nodes, t1_par = [start.copy()], [-1]  # tree rooted at start
        t2_nodes, t2_par = [goal.copy()], [-1]  # tree rooted at goal
        t1_is_start = True
        connected = False
        it = 0
        meet1 = meet2 = -1

        while not connected:
            if it >= cfg.max_iterations:
                return self._result(False, None, len(t1_nodes) + len(t2_nodes), t0, "max iterations reached", {"iterations": it})
            if self._time_left(t0) <= 0:
                return self._result(False, None, len(t1_nodes) + len(t2_nodes), t0, "planning timeout", {"iterations": it})
            it += 1

            target = rng.uniform(lo, hi)
            status1, idx1 = self._extend(t1_nodes, t1_par, target, cfg.step_size, holding_cube)
            if status1 != "trapped":
                q_target = t1_nodes[idx1]
                while True:
                    status2, idx2 = self._extend(t2_nodes, t2_par, q_target, cfg.step_size, holding_cube)
                    if status2 == "reached":
                        connected = True
                        meet1, meet2 = idx1, idx2
                        break
                    if status2 == "trapped":
                        break
            if not connected:
                t1_nodes, t2_nodes = t2_nodes, t1_nodes
                t1_par, t2_par = t2_par, t1_par
                t1_is_start = not t1_is_start

        # path1: t1 root -> meeting node; path2: t2 root -> meeting node
        path1 = self._backtrack(t1_nodes, t1_par, meet1)
        path2 = self._backtrack(t2_nodes, t2_par, meet2)
        full = path1 + path2[::-1][1:] if t1_is_start else path2 + path1[::-1][1:]

        stats = {"iterations": it, "tree_size_a": len(t1_nodes), "tree_size_b": len(t2_nodes)}
        full = self._maybe_shortcut(full, rng, finger_opening, holding_cube, stats)
        return self._result(True, full, len(t1_nodes) + len(t2_nodes), t0, "path found", stats)

    def _extend(self, nodes, parents, target, step, holding_cube):
        """One EXTEND step. Returns (status, new_index or -1)."""
        i_near = self._nearest(nodes, target)
        q_new = self._steer(nodes[i_near], target, step)
        if not self.robot_limits_ok(q_new):
            return "trapped", -1
        if not self._segment_valid(nodes[i_near], q_new, holding_cube):
            return "trapped", -1
        nodes.append(q_new)
        parents.append(i_near)
        if np.allclose(q_new, target, atol=1e-12):
            return "reached", len(nodes) - 1
        return "advanced", len(nodes) - 1
