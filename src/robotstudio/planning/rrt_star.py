"""Asymptotically optimal sampling-based planners: RRT* and Informed RRT*.

References (implemented from the papers, no third-party planning code):

- Karaman & Frazzoli, "Sampling-based Algorithms for Optimal Motion
  Planning", IJRR 2011  (RRT*, k-nearest variant).
- Gammell, Srinivasa & Barfoot, "Informed RRT*: Optimal Sampling-based Path
  Planning Focused via Direct Sampling of an Admissible Ellipsoidal Subset",
  IROS 2014  (ellipsoidal informed subset direct sampling).

Both planners reuse the shared collision interface and tree storage from
``planning/rrt.py``. Per iteration the new node:

1. samples a target (uniform / goal-biased; from the informed ellipsoid once
   a solution exists for the Informed variant),
2. selects the minimum-cost parent among the k nearest neighbours whose
   connecting segment is collision-free (vs. plain RRT's plain nearest),
3. rewires those neighbours through the new node when that lowers their
   root cost (each accepted rewire propagates the cost delta to the subtree),
4. connects/improves the goal node whenever the new node is within one step
   of it - so the goal path keeps improving until the budget runs out.

The k-nearest connection radius is ``k(n) = ceil(e * (1 + 1/d) * ln(n))``
with d = 7 (Karaman & Frazzoli Thm. 38 k-NN bound); it bounds per-iteration
collision checks, which matters at 0.05 ms/check in 7-D.

The planners record the best-solution cost per iteration
(``stats["cost_history"]``) so cost convergence can be benchmarked honestly.
Path quality is reported BOTH raw (``stats["raw_path_length"]``) and after
the usual shortcutting (``PlanningResult.path_length``).
"""

from __future__ import annotations

import numpy as np

from robotstudio.planning.base import PlanningResult
from robotstudio.planning.rrt import RRTPlanner, _Tree

_N_DIM = 7
_EULER = 2.718281828459045


def _knn_limit(n: int) -> int:
    """k(n) = ceil(e (1 + 1/d) ln n), floored at 2 (Karaman & Frazzoli 2011)."""
    return max(2, int(np.ceil(_EULER * (1.0 + 1.0 / _N_DIM) * np.log(max(n, 2)))))


class RRTStarPlanner(RRTPlanner):
    algorithm = "rrt_star"

    #: subclasses may override to restrict sampling after a solution exists
    informed = False

    # ------------------------------------------------------------------ #
    def _sample(self, rng, lo, hi, goal, goal_bias, tree: _Tree, best_cost: float):
        if self.informed and best_cost < np.inf:
            return _informed_sample(rng, best_cost, self._ellipse, lo, hi)
        if rng.random() < goal_bias:
            return goal
        return rng.uniform(lo, hi)

    def _plan_impl(self, start, goal, rng, finger_opening, holding_cube, t0) -> PlanningResult:
        cfg = self.config
        lo, hi = self.checker.robot.jnt_range[:, 0], self.checker.robot.jnt_range[:, 1]

        tree = _Tree(start)
        self._ellipse = None  # cached (C, centre, cmin) for informed sampling
        goal_idx = -1
        best_cost = np.inf
        cost_history: list[tuple[int, float]] = []
        it = 0

        while it < cfg.max_iterations:
            if self._time_left(t0) <= 0:
                break
            it += 1

            target = self._sample(rng, lo, hi, goal, cfg.goal_bias, tree, best_cost)
            i_near = tree.nearest(target)
            q_new = self._steer(tree.points[i_near], target, cfg.step_size)
            if not self.robot_limits_ok(q_new):
                continue
            if not self._segment_valid(tree.points[i_near], q_new, holding_cube):
                continue

            # ---- choose the min-cost parent among the k nearest ----
            cand = tree.near(q_new, _knn_limit(tree.n))
            best_parent = -1
            best_parent_cost = np.inf
            for c in cand:
                c = int(c)
                seg_cost = tree.cost_of(c) + float(np.linalg.norm(q_new - tree.points[c]))
                if seg_cost >= best_parent_cost:
                    continue
                if self._segment_valid(tree.points[c], q_new, holding_cube):
                    best_parent_cost = seg_cost
                    best_parent = c
            if best_parent < 0:
                continue
            idx = tree.add(q_new, best_parent)

            # ---- rewire neighbours through the new node ----
            for c in cand:
                c = int(c)
                if c == idx or c == best_parent:
                    continue
                new_cost = tree.cost_of(idx) + float(np.linalg.norm(q_new - tree.points[c]))
                if new_cost < tree.cost_of(c) - 1e-9 and self._segment_valid(
                    q_new, tree.points[c], holding_cube
                ):
                    tree.set_parent(c, idx)

            # ---- connect / improve the goal node ----
            d_goal = float(np.linalg.norm(q_new - goal))
            if d_goal <= cfg.step_size:
                c_total = tree.cost_of(idx) + d_goal
                if c_total < best_cost - 1e-9 and self._segment_valid(q_new, goal, holding_cube):
                    if goal_idx < 0:
                        goal_idx = tree.add(np.asarray(goal).copy(), idx)
                    else:
                        tree.set_parent(goal_idx, idx)
                    if self.informed and self._ellipse is None:
                        self._ellipse = _ellipse_frame(start, goal)
                    best_cost = tree.cost_of(goal_idx)

            cost_history.append((it, float(best_cost)))

        if goal_idx < 0:
            return self._result(
                False,
                None,
                tree.n,
                t0,
                "no solution found within budget",
                {"iterations": it, "cost_history": cost_history},
            )

        raw_path = tree.path_to(goal_idx)
        raw_len = float(np.sum(np.linalg.norm(np.diff(np.asarray(raw_path), axis=0), axis=1)))
        stats = {
            "iterations": it,
            "tree_size": tree.n,
            "raw_path_length": raw_len,
            "best_cost": best_cost,
            "cost_history": cost_history,
        }
        path = self._maybe_shortcut(raw_path, rng, finger_opening, holding_cube, stats)
        return self._result(True, path, tree.n, t0, "path found (asymptotically optimal search)", stats)


class InformedRRTStarPlanner(RRTStarPlanner):
    algorithm = "informed_rrt_star"
    informed = True


# --------------------------------------------------------------------------- #
# informed ellipsoid sampling (Gammell et al. 2014, direct method)
# --------------------------------------------------------------------------- #


def _ellipse_frame(start: np.ndarray, goal: np.ndarray):
    """Rotation C, centre and cmin of the informed ellipsoid for start->goal."""
    direction = goal - start
    cmin = float(np.linalg.norm(direction))
    centre = (np.asarray(start) + np.asarray(goal)) / 2.0
    U, _, Vh = np.linalg.svd(np.outer(direction, direction))
    C = U @ Vh
    if np.linalg.det(C) < 0:  # ensure a proper rotation
        U[:, -1] *= -1
        C = U @ Vh
    return C, centre, cmin


def _informed_sample(rng, cbest: float, ellipse, lo, hi) -> np.ndarray:
    """Uniform sample inside the prolate hyperspheroid, rejection to bounds."""
    C, centre, cmin = ellipse
    if cbest <= cmin + 1e-12:
        return rng.uniform(lo, hi)
    # unit ball -> scaled ball: axis0 = cbest/2, others = sqrt(cbest^2-cmin^2)/2
    L = rng.normal(size=_N_DIM)
    L /= np.linalg.norm(L)
    L *= rng.random() ** (1.0 / _N_DIM)
    L[0] *= cbest / 2.0
    L[1:] *= np.sqrt(max(cbest * cbest - cmin * cmin, 0.0)) / 2.0
    q = C @ L + centre
    if np.all(q >= lo) and np.all(q <= hi):
        return q
    # outside joint limits: retry a few times, then fall back to uniform
    for _ in range(20):
        L = rng.normal(size=_N_DIM)
        L /= np.linalg.norm(L)
        L *= rng.random() ** (1.0 / _N_DIM)
        L[0] *= cbest / 2.0
        L[1:] *= np.sqrt(max(cbest * cbest - cmin * cmin, 0.0)) / 2.0
        q = C @ L + centre
        if np.all(q >= lo) and np.all(q <= hi):
            return q
    return rng.uniform(lo, hi)
