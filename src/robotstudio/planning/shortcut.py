"""Path shortcutting with independent re-validation.

Random waypoint pairs are tentatively replaced by a direct joint-space
segment; the replacement is accepted ONLY if the dense segment check passes.
The final path is validated again by the caller/planner as a whole.
"""

from __future__ import annotations

import time

import numpy as np

from robotstudio.planning.base import JointSpacePlanner


def shortcut(
    planner: JointSpacePlanner,
    path: list[np.ndarray],
    rng: np.random.Generator,
    finger_opening: float,
    holding_cube: bool,
    tries: int,
    t0_budget: float | None = None,
    max_time: float = 2.0,
) -> list[np.ndarray]:
    """Greedy random shortcutting. Returns a NEW validated waypoint list."""
    if len(path) < 3:
        return list(path)
    path = [np.asarray(q, dtype=float) for q in path]
    t_start = time.perf_counter()
    for _ in range(tries):
        if t0_budget is not None and time.perf_counter() - t_start > max_time:
            break
        if len(path) < 3:
            break
        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))
        if planner._segment_valid(path[i], path[j], holding_cube):
            path = path[: i + 1] + path[j:]
    return path
