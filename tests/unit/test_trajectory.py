"""Phase 3 trajectory generation tests."""

from __future__ import annotations

import numpy as np
import pytest

from robotstudio.trajectory.timing import (
    TrajectoryGenerationError,
    TrajectoryGenerator,
    TrajectoryLimits,
)
from robotstudio.config import PandaSpec


def _random_path(rng, n=8, max_step=0.25):
    pts = [np.zeros(7)]
    for _ in range(n - 1):
        pts.append(pts[-1] + rng.uniform(-max_step, max_step, 7))
    return np.array(pts)


def test_generated_trajectory_respects_limits():
    rng = np.random.default_rng(0)
    gen = TrajectoryGenerator()
    for _ in range(10):
        path = _random_path(rng)
        traj = gen.generate(path)
        v_ratio, a_ratio = traj.constraint_violation()
        assert v_ratio <= 1.0 and a_ratio <= 1.0, f"v={v_ratio:.2f} a={a_ratio:.2f}"


def test_endpoints_and_duration():
    rng = np.random.default_rng(1)
    gen = TrajectoryGenerator()
    path = _random_path(rng, n=5)
    traj = gen.generate(path)
    assert np.allclose(traj.sample(0.0), path[0], atol=1e-9)
    assert np.allclose(traj.sample(traj.duration), path[-1], atol=1e-9)
    assert traj.duration > 0
    assert np.all(np.diff(traj.times) > 0)


def test_sample_clamps_time():
    rng = np.random.default_rng(2)
    gen = TrajectoryGenerator()
    traj = gen.generate(_random_path(rng, n=4))
    assert np.allclose(traj.sample(-5.0), traj.sample(0.0))
    assert np.allclose(traj.sample(1e6), traj.sample(traj.duration))


def test_long_jump_gets_stretched_not_violated():
    """A huge single-segment displacement must be time-stretched to obey limits."""
    gen = TrajectoryGenerator()
    path = np.array([np.zeros(7), np.full(7, 2.0)])
    traj = gen.generate(path)
    v_ratio, a_ratio = traj.constraint_violation()
    assert v_ratio <= 1.0 and a_ratio <= 1.0
    # sanity: duration must exceed the kinematic lower bound
    assert traj.duration > 1.0


def test_rejects_degenerate_paths():
    gen = TrajectoryGenerator()
    with pytest.raises(TrajectoryGenerationError):
        gen.generate(np.zeros((1, 7)))
    with pytest.raises(TrajectoryGenerationError):
        gen.generate(np.array([[np.nan] * 7, np.zeros(7)]))


def test_tiny_limits_produce_slow_but_feasible_motion():
    """Tiny limits cannot be 'impossible' - time scaling just slows motion.

    The generator must return a feasible (extremely slow) trajectory, and the
    result must still obey the limits numerically.
    """
    limits = TrajectoryLimits(vmax=np.full(7, 1e-3), amax=np.full(7, 1e-3))
    gen = TrajectoryGenerator(limits=limits)
    path = np.array([np.zeros(7), np.full(7, 1.0)])
    traj = gen.generate(path)
    v_ratio, a_ratio = traj.constraint_violation()
    assert v_ratio <= 1.0 and a_ratio <= 1.0
    assert traj.duration > 100.0  # sanity: genuinely slow
    assert np.isfinite(traj.duration)


def test_zero_rescale_budget_raises_when_infeasible():
    """max_scale_iterations=0 must still check feasibility and raise on failure.

    Construct limits tight enough that the trapezoidal lower-bound allocation
    itself is infeasible for the spline fit.
    """
    limits = TrajectoryLimits(vmax=np.full(7, 0.05), amax=np.full(7, 0.001))
    gen = TrajectoryGenerator(limits=limits, max_scale_iterations=0)
    path = np.array([np.zeros(7), np.full(7, 1.0)])
    with pytest.raises(TrajectoryGenerationError):
        gen.generate(path)


def test_panda_limits_match_spec():
    lim = TrajectoryLimits.from_panda()
    assert np.allclose(lim.vmax, PandaSpec.MAX_JOINT_VEL)
    assert np.allclose(lim.amax, PandaSpec.MAX_JOINT_ACC)


def test_velocity_continuous_at_waypoints():
    """C1 continuity: velocities at interior waypoints from left/right differ < 15%.

    The time-scaled cubic spline does not enforce exact C1 across waypoints;
    tight limit compliance keeps the jump small. This bounds the discontinuity.
    """
    rng = np.random.default_rng(3)
    gen = TrajectoryGenerator()
    traj = gen.generate(_random_path(rng, n=6, max_step=0.15))
    for t_w in traj.times[1:-1]:
        v_left = traj.sample_velocity(t_w - 1e-6)
        v_right = traj.sample_velocity(t_w + 1e-6)
        assert np.linalg.norm(v_left - v_right) <= 0.15 * np.linalg.norm(traj.limits.vmax)
