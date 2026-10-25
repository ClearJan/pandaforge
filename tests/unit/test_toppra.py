"""TOPP-RA time-optimal parameterization tests."""

from __future__ import annotations

import numpy as np

from robotstudio.trajectory.timing import TrajectoryGenerator, TrajectoryLimits
from robotstudio.trajectory.toppra import ToppRaGenerator


def _random_path(rng, n=8, max_step=0.25):
    pts = [np.zeros(7)]
    for _ in range(n - 1):
        pts.append(pts[-1] + rng.uniform(-max_step, max_step, 7))
    return np.array(pts)


def test_topp_duration_leq_legacy():
    """TOPP-RA is time-optimal along the path: never slower than the
    conservative trapezoidal-spline allocation (20 random paths)."""
    rng = np.random.default_rng(11)
    gen = TrajectoryGenerator(method="spline")
    topp = ToppRaGenerator()
    for _ in range(20):
        path = _random_path(rng)
        t_legacy = gen.generate(path)
        t_topp = topp.generate(path)
        assert t_topp.duration <= t_legacy.duration + 1e-6, (
            f"TOPP {t_topp.duration:.3f}s > legacy {t_legacy.duration:.3f}s"
        )


def test_topp_respects_limits_numerically():
    rng = np.random.default_rng(12)
    topp = ToppRaGenerator()
    for _ in range(10):
        traj = topp.generate(_random_path(rng))
        v_ratio, a_ratio = traj.constraint_violation()
        assert v_ratio <= 1.0 and a_ratio <= 1.0


def test_topp_zero_boundary_velocities():
    rng = np.random.default_rng(13)
    traj = ToppRaGenerator().generate(_random_path(rng, n=6))
    v0 = traj.sample_velocity(0.0)
    v1 = traj.sample_velocity(traj.duration)
    assert np.linalg.norm(v0) < 1e-6 and np.linalg.norm(v1) < 1e-6


def test_topp_endpoints_and_time_clamped():
    rng = np.random.default_rng(14)
    path = _random_path(rng, n=5)
    traj = ToppRaGenerator().generate(path)
    assert np.allclose(traj.sample(0.0), path[0], atol=1e-9)
    assert np.allclose(traj.sample(traj.duration), path[-1], atol=1e-9)
    assert np.allclose(traj.sample(-3.0), traj.sample(0.0))
    assert np.allclose(traj.sample(1e9), traj.sample(traj.duration))


def test_topp_handles_degenerate_paths():
    topp = ToppRaGenerator()
    rng = np.random.default_rng(15)
    # duplicate consecutive waypoints must not break the finite differences
    path = _random_path(rng, n=5)
    path = np.insert(path, 3, path[3], axis=0)
    traj = topp.generate(path)
    assert traj.duration > 0 and np.isfinite(traj.duration)
    # two-point path falls back cleanly
    t2 = topp.generate(np.array([np.zeros(7), np.full(7, 0.5)]))
    assert t2.duration > 0


def test_default_generator_uses_topp_and_shortens():
    """The default factory is TOPP-RA and produces shorter trajectories."""
    rng = np.random.default_rng(16)
    path = _random_path(rng, n=9, max_step=0.3)
    default = TrajectoryGenerator().generate(path)
    legacy = TrajectoryGenerator(method="spline").generate(path)
    assert default.duration <= legacy.duration + 1e-6


def test_topp_with_tiny_limits_still_feasible():
    """Tiny limits on a planner-like path (few long segments, low curvature).

    NOTE on a real discretization boundary: TOPP derives curvature by finite
    differences of the waypoint polyline; sharp synthetic zig-zags have FD
    curvature far above the executed spline's true curvature, so TOPP can be
    CONSERVATIVE there (feasible but slower). Dominance is asserted on
    planner-like paths (shortcut output shape); feasibility always.
    """
    limits = TrajectoryLimits(vmax=np.full(7, 0.02), amax=np.full(7, 0.02))
    rng = np.random.default_rng(17)
    # planner-like path: 3 waypoints, moderate segment lengths
    path = np.array([np.zeros(7), rng.uniform(-0.3, 0.3, 7), rng.uniform(-0.3, 0.3, 7)])
    t_topp = ToppRaGenerator(limits).generate(path)
    v_ratio, a_ratio = t_topp.constraint_violation()
    assert v_ratio <= 1.0 and a_ratio <= 1.0
    t_legacy = TrajectoryGenerator(limits, method="spline").generate(path)
    assert t_topp.duration <= t_legacy.duration + 1e-6
    # sharp zig-zag: feasibility must still hold (fallback guarantees it)
    sharp = _random_path(rng, n=6)
    t_sharp = ToppRaGenerator(limits).generate(sharp)
    vr, ar = t_sharp.constraint_violation()
    assert vr <= 1.0 and ar <= 1.0
