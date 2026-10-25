"""Phase 4 physics tests: full pick-and-place through real contact dynamics.

These tests run the actual engine (headless) at elevated realtime factor and
judge success ONLY through physical evidence (cube displacement, resting
position, gripper state) - exactly what the product's success criteria demand.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from robotstudio.manipulation.fsm import (
    TERMINAL_FAIL,
    TERMINAL_OK,
    GraspFSM,
    GraspState,
    GraspTaskConfig,
)
from robotstudio.simulation.engine import SimulationEngine


@pytest.fixture()
def grasp_engine():
    eng = SimulationEngine(scene_name="tabletop", headless=True)
    eng.start(with_viewer=False)
    eng.set_speed(3.0)  # 3x realtime; physics timestep unchanged
    yield eng
    eng.stop()


def _run_task(eng, place=(0.55, -0.30), timeout_s=240.0):
    snap = eng.snapshot()
    cube_start = snap.cube_pos.copy()
    spec = GraspTaskConfig(
        cube_start=tuple(cube_start),
        cube_half=0.025,
        place_target=(place[0], place[1], 0.0),
        surface_z=0.40,
        seed=2028,
    )
    fsm = GraspFSM(eng, spec)
    eng.set_controller(fsm)
    fsm.start()

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if fsm.is_terminal:
            break
        time.sleep(0.1)
    return fsm, cube_start


def test_full_pick_and_place_success(grasp_engine):
    """The canonical demo: grasp the cube and place it at the target."""
    eng = grasp_engine
    fsm, cube_start = _run_task(eng, place=(0.55, -0.30))

    assert fsm.state in TERMINAL_OK | TERMINAL_FAIL, f"task stuck in {fsm.state.value}"
    assert fsm.state == GraspState.SUCCESS, f"task failed: {fsm.state.value}: {fsm.error}"

    # independent physical verification (not trusting the FSM's own verdict)
    final = eng.snapshot()
    cube_final = final.cube_pos
    target_xy = np.array([0.55, -0.30])
    xy_err = float(np.linalg.norm(cube_final[:2] - target_xy))
    assert xy_err <= 0.035, f"cube xy error {xy_err*1000:.0f} mm"
    assert 0.39 <= cube_final[2] <= 0.46, f"cube resting z {cube_final[2]:.3f} not on table"
    moved = float(np.linalg.norm(cube_final - cube_start))
    assert moved > 0.15, f"cube barely moved ({moved*1000:.0f} mm) - not a real transfer"
    # gripper must be open at the end
    opening = float(np.mean(final.qpos[eng.robot.finger_qposadr]))
    assert opening > 0.035, "gripper did not release"
    # all required phases must have been visited in order
    visited = [r.state for r in fsm.history]
    for required in (
        GraspState.PREGRASP,
        GraspState.APPROACH,
        GraspState.CLOSE,
        GraspState.VERIFY_GRASP,
        GraspState.LIFT,
        GraspState.TRANSFER,
        GraspState.PLACE,
        GraspState.RELEASE,
        GraspState.VERIFY_PLACE,
        GraspState.RETREAT,
        GraspState.SUCCESS,
    ):
        assert required in visited, f"phase {required.value} never visited: {[s.value for s in visited]}"


def test_roof_blocked_falls_back_to_side_grasp():
    """Shelf pick: top approach blocked by a roof -> automatic SIDE grasp.

    The candidate generator must reject the top-down pose (roof collision)
    and the FSM must complete the whole task with a horizontal grasp,
    judged by the same physical criteria as the top-grasp test.
    """
    from robotstudio.manipulation.fsm import GraspFSM, GraspState, GraspTaskConfig

    eng = SimulationEngine(scene_name="roof_grasp", headless=True)
    eng.start(with_viewer=False)
    eng.set_speed(3.0)
    try:
        import time as _t

        _t.sleep(0.6)
        snap = eng.snapshot()
        spec = GraspTaskConfig(
            cube_start=tuple(snap.cube_pos),
            cube_half=0.025,
            place_target=(0.45, -0.20, 0.0),
            surface_z=0.50,
            seed=2028,
            lift_height=0.06,
        )
        fsm = GraspFSM(eng, spec)
        eng.set_controller(fsm)
        fsm.start()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 240.0 and not fsm.is_terminal:
            time.sleep(0.1)
        assert fsm.state == GraspState.SUCCESS, f"{fsm.state.value}: {fsm.error}"
        # the chosen candidate MUST be a side grasp (top is blocked by the roof)
        assert fsm.candidate is not None and not fsm.candidate.is_top
        # physical evidence
        final = eng.snapshot()
        xy_err = float(np.linalg.norm(final.cube_pos[:2] - np.array([0.45, -0.20])))
        assert xy_err <= 0.035
        assert 0.49 <= final.cube_pos[2] <= 0.57  # resting on the place pedestal
    finally:
        eng.stop()


def test_grasp_candidate_generation_properties():
    """Unit properties of the antipodal candidate generator."""
    from robotstudio.manipulation.grasp import grasp_candidates

    cube = np.array([0.45, 0.18, 0.525])
    cands = grasp_candidates(cube, 0.025)
    assert len(cands) == 5  # top + 4 sides
    assert cands[0].is_top and not any(c.is_top for c in cands[1:])
    for c in cands:
        # approach is hand +z: pregrasp sits BACK along the approach
        offset = c.pregrasp.pos - c.grasp.pos
        assert np.allclose(offset, -c.approach * np.linalg.norm(offset), atol=1e-9)
        # hand origin one GRASP_LIFT from the cube centre, along -a
        assert np.allclose(c.grasp.pos, cube - c.approach * 0.1079, atol=1e-3)
    # top candidate matches the legacy top-down frame
    assert abs(cands[0].quat_wxyz @ np.array([0.0, 1.0, 0.0, 0.0])) > 0.999 or np.allclose(
        cands[0].quat_wxyz, [0.0, 1.0, 0.0, 0.0], atol=1e-9
    )


def test_grasp_rejects_unreachable_place(grasp_engine):
    """A place target outside the workspace must fail with a clear reason."""
    eng = grasp_engine
    spec = GraspTaskConfig(
        cube_start=(0.55, 0.0, 0.425),
        place_target=(1.4, 0.9, 0.0),  # far outside reach
        surface_z=0.40,
        seed=1,
    )
    fsm = GraspFSM(eng, spec)
    eng.set_controller(fsm)
    fsm.start()

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 240.0 and not fsm.is_terminal:
        time.sleep(0.1)
    assert fsm.state in TERMINAL_FAIL, f"expected failure, got {fsm.state.value}"
    assert fsm.error, "failure reason must be reported"


def test_abort_stops_task(grasp_engine):
    """User abort mid-task must reach ABORTED and stop the arm."""
    eng = grasp_engine
    spec = GraspTaskConfig(cube_start=(0.55, 0.0, 0.425), place_target=(0.55, -0.30, 0.0), surface_z=0.40, seed=3)
    fsm = GraspFSM(eng, spec)
    eng.set_controller(fsm)
    fsm.start()

    # let it get into motion, then abort
    deadline = time.perf_counter() + 60
    while time.perf_counter() < deadline and fsm.state not in (GraspState.APPROACH, GraspState.CLOSE, GraspState.LIFT, GraspState.TRANSFER):
        if fsm.is_terminal:
            break
        time.sleep(0.05)
    fsm.abort()
    assert fsm.state == GraspState.ABORTED
    time.sleep(0.5)
    q1 = eng.snapshot().qpos.copy()
    time.sleep(0.5)
    q2 = eng.snapshot().qpos.copy()
    drift = float(np.max(np.abs(q2 - q1)))
    assert drift < 0.05, f"arm kept moving {drift:.3f} rad after abort"
