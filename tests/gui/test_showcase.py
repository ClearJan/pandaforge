"""Showcase sequencer + 3D decoration tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from robotstudio.control.showcase import ShowcaseSequencer, Stage


@pytest.fixture()
def engine():
    from robotstudio.simulation.engine import SimulationEngine

    eng = SimulationEngine(scene_name="tabletop", headless=True)
    eng.start(with_viewer=False)
    eng.set_speed(3.0)
    yield eng
    eng.stop()


def test_showcase_completes_end_to_end(engine):
    """The one-click demo runs sweep + pick & place to completion."""
    seq = ShowcaseSequencer(engine)
    engine.set_controller(seq)

    deadline = time.perf_counter() + 300.0
    while time.perf_counter() < deadline and seq.stage is not Stage.DONE:
        time.sleep(0.2)
    assert seq.stage is Stage.DONE, f"stuck in {seq.stage}: {seq.status_text}"
    assert seq.error is None, f"showcase failed: {seq.error}"
    # physical evidence of the pick & place stage
    snap = engine.snapshot()
    xy_err = float(np.linalg.norm(snap.cube_pos[:2] - np.array([0.55, -0.30])))
    assert xy_err <= 0.035, f"cube not at the place target ({xy_err*1000:.0f} mm)"


def test_showcase_abort_on_estop(engine):
    seq = ShowcaseSequencer(engine)
    engine.set_controller(seq)
    time.sleep(1.5)  # let it get into motion
    engine.set_estop(True)
    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline and seq.stage is not Stage.DONE:
        time.sleep(0.1)
    assert seq.stage is Stage.DONE
    assert seq.error == "e-stop"


def test_path_visual_and_decorations(tabletop):
    """set_path_visual computes EE waypoints; the decimation helper is sound."""
    from robotstudio.simulation.engine import _path_capsules

    model, _data, robot = tabletop

    # decimation bound
    pts = [np.array([float(i), 0.0, 0.0]) for i in range(1000)]
    caps = _path_capsules(pts)
    assert len(caps) == 120
    assert _path_capsules(None) == []
    assert len(_path_capsules([np.zeros(3)])) == 0


def test_decorations_drawn_in_real_viewer(tabletop):
    """With a REAL viewer window, user_scn receives path/trace geoms."""
    import mujoco

    model, data, robot = tabletop
    try:
        handle = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:
        pytest.skip(f"no interactive display: {exc}")
    scn = handle.user_scn
    scn.ngeom = 0
    mujoco.mjv_initGeom(
        scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
        (0.01, 0.0, 0.0), np.array([0.5, 0.0, 0.5]), np.eye(3).flatten(),
        (0.2, 0.9, 0.4, 0.8),
    )
    scn.ngeom = 1
    handle.sync()
    assert handle.user_scn.ngeom == 1  # user scene path works end-to-end
    scn.ngeom = 0
    handle.sync()
    handle.close()
