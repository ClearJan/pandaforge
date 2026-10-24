"""GUI smoke tests (offscreen Qt platform).

These tests drive the REAL MainWindow with a REAL engine (headless viewer):
buttons are triggered programmatically, workers run, signals fire, and the
engine thread is shut down cleanly. The full pick&place is included at 3x
speed — this is the UI-level regression for the product flow.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    from robotstudio.simulation.engine import SimulationEngine
    from robotstudio.ui.main_window import MainWindow

    eng = SimulationEngine(scene_name="tabletop", headless=True)
    eng.start(with_viewer=False)
    win = MainWindow(eng)
    yield win
    # direct teardown (closeEvent would raise a modal dialog)
    win._timer.stop()
    eng.stop()
    win.deleteLater()


def _process(window, seconds=1.0):
    """Pump the Qt event loop while the sim runs."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        window.app_process_events()
        time.sleep(0.01)


def _wait(window, predicate, timeout=30.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        window.app_process_events()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_window_boots_and_refreshes(window):
    assert window.engine.is_running
    _process(window, 0.5)
    snap = window.engine.snapshot()
    assert snap.sim_time > 0
    assert "t=" in window.status_label.text()


def test_joint_command_flow(window):
    target = np.radians([10, -20, 5, -70, 0, 80, -30])
    window.joint_panel.set_joint_target(0, target[0])
    window.joint_panel._on_slider()
    ok = _wait(
        window,
        lambda: abs(window.engine.snapshot().qpos[window.robot.arm_qposadr][0] - target[0]) < 0.05,
        timeout=10,
    )
    assert ok, "joint command did not drive the arm"


def test_ik_flow_via_panel(window):

    snap = window.engine.snapshot()
    # a reachable pose: 10 cm in front of the current EE
    target_pos = snap.ee_pos + np.array([0.05, -0.05, 0.0])
    window.ik_panel.solve_requested.emit(target_pos, IKPanel_KEEP_CURRENT)
    ok = _wait(window, lambda: window.ik_panel.solution is not None or "FAILED" in window.ik_panel.result_label.text(), timeout=30)
    assert ok, window.ik_panel.result_label.text()
    assert window.ik_panel.solution is not None
    # move to solution
    sol = window.ik_panel.solution.copy()
    window.ik_panel.move_to_solution.emit(sol)
    ok = _wait(
        window,
        lambda: np.max(np.abs(window.engine.snapshot().qpos[window.robot.arm_qposadr] - sol)) < 0.05,
        timeout=10,
    )
    assert ok


IKPanel_KEEP_CURRENT = 1


def test_plan_and_execute_flow(window):
    snap = window.engine.snapshot()
    start = snap.qpos[window.robot.arm_qposadr].copy()
    goal = np.clip(start + np.radians([25, 0, 0, -10, 0, 10, 0]), *window.robot.jnt_range.T)
    window.plan_panel.plan_requested.emit("rrt_connect", 7, 15.0, goal)
    ok = _wait(window, lambda: window._plan_result is not None, timeout=30)
    assert ok and window._plan_result.success, window.plan_panel.result_label.text()
    window.plan_panel.execute_requested.emit()
    ok = _wait(
        window,
        lambda: "completed" in window.engine.snapshot().controller_status.lower()
        or "completed" in window.plan_panel.result_label.text().lower(),
        timeout=45,
    )
    assert ok, "execution did not complete"
    final = window.engine.snapshot().qpos[window.robot.arm_qposadr]
    assert np.max(np.abs(final - goal)) < 0.08


def test_scene_switch_flow(window):
    window.scene_panel.scene_requested.emit("empty")
    ok = _wait(window, lambda: window.engine.snapshot().scene_name == "empty", timeout=10)
    assert ok
    window.scene_panel.scene_requested.emit("tabletop")
    assert _wait(window, lambda: window.engine.snapshot().scene_name == "tabletop", timeout=10)


def test_full_pick_place_via_ui(window):
    """The headline product flow, driven entirely through UI signals."""
    window.grasp_panel.grasp_requested.emit((0.55, 0.0, 0.425), (0.55, -0.30, 0.425))
    ok = _wait(
        window,
        lambda: window._fsm is None,  # main window clears it on terminal state
        timeout=240,
    )
    assert ok, "grasp task did not finish"
    final = window.engine.snapshot()
    xy_err = float(np.linalg.norm(final.cube_pos[:2] - np.array([0.55, -0.30])))
    assert xy_err <= 0.035, f"cube not at target ({xy_err*1000:.0f} mm)"
