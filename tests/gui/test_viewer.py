"""Viewer regression tests.

Regression: `mujoco.viewer` is a lazy submodule - `import mujoco` alone does
NOT expose it, and using `mujoco.viewer.launch_passive` without an explicit
`import mujoco.viewer` raises AttributeError (caught by our graceful
headless fallback, which silently degraded the app). These tests pin the
import contract and the real launch path.
"""

from __future__ import annotations

import pytest


def test_mujoco_viewer_submodule_importable():
    """The lazy-submodule trap: importing mujoco alone must not be relied on."""
    import mujoco

    assert hasattr(mujoco, "viewer"), (
        "mujoco.viewer missing after 'import mujoco' - code must explicitly "
        "'import mujoco.viewer' (see robotstudio/simulation/viewer.py)"
    )


def test_viewer_module_imports_cleanly():
    from robotstudio.simulation import viewer as viewer_mod

    assert hasattr(viewer_mod, "ViewerHandle")


def test_real_viewer_launch_and_close(tabletop):
    """Launch the ACTUAL passive viewer window and close it.

    Opt-in via PANDAFORGE_REAL_VIEWER=1: on CI runners without an
    interactive desktop, missing GL crashes the process NATIVELY (segfault
    before the Python exception layer), which a try/except cannot convert
    to a skip. Run locally to exercise the real path.
    """
    import os

    if os.environ.get("PANDAFORGE_REAL_VIEWER") != "1":
        pytest.skip("real-viewer test is opt-in (PANDAFORGE_REAL_VIEWER=1)")
    import mujoco

    model, data, _robot = tabletop
    try:
        handle = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:  # display/GL unavailable in this environment
        pytest.skip(f"no interactive GL display available here: {exc}")
    assert handle.is_running()
    handle.sync()
    handle.close()


def test_viewer_handle_wrapper_launches(tabletop):
    """The RobotStudio wrapper launches the real window (show=True)."""

    from robotstudio.simulation.viewer import ViewerHandle

    model, data, _robot = tabletop
    handle = ViewerHandle(model, data, show=True)
    if not handle.show:
        pytest.skip("viewer degraded to headless (no interactive display)")
    assert handle.is_running
    handle.sync()
    handle.close()
    assert handle._viewer is None
