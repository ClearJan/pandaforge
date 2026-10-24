"""Viewer lifecycle wrapper around ``mujoco.viewer.launch_passive``.

The viewer window renders on its own thread; our engine thread only calls
``sync()`` every frame. Scene switching does NOT reload the model: obstacles
are toggled in-place (body pose + geom contype), so the window stays open.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robotstudio.utils.logging import get_logger

log = get_logger("viewer")


class ViewerHandle:
    """Thin wrapper; safe no-op when running headless."""

    def __init__(self, model, data, show: bool) -> None:
        self.show = show
        self._viewer = None
        if show:
            try:
                self._viewer = mujoco.viewer.launch_passive(model, data)
                log.info("passive viewer launched")
            except Exception as exc:  # no display / driver issue -> degrade to headless
                log.warning("viewer unavailable (%s); continuing headless", exc)
                self.show = False

    def sync(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    @property
    def is_running(self) -> bool:
        if self._viewer is None:
            return True  # headless: nothing to close
        return self._viewer.is_running()

    def set_camera(self, lookat, distance: float, azimuth: float, elevation: float) -> None:
        if self._viewer is None:
            return
        cam = self._viewer.cam
        cam.lookat[:] = np.asarray(lookat, dtype=float)
        cam.distance = float(distance)
        cam.azimuth = float(azimuth)
        cam.elevation = float(elevation)

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
            log.info("viewer closed")
