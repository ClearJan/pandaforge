"""PyInstaller entry point for the packaged RobotStudio app.

- normal launch  : opens the full application
- --selftest     : headless verification of the bundle (model load + physics),
                   used by build.cmd to validate the distribution
"""

from __future__ import annotations

import sys


def main() -> int:
    if "--selftest" in sys.argv:
        import time

        from robotstudio.simulation.engine import SimulationEngine

        eng = SimulationEngine(scene_name="tabletop", headless=True)
        eng.start(with_viewer=False)
        time.sleep(1.5)
        snap = eng.snapshot()
        ok = snap.sim_time > 0.5 and snap.ee_pos is not None
        eng.stop()
        if not ok:
            return 2
        return 0
    from robotstudio.app import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
