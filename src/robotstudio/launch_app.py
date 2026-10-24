"""PyInstaller entry point for the packaged RobotStudio app.

- normal launch  : opens the full application
- --selftest     : headless verification of the bundle (model load + physics),
                   used by build.cmd to validate the distribution
"""

from __future__ import annotations

import sys


def main() -> int:
    if "--selftest" in sys.argv:
        # Headless acceptance of the packaged bundle: model load + physics
        # + planning + trajectory execution (validates mujoco plugins,
        # bundled assets and scipy inside the distribution).
        import time

        import numpy as np

        from robotstudio.config import PandaSpec
        from robotstudio.control.executor import TrajectoryExecutor
        from robotstudio.collision.checker import CollisionChecker
        from robotstudio.planning.base import PlannerConfig
        from robotstudio.planning.rrt import RRTConnectPlanner
        from robotstudio.simulation.engine import SimulationEngine
        from robotstudio.trajectory.timing import TrajectoryGenerator

        eng = SimulationEngine(scene_name="tabletop", headless=True)
        eng.start(with_viewer=False)
        time.sleep(1.5)
        snap = eng.snapshot()
        if not (snap.sim_time > 0.5 and snap.ee_pos is not None):
            return 2
        # plan a short free motion and execute it in physics
        checker = CollisionChecker(eng.model, eng.robot)
        start = np.array(PandaSpec.HOME_QPOS)
        goal = start + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        res = RRTConnectPlanner(checker, PlannerConfig(seed=1, max_planning_time=15.0)).plan(start, goal)
        if not res.success:
            return 3
        traj = TrajectoryGenerator().generate(res.path)
        executor = TrajectoryExecutor(traj)
        eng.set_controller(executor)
        deadline = time.time() + traj.duration + 20.0
        while time.time() < deadline and not executor.stats.finished:
            time.sleep(0.1)
        eng.stop()
        if not (executor.stats.finished and executor.stats.success):
            return 4
        return 0
    from robotstudio.app import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
