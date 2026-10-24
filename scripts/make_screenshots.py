"""Generate real demo screenshots for the README (no mock imagery).

1. mujoco offscreen renders of the tabletop scene (home + grasp approach)
2. an offscreen Qt grab of the actual main window
Outputs into docs/img/.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from robotstudio.config import PandaSpec
from robotstudio.simulation.scenes import preset_scenes, write_scene_file

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def render_scene(out: Path, cam_pos=(1.35, -1.35, 1.15), lookat=(0.35, -0.05, 0.35)) -> None:
    path = write_scene_file(preset_scenes()["tabletop"])
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(300):
        mujoco.mj_step(model, data)
    model.vis.global_.offwidth = 1400
    model.vis.global_.offheight = 900
    renderer = mujoco.Renderer(model, height=900, width=1400)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = 1.6
    cam.azimuth = -50
    cam.elevation = -25
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    from PIL import Image

    Image.fromarray(img).save(out)
    renderer.close()
    print("wrote", out)


def render_execution(out: Path) -> None:
    """Mid-trajectory frame: the arm executing a planned motion."""
    from robotstudio.collision.checker import CollisionChecker
    from robotstudio.planning.base import PlannerConfig
    from robotstudio.planning.rrt import RRTConnectPlanner
    from robotstudio.trajectory.timing import TrajectoryGenerator

    path = write_scene_file(preset_scenes()["single_obs"])
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    robot = __import__("robotstudio.robots", fromlist=["PandaRobot"]).PandaRobot(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    checker = CollisionChecker(model, robot)
    start = np.array(PandaSpec.HOME_QPOS)
    goal = start + np.array([0.3, -0.6, 0.2, -0.2, 0.1, 0.2, 0.3])
    goal = np.clip(goal, *robot.jnt_range.T)
    res = RRTConnectPlanner(checker, PlannerConfig(seed=11, max_planning_time=15)).plan(start, goal)
    assert res.success, res.message
    traj = TrajectoryGenerator().generate(res.path)
    t_mid = 0.45 * traj.duration
    q_mid = traj.sample(t_mid)
    data.qpos[robot.arm_qposadr] = q_mid
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = 1400
    model.vis.global_.offheight = 900
    renderer = mujoco.Renderer(model, height=900, width=1400)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.3, -0.1, 0.45)
    cam.distance = 1.7
    cam.azimuth = -55
    cam.elevation = -20
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    from PIL import Image

    Image.fromarray(img).save(out)
    renderer.close()
    print("wrote", out)


def grab_ui(out: Path) -> None:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from robotstudio.simulation.engine import SimulationEngine
    from robotstudio.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    eng = SimulationEngine(scene_name="tabletop", headless=True)
    eng.start(with_viewer=False)
    win = MainWindow(eng)
    win.resize(1440, 920)
    for _ in range(60):
        app.processEvents()
        time_sleep(0.02)
    win.grab().save(str(out))
    win._timer.stop()
    eng.stop()
    print("wrote", out)


def time_sleep(s):
    import time

    time.sleep(s)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    render_scene(IMG / "scene_tabletop.png")
    render_execution(IMG / "scene_planning.png")
    grab_ui(IMG / "ui_main.png")
