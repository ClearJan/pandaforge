"""Shared fixtures: a loaded tabletop scene + shadow data for planning threads."""

from __future__ import annotations

import mujoco
import pytest

from robotstudio.collision.checker import CollisionChecker
from robotstudio.kinematics.ik import IKSolver
from robotstudio.robots.panda import PandaRobot
from robotstudio.simulation.scenes import preset_scenes, write_scene_file


def build_scene(name: str = "tabletop"):
    """Load a preset scene and return (model, data, robot). Data starts at home."""
    path = write_scene_file(preset_scenes()[name])
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    robot = PandaRobot(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, robot


@pytest.fixture(scope="session")
def tabletop():
    return build_scene("tabletop")


@pytest.fixture(scope="session")
def single_obs():
    return build_scene("single_obs")


@pytest.fixture(scope="session")
def narrow():
    return build_scene("narrow")


@pytest.fixture(scope="session")
def empty():
    return build_scene("empty")


@pytest.fixture()
def checker(tabletop):
    model, _data, robot = tabletop
    return CollisionChecker(model, robot)


@pytest.fixture()
def ik(tabletop):
    model, _data, robot = tabletop
    return IKSolver(model, mujoco.MjData(model), robot, seed=7)
