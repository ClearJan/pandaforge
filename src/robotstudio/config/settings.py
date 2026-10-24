"""Central configuration: paths, robot constants, control defaults.

Units used everywhere in this project:
- lengths: meters, angles: radians (gripper UI additionally shows 0-255 control units),
- time: seconds.
"""

from __future__ import annotations

from pathlib import Path

# Project root = repository root (two levels above this file's src/robotstudio/config).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
ASSETS_ROOT = PROJECT_ROOT / "assets"
ROBOTS_ROOT = ASSETS_ROOT / "robots"
PANDA_ROOT = ROBOTS_ROOT / "panda"
PANDA_XML = PANDA_ROOT / "panda.xml"
SCENES_ROOT = PROJECT_ROOT / "scenes"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
REPORTS_ROOT = PROJECT_ROOT / "reports"
LOGS_ROOT = PROJECT_ROOT / "logs"
BENCH_ROOT = REPORTS_ROOT / "benchmarks"


class PandaSpec:
    """Static facts about the Menagerie Franka Emika Panda MJCF model.

    Source: mujoco_menagerie @ commit f3b6013 (Apache-2.0), assets/robots/panda/.
    """

    XML_PATH = PANDA_XML
    MODEL_NAME = "Franka Emika Panda (mujoco_menagerie)"
    N_ARM_JOINTS = 7
    N_FINGER_JOINTS = 2
    N_QPOS = 9  # 7 hinge + 2 slide
    N_CTRL = 8  # 7 arm actuators + 1 gripper tendon actuator
    EE_BODY = "hand"  # end-effector body used for FK/Jacobian/pose
    # Joint order in qpos/ctrl for the arm (indices 0..6).
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    FINGER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]
    # Home keyframe from panda.xml (arm radians + finger joint position, m).
    HOME_QPOS = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
    FINGER_OPEN = 0.04  # meters per finger; total opening 0.08 m
    FINGER_CLOSED = 0.0
    # Gripper actuator ctrl is tendon gap mapped: ctrl 255 -> 0.04 m per finger.
    GRIPPER_CTRL_OPEN = 255.0
    GRIPPER_CTRL_CLOSED = 0.0
    GRIPPER_CTRL_TO_M = 0.04 / 255.0  # ctrl -> finger joint displacement
    # Conservative default dynamic limits used by the trajectory planner.
    MAX_JOINT_VEL = (1.0, 1.0, 1.5, 1.25, 1.5, 1.5, 1.5)  # rad/s (~50% of spec)
    MAX_JOINT_ACC = (2.5, 2.5, 3.75, 3.125, 3.75, 3.75, 3.75)  # rad/s^2
    ARMATURE = 0.1
    DAMPING = 1.0


class SimDefaults:
    TIMESTEP = 0.002  # s, matches panda.xml
    CONTROL_HZ = 50.0  # controller/trajectory streaming rate
    REALTIME_HZ = 60.0  # engine loop rate (viewer sync cadence)
    MAX_REALTIME_FACTOR = 4.0


class IKDefaults:
    POS_TOL = 0.005  # m
    ROT_TOL = 0.05  # rad (~2.9 deg)
    MAX_ITERS = 300
    DAMPING = 1e-3
    MAX_TRANSLATION_STEP = 0.1  # m per iteration
    MAX_ROTATION_STEP = 0.5  # rad per iteration

