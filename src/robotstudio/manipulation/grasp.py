"""Grasp geometry for a box object with the Panda gripper (top-down grasp).

Conventions verified against the menagerie model:
- The ``hand`` body +z axis points along the fingers (toward the fingertips).
- Fingertip pad centres sit ~0.1025 m along hand +z from the hand origin
  (finger joint at 0.0584 + pad at ~0.044), and the fingers separate along
  hand ±y.
- A top-down grasp therefore aligns hand +z with world -z; the fingers then
  close along world ±y, gripping two opposite faces of an axis-aligned cube.

The hand-origin height is set so the fingertip pads land at the cube's
mid-height.
"""

from __future__ import annotations

import numpy as np

from robotstudio.config import PandaSpec
from robotstudio.kinematics.pose import Pose

# hand origin -> fingertip pad centre distance along hand +z (m, measured
# from the menagerie collision model: pad box centres at rel z = 0.1029).
HAND_TO_PAD = 0.1029
# Pads grip slightly ABOVE the cube mid-height: the finger/hand convex-hull
# collision meshes extend deeper than the pads, so a centred grasp makes the
# palm hull graze the cube top (measured -2.8 mm penetration). +5 mm keeps a
# verified-clear descent while still gripping the upper half of the cube.
GRASP_PAD_OFFSET = 0.005
GRASP_LIFT = HAND_TO_PAD + GRASP_PAD_OFFSET  # hand origin z offset above cube centre
PREGRASP_CLEARANCE = 0.06  # additional approach height (m)
GRASP_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)  # hand +z -> world -z (top-down)
# Squeeze depth: command the fingers well narrower than the object so the
# pads press it hard enough to lift it. Measured on this model (100 g cube):
# target opening 0.0116 m -> 1.34 N total normal force (slips), 0.0053 m ->
# 1.96 N (holds, 3.7x friction margin over the 0.98 N weight). The contact
# solver stops the fingers at the cube faces; the tendon actuator's position
# servo (stiffness 100 N/m) turns the overshoot into grip force.
SQUEEZE_DEPTH = 0.020


def grasp_pose(cube_center: np.ndarray, cube_half: float) -> Pose:
    """Hand pose with fingertip pads at the cube mid-height, top-down."""
    c = np.asarray(cube_center, dtype=float)
    return Pose.make(c + [0.0, 0.0, GRASP_LIFT], GRASP_QUAT_WXYZ)


def pregrasp_pose(cube_center: np.ndarray, cube_half: float) -> Pose:
    """Hover pose ``PREGRASP_CLEARANCE`` above the grasp pose."""
    g = grasp_pose(cube_center, cube_half)
    return Pose.make(g.pos + [0.0, 0.0, PREGRASP_CLEARANCE], g.quat_wxyz)


RELEASE_CLEARANCE = 0.010  # cube bottom hover above the surface at release (m)
# 10 mm: leaves margin for servo tracking error so the (convex-hull) finger
# meshes cannot graze the table at low edge positions; the cube free-falls
# the clearance and settles - verified stable in grasp trials.


def place_pose(place_target: np.ndarray, cube_half: float, surface_z: float) -> Pose:
    """Hand pose for releasing the cube, hovering RELEASE_CLEARANCE above it.

    The cube centre z = surface + half + clearance; on release it drops the
    clearance and settles on the surface.
    """
    t = np.asarray(place_target, dtype=float)
    cube_center_z = surface_z + cube_half + RELEASE_CLEARANCE
    return Pose.make([t[0], t[1], cube_center_z + GRASP_LIFT], GRASP_QUAT_WXYZ)


def place_pregrasp_pose(place_target: np.ndarray, cube_half: float, surface_z: float) -> Pose:
    p = place_pose(place_target, cube_half, surface_z)
    return Pose.make(p.pos + [0.0, 0.0, PREGRASP_CLEARANCE], p.quat_wxyz)


def gripper_opening_for(cube_half: float) -> float:
    """Per-finger opening (m) that squeezes the cube faces.

    Finger joint position = distance from the hand centreline; commanding
    ``cube_half - SQUEEZE_DEPTH`` makes the pads press the faces with a
    force-limited squeeze.
    """
    return float(np.clip(cube_half - SQUEEZE_DEPTH, PandaSpec.FINGER_CLOSED, PandaSpec.FINGER_OPEN))
