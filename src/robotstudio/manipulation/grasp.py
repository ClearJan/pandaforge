"""Grasp geometry for a box object with the Panda gripper.

Conventions verified against the menagerie model:
- The ``hand`` body +z axis points along the fingers (toward the fingertips).
- Fingertip pad centres sit 0.1029 m along hand +z from the hand origin, and
  the fingers separate along hand ±y.
- The grasp approach direction ``a`` (pointing INTO the cube) equals hand +z;
  a top-down grasp is a = world -z (quaternion (0,1,0,0)).

Antipodal candidate generation: for an axis-aligned cube the valid pad
oppositions are the 6 face normals. Top-down is preferred (most stable,
clears the table), then the four horizontal side grasps (fingers closing
horizontally, pads pressing opposite side faces). The bottom face is
excluded (unreachable through the table). :func:`grasp_candidates` returns
them in preference order; the FSM picks the first candidate whose pregrasp
IK + approach segment is collision-free - so a blocked top face (e.g. a low
roof obstacle) automatically falls back to a side grasp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotstudio.config import PandaSpec
from robotstudio.kinematics.pose import Pose, mat_to_quat

# hand origin -> fingertip pad centre distance along hand +z (m, measured
# from the menagerie collision model: pad box centres at rel z = 0.1029).
HAND_TO_PAD = 0.1029
# Pads grip slightly ABOVE the cube centre-line: the finger/hand convex-hull
# collision meshes extend deeper than the pads, so a centred top grasp makes
# the palm hull graze the cube top (measured -2.8 mm penetration). +5 mm
# keeps a verified-clear descent while still gripping the upper half.
GRASP_PAD_OFFSET = 0.005
GRASP_LIFT = HAND_TO_PAD + GRASP_PAD_OFFSET  # hand origin offset above cube centre
PREGRASP_CLEARANCE = 0.06  # hover distance along -a (m)
GRASP_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)  # hand +z -> world -z (top-down)
# Squeeze depth: command the fingers well narrower than the object so the
# pads press it hard enough to lift it. Measured on this model (100 g cube):
# target opening 0.0116 m -> 1.34 N total normal force (slips), 0.0053 m ->
# 1.96 N (holds, 3.7x friction margin over the 0.98 N weight).
SQUEEZE_DEPTH = 0.020

RELEASE_CLEARANCE = 0.010  # cube bottom hover above the surface at release (m)

_WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class GraspCandidate:
    """One antipodal grasp: approach axis + hand orientation + poses."""

    approach: np.ndarray  # unit vector INTO the cube (equals hand +z)
    quat_wxyz: np.ndarray
    grasp: Pose  # pads on the cube, hand +z = approach
    pregrasp: Pose  # hover PREGRASP_CLEARANCE back along the approach

    @property
    def is_top(self) -> bool:
        return abs(float(np.dot(self.approach, _WORLD_UP))) > 0.9


def _hand_quat_for(approach: np.ndarray) -> np.ndarray:
    """Hand orientation with hand +z = approach and fingers closing sensibly.

    For horizontal approaches the fingers (hand ±y) close along the world
    horizontal axis (x_h = world up -> y_h = z_h × x_h horizontal); for the
    top approach this reduces to the classic (0,1,0,0) frame (modulo the
    handedness sign of the closing axis, which is irrelevant).
    """
    a = np.asarray(approach, dtype=float)
    if abs(float(np.dot(a, _WORLD_UP))) > 0.9:
        x_h = np.array([1.0, 0.0, 0.0])
    else:
        x_h = _WORLD_UP
    z_h = a
    y_h = np.cross(z_h, x_h)
    y_h /= np.linalg.norm(y_h)
    x_h = np.cross(y_h, z_h)  # re-orthogonalize
    R = np.column_stack([x_h, y_h, z_h])
    return mat_to_quat(R)


def grasp_candidates(cube_center: np.ndarray, cube_half: float) -> list[GraspCandidate]:
    """Antipodal grasp candidates in preference order (top first, then sides)."""
    c = np.asarray(cube_center, dtype=float)
    approaches = [
        np.array([0.0, 0.0, -1.0]),  # top (preferred: most stable)
        np.array([0.0, 1.0, 0.0]),  # side: approach from -y
        np.array([0.0, -1.0, 0.0]),  # side: approach from +y
        np.array([1.0, 0.0, 0.0]),  # side: approach from -x
        np.array([-1.0, 0.0, 0.0]),  # side: approach from +x
    ]
    out = []
    for a in approaches:
        quat = _hand_quat_for(a)
        grasp_pos = c - a * GRASP_LIFT  # pads at cube centre via hand +z = a
        pregrasp_pos = grasp_pos - a * PREGRASP_CLEARANCE
        out.append(
            GraspCandidate(
                approach=a,
                quat_wxyz=quat,
                grasp=Pose.make(grasp_pos, quat),
                pregrasp=Pose.make(pregrasp_pos, quat),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# backward-compatible single-pose helpers (top-down grasp)
# --------------------------------------------------------------------------- #


def grasp_pose(cube_center: np.ndarray, cube_half: float) -> Pose:
    """Top-down grasp pose (pads at the cube's upper half)."""
    return grasp_candidates(cube_center, cube_half)[0].grasp


def pregrasp_pose(cube_center: np.ndarray, cube_half: float) -> Pose:
    """Top-down pregrasp hover."""
    return grasp_candidates(cube_center, cube_half)[0].pregrasp


def place_pose_for(
    candidate: GraspCandidate,
    place_target: np.ndarray,
    cube_half: float,
    surface_z: float,
) -> Pose:
    """Release pose for a chosen grasp candidate (same hand orientation).

    The cube centre is placed at surface + half + RELEASE_CLEARANCE; the hand
    keeps the grasp orientation so the carried cube stays axis-aligned.
    """
    t = np.asarray(place_target, dtype=float)
    cube_center = np.array([t[0], t[1], surface_z + cube_half + RELEASE_CLEARANCE])
    hand_pos = cube_center - candidate.approach * GRASP_LIFT
    return Pose.make(hand_pos, candidate.quat_wxyz)


def place_pregrasp_for(
    candidate: GraspCandidate,
    place_target: np.ndarray,
    cube_half: float,
    surface_z: float,
) -> Pose:
    p = place_pose_for(candidate, place_target, cube_half, surface_z)
    return Pose.make(p.pos - candidate.approach * PREGRASP_CLEARANCE, p.quat_wxyz)


def place_pose(place_target: np.ndarray, cube_half: float, surface_z: float) -> Pose:
    """Top-down release pose (backward-compatible helper)."""
    top = GraspCandidate(
        approach=np.array([0.0, 0.0, -1.0]),
        quat_wxyz=np.asarray(GRASP_QUAT_WXYZ, dtype=float),
        grasp=Pose.make(np.zeros(3)),
        pregrasp=Pose.make(np.zeros(3)),
    )
    return place_pose_for(top, place_target, cube_half, surface_z)


def gripper_opening_for(cube_half: float) -> float:
    """Per-finger opening (m) that squeezes the cube faces."""
    return float(np.clip(cube_half - SQUEEZE_DEPTH, PandaSpec.FINGER_CLOSED, PandaSpec.FINGER_OPEN))
