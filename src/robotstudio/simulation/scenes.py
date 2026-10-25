"""Programmatic scene construction for RobotStudio.

A scene is described by a JSON-serializable :class:`SceneSpec` (obstacles,
table, grasp cube, place target). ``build_scene_xml`` renders it as MJCF and
saves it next to the Panda model (MuJoCo resolves ``include``/``meshdir``
relative to the main XML file, so the generated file must be a sibling of
``panda.xml`` - the same mechanism menagerie's own ``scene.xml`` uses).

Preset scenes:
1. ``empty``        - free motion, floor only.
2. ``tabletop``     - worktable + grasp cube + place target (grasp demo).
3. ``single_obs``   - table + one obstacle blocking a straight path.
4. ``narrow``       - multiple obstacles forming a narrow passage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from robotstudio.config import PANDA_ROOT
from robotstudio.utils.logging import get_logger

log = get_logger("scenes")

# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #


@dataclass
class BoxObstacle:
    kind: str = "box"
    name: str = "obstacle"
    size: tuple[float, float, float] = (0.1, 0.1, 0.1)  # half-extents, m
    pos: tuple[float, float, float] = (0.5, 0.0, 0.05)  # center, m
    quat: tuple[float, float, float, float] = (1, 0, 0, 0)  # wxyz
    rgba: tuple[float, float, float, float] = (0.8, 0.3, 0.2, 0.6)
    enabled: bool = True


@dataclass
class SphereObstacle:
    kind: str = "sphere"
    name: str = "obstacle"
    radius: float = 0.1
    pos: tuple[float, float, float] = (0.5, 0.0, 0.2)
    rgba: tuple[float, float, float, float] = (0.8, 0.3, 0.2, 0.6)
    enabled: bool = True


@dataclass
class CylinderObstacle:
    kind: str = "cylinder"
    name: str = "obstacle"
    radius: float = 0.05
    half_length: float = 0.15
    pos: tuple[float, float, float] = (0.5, 0.0, 0.15)
    quat: tuple[float, float, float, float] = (1, 0, 0, 0)
    rgba: tuple[float, float, float, float] = (0.8, 0.3, 0.2, 0.6)
    enabled: bool = True


@dataclass
class MovingObstacleSpec:
    """Kinematic obstacle on a deterministic sinusoidal script (mocap body).

    The pose is a pure function of simulation time, so planning against it
    stays deterministic and the executor can guard with a time horizon.
    """

    enabled: bool = False
    size: tuple[float, float, float] = (0.02, 0.12, 0.12)  # half-extents, m
    center: tuple[float, float, float] = (0.45, 0.0, 0.55)
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)  # sweep direction
    amplitude: float = 0.35  # m, sweep half-range along the axis
    period: float = 7.0  # s
    phase: float = 0.0  # rad
    rgba: tuple[float, float, float, float] = (0.9, 0.55, 0.15, 0.9)

    def pose_at(self, t: float) -> tuple[float, float, float]:
        off = self.amplitude * np.sin(2.0 * np.pi * t / self.period + self.phase)
        a = np.asarray(self.axis, dtype=float)
        a = a / max(np.linalg.norm(a), 1e-9)
        c = np.asarray(self.center, dtype=float)
        p = c + a * off
        return (float(p[0]), float(p[1]), float(p[2]))


@dataclass
class TableSpec:
    enabled: bool = True
    top_height: float = 0.40  # z of the table surface, m
    center: tuple[float, float] = (0.55, 0.0)  # x, y of table center, m
    half_extents: tuple[float, float] = (0.35, 0.45)  # x, y half extents, m
    thickness: float = 0.03  # top plate thickness, m
    rgba: tuple[float, float, float, float] = (0.45, 0.35, 0.25, 1.0)


@dataclass
class SceneSpec:
    name: str = "tabletop"
    description: str = ""
    table: TableSpec = field(default_factory=TableSpec)
    obstacles: list[BoxObstacle | SphereObstacle | CylinderObstacle] = field(default_factory=list)
    # grasp cube (free body); disabled => not instantiated
    cube_enabled: bool = True
    cube_size: float = 0.025  # half-extent, m (5 cm cube)
    cube_pos: tuple[float, float, float] = (0.55, 0.0, 0.425)  # center, m
    cube_rgba: tuple[float, float, float, float] = (0.2, 0.6, 0.9, 1.0)
    # placement goal marker (mocap target, visual only; physics uses tolerance check)
    place_enabled: bool = True
    place_pos: tuple[float, float, float] = (0.55, -0.3, 0.425)
    moving_obstacle: MovingObstacleSpec = field(default_factory=MovingObstacleSpec)
    max_cube_pos: tuple[float, float, float] | None = None  # for randomization

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> SceneSpec:
        table = TableSpec(**d.get("table", {}))
        obstacles = []
        for o in d.get("obstacles", []):
            kind = o.get("kind", "box")
            cls = {"box": BoxObstacle, "sphere": SphereObstacle, "cylinder": CylinderObstacle}[kind]
            o = dict(o)
            o.pop("kind", None)
            obstacles.append(cls(**o))
        return SceneSpec(
            name=d.get("name", "scene"),
            description=d.get("description", ""),
            table=table,
            obstacles=obstacles,
            cube_enabled=d.get("cube_enabled", True),
            cube_size=d.get("cube_size", 0.025),
            cube_pos=d.get("cube_pos", (0.55, 0.0, 0.425)),
            cube_rgba=d.get("cube_rgba", (0.2, 0.6, 0.9, 1.0)),
            place_enabled=d.get("place_enabled", True),
            place_pos=d.get("place_pos", (0.55, -0.3, 0.425)),
            moving_obstacle=MovingObstacleSpec(**d.get("moving_obstacle", {})),
            max_cube_pos=d.get("max_cube_pos"),
        )


# --------------------------------------------------------------------------- #
# XML generation
# --------------------------------------------------------------------------- #


def _fmt(values) -> str:
    if isinstance(values, (int, float)):
        return f"{values:.6g}"
    return " ".join(f"{v:.6g}" for v in values)


def all_obstacles() -> dict[str, BoxObstacle | SphereObstacle | CylinderObstacle]:
    """Union of every preset obstacle, keyed by body name (union layout)."""
    out: dict[str, BoxObstacle | SphereObstacle | CylinderObstacle] = {}
    for scene in preset_scenes().values():
        for o in scene.obstacles:
            out.setdefault(o.name, o)
    return out


def build_scene_xml(spec: SceneSpec) -> str:
    """Render a SceneSpec as a complete MJCF string (sibling-of-panda layout).

    UNION LAYOUT: every scene file contains every body (table, all preset
    obstacles, cube, place marker), ALL with collision enabled. Disabled
    bodies are parked far below the floor (pure position hiding): MuJoCo
    bakes contype=0 geoms out of the collision pipeline at compile time, so
    runtime flag flips cannot re-enable them - position hiding is the only
    switch that supports true in-place preset changes.
    """
    parts: list[str] = []
    parts.append(
        '<mujoco model="robotstudio scene">\n'
        "  <!-- Generated by RobotStudio scene builder. Robot model: Franka Emika\n"
        "       Panda from google-deepmind/mujoco_menagerie (Apache-2.0). -->\n"
        '  <include file="panda.xml"/>\n'
    )

    # Visual settings: cleaner default camera.
    parts.append(
        "  <visual>\n"
        "    <headlight diffuse=\"0.6 0.6 0.6\" ambient=\"0.3 0.3 0.3\" specular=\"0 0 0\"/>\n"
        '    <rgba haze="0.15 0.25 0.35 1"/>\n'
        '    <global azimuth="120" elevation="-20"/>\n'
        "  </visual>\n"
    )

    parts.append(
        "  <asset>\n"
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n'
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n'
        "  </asset>\n"
    )

    wb: list[str] = [
        '    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>\n',
        '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n',
    ]

    # --- table (always present; hidden when disabled) ---
    t = spec.table
    top_z = t.top_height
    plate_pos = [t.center[0], t.center[1], top_z - t.thickness / 2.0] if spec.table.enabled else [0, 0, -5.0]
    legs_pos = [t.center[0], t.center[1], (top_z - t.thickness) / 2.0] if spec.table.enabled else [0, 0, -5.0]
    wb.append(
        f'    <body name="table" pos="{_fmt(plate_pos)}">\n'
        f'      <geom name="table_top" type="box" size="{_fmt([t.half_extents[0], t.half_extents[1], t.thickness / 2.0])}" rgba="{_fmt(t.rgba)}" friction="0.8 0.02 0.001"/>\n'
        "    </body>\n"
        f'    <body name="table_legs" pos="{_fmt(legs_pos)}">\n'
    )
    lx, ly = t.half_extents
    for sx in (-1, 1):
        for sy in (-1, 1):
            wb.append(
                f'      <geom name="table_leg_{sx}_{sy}" type="box" size="0.02 0.02 {(top_z - t.thickness) / 2.0}" pos="{_fmt([sx * lx * 0.85, sy * ly * 0.85, 0])}" rgba="0.3 0.3 0.3 1"/>\n'
            )
    wb.append("    </body>\n")

    # --- obstacles: union of ALL presets; disabled ones parked + collision-off ---
    spec_obs = {o.name: o for o in spec.obstacles if o.enabled}
    for obs in all_obstacles().values():
        enabled = obs.name in spec_obs
        o = spec_obs.get(obs.name, obs)
        pos = o.pos if enabled else (0.0, 0.0, -5.0)
        if o.kind == "box":
            wb.append(
                f'    <body name="{o.name}" pos="{_fmt(pos)}" quat="{_fmt(o.quat)}">\n'
                f'      <geom name="{o.name}_geom" type="box" size="{_fmt(o.size)}" rgba="{_fmt(o.rgba)}" friction="0.8 0.02 0.001"/>\n'
                "    </body>\n"
            )
        elif o.kind == "sphere":
            wb.append(
                f'    <body name="{o.name}" pos="{_fmt(pos)}">\n'
                f'      <geom name="{o.name}_geom" type="sphere" size="{_fmt(o.radius)}" rgba="{_fmt(o.rgba)}"/>\n'
                "    </body>\n"
            )
        elif o.kind == "cylinder":
            wb.append(
                f'    <body name="{o.name}" pos="{_fmt(pos)}" quat="{_fmt(o.quat)}">\n'
                f'      <geom name="{o.name}_geom" type="cylinder" size="{_fmt([o.radius, o.half_length])}" rgba="{_fmt(o.rgba)}"/>\n'
                "    </body>\n"
            )

    # --- grasp cube (always present) ---
    cube_pos = spec.cube_pos if spec.cube_enabled else (0.0, 0.0, -5.0)
    wb.append(
        f'    <body name="cube" pos="{_fmt(cube_pos)}">\n'
        "      <freejoint name=\"cube_joint\"/>\n"
        f'      <geom name="cube_geom" type="box" size="{_fmt([spec.cube_size] * 3)}" rgba="{_fmt(spec.cube_rgba)}" mass="0.1" friction="0.9 0.02 0.002" condim="4"/>\n'
        f'      <site name="cube_site" size="0.001"/>\n'
        "    </body>\n"
    )

    # --- moving obstacle (mocap body, always present; parked when off) ---
    mo = spec.moving_obstacle
    mo_pos = mo.pose_at(0.0) if mo.enabled else (0.0, 0.0, -5.0)
    wb.append(
        f'    <body name="moving_obstacle" mocap="true" pos="{_fmt(mo_pos)}">\n'
        f'      <geom name="moving_obstacle_geom" type="box" size="{_fmt(mo.size)}" rgba="{_fmt(mo.rgba)}" friction="0.8 0.02 0.001"/>\n'
        "    </body>\n"
    )

    # --- place marker (always present) ---
    place_pos = spec.place_pos if spec.place_enabled else (0.0, 0.0, -5.0)
    wb.append(
        f'    <body name="place_target" mocap="true" pos="{_fmt(place_pos)}">\n'
        f'      <geom name="place_marker" type="cylinder" size="{_fmt([spec.cube_size * 1.35, 0.001])}" rgba="0.2 0.9 0.3 0.4" contype="0" conaffinity="0"/>\n'
        f'      <site name="place_site" size="0.002"/>\n'
        "    </body>\n"
    )

    parts.append("  <worldbody>\n" + "".join(wb) + "  </worldbody>\n")
    # NOTE: no extra keyframe here - panda.xml's "home" keyframe (index 0)
    # is inherited through the include and used by mj_resetDataKeyframe(0).
    parts.append("</mujoco>\n")
    return "".join(parts)


def scene_xml_path(name: str) -> Path:
    return PANDA_ROOT / f"rs_scene_{name}.xml"


def write_scene_file(spec: SceneSpec, force: bool = True) -> Path:
    """Materialize a scene spec as an MJCF file loadable by MuJoCo."""
    path = scene_xml_path(spec.name)
    if path.exists() and not force:
        return path
    path.write_text(build_scene_xml(spec), encoding="utf-8")
    log.info("scene written: %s", path)
    return path


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #


def preset_scenes() -> dict[str, SceneSpec]:
    """The four P0 preset scenes plus the plain tabletop variant."""
    scenes: dict[str, SceneSpec] = {
        "empty": SceneSpec(
            name="empty",
            description="Free motion: floor and robot only.",
            table=TableSpec(enabled=False),
            cube_enabled=False,
            place_enabled=False,
        ),
        "tabletop": SceneSpec(
            name="tabletop",
            description="Worktable with a graspable cube and a place target.",
        ),
        "single_obs": SceneSpec(
            name="single_obs",
            description="Worktable with one obstacle blocking the direct reach to the left workspace.",
            obstacles=[
                BoxObstacle(
                    name="wall_obstacle",
                    size=(0.20, 0.03, 0.20),
                    pos=(0.45, -0.24, 0.50),
                    rgba=(0.85, 0.35, 0.2, 0.7),
                )
            ],
        ),
        "narrow": SceneSpec(
            name="narrow",
            description="Narrow passage between two shelf obstacles over the table.",
            obstacles=[
                BoxObstacle(
                    name="shelf_left",
                    size=(0.15, 0.02, 0.25),
                    pos=(0.45, -0.18, 0.65),
                    rgba=(0.5, 0.5, 0.55, 0.8),
                ),
                BoxObstacle(
                    name="shelf_right",
                    size=(0.15, 0.02, 0.25),
                    pos=(0.45, 0.18, 0.65),
                    rgba=(0.5, 0.5, 0.55, 0.8),
                ),
                BoxObstacle(
                    name="post_front",
                    size=(0.02, 0.02, 0.30),
                    pos=(0.32, 0.0, 0.20),
                    rgba=(0.5, 0.5, 0.55, 0.8),
                ),
            ],
        ),
        "moving_wall": SceneSpec(
            name="moving_wall",
            description="A panel sweeps through the workspace on a known script - "
            "execution is guarded and replans online when the corridor closes.",
            cube_enabled=False,
            place_enabled=False,
            moving_obstacle=MovingObstacleSpec(
                enabled=True,
                size=(0.02, 0.12, 0.12),
                center=(0.45, 0.0, 0.55),
                axis=(0.0, 1.0, 0.0),
                amplitude=0.32,
                period=11.0,
                # start at the far end of the sweep: t=0 leaves the home
                # pose free, then the panel sweeps INTO the corridor during
                # execution (forcing guarded replans)
                phase=1.5708,
            ),
        ),
        "roof_grasp": SceneSpec(
            name="roof_grasp",
            description="Shelf-to-shelf transfer: a roof blocks the top-down "
            "approach, so the grasp planner falls back to a horizontal side "
            "grasp; the place target sits on a matching pedestal.",
            cube_pos=(0.45, 0.18, 0.525),
            place_pos=(0.45, -0.20, 0.425),
            obstacles=[
                BoxObstacle(
                    name="roof_obstacle",
                    size=(0.085, 0.085, 0.012),
                    pos=(0.45, 0.18, 0.665),
                    rgba=(0.85, 0.35, 0.2, 0.75),
                ),
                BoxObstacle(
                    name="pedestal",
                    size=(0.06, 0.06, 0.05),
                    pos=(0.45, 0.18, 0.45),
                    rgba=(0.5, 0.5, 0.55, 1.0),
                ),
                BoxObstacle(
                    name="place_pedestal",
                    size=(0.06, 0.06, 0.05),
                    pos=(0.45, -0.20, 0.45),
                    rgba=(0.5, 0.5, 0.55, 1.0),
                ),
            ],
        ),
    }
    return scenes


def save_scene_json(spec: SceneSpec, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    return path


def load_scene_json(path: Path) -> SceneSpec:
    return SceneSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def ensure_preset_scene_files() -> list[Path]:
    """(Re)generate all preset scene XML files. Returns written paths."""
    return [write_scene_file(spec) for spec in preset_scenes().values()]


def cube_z_on_surface(spec: SceneSpec, surface_z: float) -> float:
    """Resting z of the cube center on a surface (table top or floor)."""
    return surface_z + spec.cube_size


def cube_top_z(spec: SceneSpec) -> float:
    return spec.cube_pos[2] + spec.cube_size


def table_surface_z(spec: SceneSpec) -> float:
    return spec.table.top_height if spec.table.enabled else 0.0


def cube_spawn_pos(spec: SceneSpec) -> tuple[float, float, float]:
    """Cube spawn position clamped to rest on the supporting surface."""
    x, y, _ = spec.cube_pos
    return (x, y, cube_z_on_surface(spec, table_surface_z(spec)))


def valid_place(spec: SceneSpec, rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """A valid placement z for the place marker (on the table surface)."""
    x, y, _ = spec.place_pos
    return (x, y, cube_z_on_surface(spec, table_surface_z(spec)))
