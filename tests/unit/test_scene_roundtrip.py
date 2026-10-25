"""Scene JSON save/load integrity.

Regression: reloaded specs compared unequal to the originals because JSON
arrays deserialize as lists while the dataclasses use tuples - any
spec == spec2 check was a false negative. from_dict now normalizes every
sequence field to tuples; these tests pin that contract for ALL presets.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from robotstudio.simulation.scenes import (
    load_scene_json,
    preset_scenes,
    save_scene_json,
    write_scene_file,
)


@pytest.mark.parametrize("name", list(preset_scenes()))
def test_scene_json_roundtrip_equal(name):
    spec = preset_scenes()[name]
    path = os.path.join(tempfile.gettempdir(), f"pandaforge_rt_{name}.json")
    save_scene_json(spec, path)
    try:
        spec2 = load_scene_json(path)
        assert spec2 == spec, f"{name}: reloaded spec differs from original"
    finally:
        os.remove(path)


@pytest.mark.parametrize("name", list(preset_scenes()))
def test_scene_json_roundtrip_still_loads_in_mujoco(name):
    """Equality is not enough - the reloaded spec must still build."""
    import mujoco

    spec2 = load_scene_json(
        _save(preset_scenes()[name])
    )
    path = write_scene_file(spec2)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nu == 8


def _save(spec) -> str:
    path = os.path.join(tempfile.gettempdir(), "pandaforge_rt_build.json")
    save_scene_json(spec, path)
    return path
