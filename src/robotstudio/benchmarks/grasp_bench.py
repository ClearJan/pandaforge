"""Grasp benchmark: full physical pick&place trials.

Trials (protocol):
- ``base``      : fixed cube at (0.55, 0), fixed place (0.55, -0.30) - 20 runs.
- ``jitter``    : cube position randomized within +/-3 cm, place fixed - 10 runs.
- ``hard_place``: place targets at the workspace edge - 5 runs.
- ``impossible``: out-of-reach place target - expected to FAIL - 3 runs
                  (counted separately; a success here would be suspicious).
Each trial runs the live engine headless at 3x realtime. Success requires the
FSM verdict AND the independent physical post-check (cube at target, resting,
gripper open, moved > 15 cm).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from robotstudio.config import BENCH_ROOT
from robotstudio.manipulation.fsm import GraspFSM, GraspState, GraspTaskConfig
from robotstudio.simulation.engine import SimulationEngine
from robotstudio.utils.logging import get_logger

log = get_logger("bench.grasp")


def _physical_success(snapshot, place_xy, cube_start, tol=0.035) -> tuple[bool, str]:
    cube = snapshot.cube_pos
    xy_err = float(np.linalg.norm(cube[:2] - np.asarray(place_xy)))
    on_table = 0.39 <= cube[2] <= 0.46
    moved = float(np.linalg.norm(cube - cube_start)) > 0.15
    open_gripper = float(np.mean(snapshot.qpos[7:9])) > 0.035
    if xy_err <= tol and on_table and moved and open_gripper:
        return True, f"xy_err={xy_err * 1000:.0f}mm"
    return False, f"xy_err={xy_err * 1000:.0f}mm on_table={on_table} moved={moved} open={open_gripper}"


def run_grasp_benchmark(base_runs: int = 20, jitter_runs: int = 10, hard_runs: int = 5, impossible_runs: int = 3, speed: float = 3.0, seed: int = 2028) -> dict:
    rows = []
    eng = SimulationEngine(scene_name="tabletop", headless=True)
    eng.start(with_viewer=False)
    eng.set_speed(speed)
    surface_z = 0.40

    def one_trial(kind: str, cube_xy: tuple[float, float], place_xy: tuple[float, float], expect_fail: bool) -> dict:
        eng.apply_scene_spec(
            __import__("robotstudio.simulation.scenes", fromlist=["preset_scenes"]).preset_scenes()["tabletop"]
            if cube_xy == (0.55, 0.0)
            else _scene_with_cube(cube_xy)
        )
        eng.reset()
        time.sleep(0.3)  # let the reset settle
        snap0 = eng.snapshot()
        cube_start = snap0.cube_pos.copy()
        spec = GraspTaskConfig(
            cube_start=tuple(cube_start),
            cube_half=0.025,
            place_target=(place_xy[0], place_xy[1], 0.0),
            surface_z=surface_z,
            seed=seed,
        )
        fsm = GraspFSM(eng, spec)
        eng.set_controller(fsm)
        fsm.start()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 300.0 and not fsm.is_terminal:
            time.sleep(0.05)
        wall = time.perf_counter() - t0
        final = eng.snapshot()
        phys_ok, phys_msg = _physical_success(final, place_xy, cube_start)
        verdict = fsm.state == GraspState.SUCCESS and phys_ok
        if expect_fail:
            verdict = fsm.state in (GraspState.FAILED, GraspState.TIMEOUT) and not phys_ok
        row = {
            "kind": kind,
            "cube_x": cube_xy[0],
            "cube_y": cube_xy[1],
            "place_x": place_xy[0],
            "place_y": place_xy[1],
            "fsm_state": fsm.state.value,
            "physical_check": phys_msg,
            "success": bool(verdict),
            "wall_time_s": wall,
            "sim_time_s": final.sim_time,
            "error": fsm.error or "",
        }
        rows.append(row)
        log.info("%s trial %s -> %s", kind, (cube_xy, place_xy), verdict)
        # detach controller so the next trial starts clean
        eng.set_controller(None)
        eng.set_estop(False)
        return row

    rng = np.random.default_rng(seed)
    for i in range(base_runs):
        one_trial("base", (0.55, 0.0), (0.55, -0.30), expect_fail=False)
    for i in range(jitter_runs):
        cube = (0.55 + rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03))
        one_trial("jitter", cube, (0.55, -0.30), expect_fail=False)
    for i in range(hard_runs):
        # near the table edges (table spans x 0.20..0.90, y -0.45..0.45),
        # kept on-table and inside the practical workspace
        x = rng.uniform(0.24, 0.86)
        y = rng.choice([-1.0, 1.0]) * rng.uniform(0.30, 0.42)
        one_trial("hard_place", (0.55, 0.0), (x, y), expect_fail=False)
    for _ in range(impossible_runs):
        one_trial("impossible", (0.55, 0.0), (1.25, 0.95), expect_fail=True)

    eng.stop()

    out = BENCH_ROOT / "grasp_trials.csv"
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary: dict = {}
    for kind in ("base", "jitter", "hard_place", "impossible"):
        rs = [r for r in rows if r["kind"] == kind]
        if rs:
            summary[kind] = {
                "n": len(rs),
                "success_rate": sum(1 for r in rs if r["success"]) / len(rs),
                "median_wall_s": float(np.median([r["wall_time_s"] for r in rs])),
            }
    with open(BENCH_ROOT / "grasp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("grasp benchmark done: %s", json.dumps(summary))
    return summary


def _scene_with_cube(cube_xy):
    from robotstudio.simulation.scenes import preset_scenes

    spec = preset_scenes()["tabletop"]
    spec.cube_pos = (cube_xy[0], cube_xy[1], 0.425)
    return spec


if __name__ == "__main__":
    print(json.dumps(run_grasp_benchmark(), indent=2))
