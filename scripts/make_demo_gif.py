"""Generate the README demo GIF (real simulation frames + stage captions).

Runs the one-click showcase headless with offscreen MuJoCo rendering, draws
the live FSM stage / sim time as a caption, and writes docs/img/demo.gif.
Capturing starts at the first motion (no static opening frames).
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from robotstudio.control.showcase import ShowcaseSequencer, Stage
from robotstudio.simulation.engine import SimulationEngine

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


def main() -> None:
    eng = SimulationEngine(scene_name="tabletop", headless=True)
    eng.start(with_viewer=False)
    eng.set_speed(2.0)

    model = eng.model
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 600
    renderer = mujoco.Renderer(model, height=600, width=960)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.40, -0.05, 0.40)
    cam.distance = 1.75
    cam.azimuth = -55
    cam.elevation = -22

    seq = ShowcaseSequencer(eng)
    eng.set_controller(seq)

    frames: list[tuple[np.ndarray, str]] = []
    t0 = time.perf_counter()
    last_capture = 0.0
    started = False
    while time.perf_counter() - t0 < 200.0:
        if seq.stage is Stage.DONE:
            break
        if not started and seq.stage is not Stage.SWITCH_TO_OBSTACLE:
            started = True  # first motion phase reached
        if started and time.perf_counter() - last_capture >= 0.22:
            last_capture = time.perf_counter()
            renderer.update_scene(eng.data, camera=cam)
            frames.append((renderer.render(), seq.status_text))
        time.sleep(0.02)
    for _ in range(10):  # hold the final state
        renderer.update_scene(eng.data, camera=cam)
        frames.append((renderer.render(), seq.status_text))
    eng.stop()
    renderer.close()

    out = IMG / "demo.gif"
    imgs = []
    for f, caption in frames:
        img = Image.fromarray(f).convert("RGB")
        draw = ImageDraw.Draw(img)
        bar = ImageDraw.Draw(img)
        bar.rectangle([0, 0, 960, 34], fill=(20, 24, 30))
        draw.text((14, 9), caption[:110], fill=(126, 212, 145))
        imgs.append(img)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=80, loop=0, optimize=True)
    print(f"demo gif written: {out} ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
