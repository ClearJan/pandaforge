"""Application entry point.

Launches the MuJoCo viewer window (engine thread) and the PySide6 control
window as one product. `python -m robotstudio` or the `robotstudio` script.
"""

from __future__ import annotations

import signal
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    scene = "tabletop"
    if "--scene" in argv:
        i = argv.index("--scene")
        scene = argv[i + 1]
    headless_ui = "--no-viewer" in argv

    from PySide6.QtWidgets import QApplication

    from robotstudio.simulation.engine import SimulationEngine
    from robotstudio.ui.main_window import MainWindow
    from robotstudio.utils.logging import get_logger

    log = get_logger("app")
    log.info("PandaForge starting (scene=%s)", scene)

    app = QApplication(argv)
    app.setApplicationName("PandaForge")
    font = app.font()
    font.setFamily("Segoe UI")
    font.setPointSizeF(9.5)
    app.setFont(font)

    engine = SimulationEngine(scene_name=scene, headless=headless_ui)
    engine.start(with_viewer=not headless_ui)

    win = MainWindow(engine)
    win.show()

    # Ctrl-C in the terminal should quit the Qt loop instead of being ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    exit_code = app.exec()
    engine.stop()
    log.info("PandaForge exited (code=%d)", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
