@echo off
REM ============================================================
REM RobotStudio - one-time environment setup for Windows
REM Creates the virtual environment, installs pinned
REM dependencies, and regenerates the preset scene files.
REM ============================================================
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [setup] ERROR: Python not found on PATH. Install Python 3.11+ first.
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo [setup] ERROR: Python 3.11 or newer is required.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [setup] creating virtual environment...
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

echo [setup] upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo [setup] installing pinned dependencies from requirements-lock.txt...
".venv\Scripts\python.exe" -m pip install --timeout 60 --retries 8 -r requirements-lock.txt
if errorlevel 1 (
    echo [setup] ERROR: dependency installation failed. Check your network and retry.
    exit /b 1
)

echo [setup] installing RobotStudio in editable mode...
".venv\Scripts\python.exe" -m pip install -e . --no-deps --quiet
if errorlevel 1 exit /b 1

echo [setup] regenerating preset scene files...
".venv\Scripts\python.exe" -c "from robotstudio.simulation.scenes import ensure_preset_scene_files; print(len(ensure_preset_scene_files()), 'scenes written')"

echo [setup] running a 5-second headless self-check...
".venv\Scripts\python.exe" -c "import mujoco, numpy; from robotstudio.simulation.scenes import preset_scenes, write_scene_file; m = mujoco.MjModel.from_xml_path(str(write_scene_file(preset_scenes()['tabletop']))); d = mujoco.MjData(m); [mujoco.mj_step(m, d) for _ in range(500)]; print(f'self-check OK: nq={m.nq} nu={m.nu} time={d.time}')"
if errorlevel 1 (
    echo [setup] ERROR: self-check failed.
    exit /b 1
)

echo.
echo [setup] DONE. Run with:  run.cmd
endlocal
