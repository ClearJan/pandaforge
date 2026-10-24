@echo off
REM ============================================================
REM RobotStudio - build the Windows distribution package
REM Produces dist\RobotStudio\ (one-folder bundle) and verifies
REM that the packaged executable launches headless.
REM ============================================================
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [build] virtual environment missing. Run setup.cmd first.
    exit /b 1
)

echo [build] cleaning previous output...
if exist build rmdir /s /q build
if exist dist\RobotStudio rmdir /s /q dist\RobotStudio

echo [build] running PyInstaller (one-folder)...
".venv\Scripts\python.exe" -m PyInstaller ^
    --name RobotStudio ^
    --noconfirm ^
    --windowed ^
    --paths src ^
    --add-data "assets;assets" ^
    --hidden-import robotstudio.ui.main_window ^
    --hidden-import robotstudio.ui.panels ^
    --hidden-import robotstudio.ui.workers ^
    --collect-all mujoco ^
    --collect-all glfw ^
    src\robotstudio\launch_app.py
if errorlevel 1 (
    echo [build] ERROR: PyInstaller failed.
    exit /b 1
)

echo [build] verifying the packaged executable (headless self-test)...
".venv\Scripts\python.exe" -c "import subprocess, sys; r = subprocess.run([r'dist\RobotStudio\RobotStudio.exe', '--selftest'], timeout=120); sys.exit(r.returncode)"
if errorlevel 1 (
    echo [build] ERROR: packaged exe failed the self-test.
    exit /b 1
)

echo.
echo [build] DONE. Package at: dist\RobotStudio\RobotStudio.exe
echo         (launch with dist\RobotStudio\RobotStudio.exe, or dist\RobotStudio\run_packaged.cmd)
endlocal
