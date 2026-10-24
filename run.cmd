@echo off
REM ============================================================
REM RobotStudio - launch the desktop application
REM ============================================================
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [run] virtual environment missing. Run setup.cmd first.
    exit /b 1
)
".venv\Scripts\python.exe" -m robotstudio %*
endlocal
