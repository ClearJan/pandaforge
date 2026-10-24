@echo off
REM ============================================================
REM RobotStudio - run the automated test suite
REM ============================================================
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [test] virtual environment missing. Run setup.cmd first.
    exit /b 1
)
".venv\Scripts\python.exe" -m pytest tests %*
endlocal
