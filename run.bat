@echo off
setlocal
cd /d "%~dp0"

echo [Skunkworks] Authenticating environment...

:: Check if local environment exists
if not exist "env\Scripts\python.exe" (
    echo [ERROR] No local environment found. Please run "setup_project.bat" first.
    pause
    exit /b
)

:: Run the app using the local python
echo [Skunkworks] Launching application...
env\Scripts\python.exe run_app.py

endlocal
