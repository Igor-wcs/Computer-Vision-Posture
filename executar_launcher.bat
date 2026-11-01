@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0launcher.py"
) else (
    py -3.12 "%~dp0launcher.py"
)

if errorlevel 1 pause
endlocal
