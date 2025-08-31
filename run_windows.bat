@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

echo === Adeno Predict - Windows launcher ===

rem Select Python interpreter (prefer py launcher with 3.11)
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PYTHON=py -3.11"
) else (
  set "PYTHON=python"
)

echo [1/4] Creating virtual environment...
%PYTHON% -m venv .venv
if errorlevel 1 (
  echo Failed to create virtual environment. Ensure Python 3.11 is installed and on PATH.
  pause
  exit /b 1
)

echo [2/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [3/4] Installing project dependencies...
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

echo [4/4] Starting Streamlit app...
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
if errorlevel 1 (
  echo Streamlit failed to start. Try running manually:
  echo   .venv\Scripts\python.exe -m streamlit run streamlit_app.py
  pause
  exit /b 1
)
