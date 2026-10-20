@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the Python environment first using the setup commands in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats=false
if errorlevel 1 pause
