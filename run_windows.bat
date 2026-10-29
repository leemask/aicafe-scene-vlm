@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.10 or newer from python.org, then run again.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8000 in your browser.
.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
pause
