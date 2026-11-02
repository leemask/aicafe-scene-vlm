@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv-gpu
if errorlevel 1 goto fail
".venv-gpu\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv-gpu\Scripts\python.exe" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto fail
".venv-gpu\Scripts\python.exe" -m pip install -r requirements.txt transformers==4.57.1
if errorlevel 1 goto fail
".venv-gpu\Scripts\python.exe" scripts\check_gpu.py
if errorlevel 1 goto fail
echo Setup complete. Run run_gpu_sample.cmd next.
pause
exit /b 0
:fail
echo Setup failed. Copy the error text above.
pause
exit /b 1
