@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-gpu\Scripts\python.exe" goto setup
".venv-gpu\Scripts\python.exe" scripts\check_gpu.py
if errorlevel 1 goto fail
for /f %%i in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString()"') do set "RUN_ID=gpu-sample-%%i"
if not defined RUN_ID goto fail
".venv-gpu\Scripts\python.exe" -u scripts\compare_vlm.py --run-id "%RUN_ID%" --device cuda --dtype float16 --batch-size 1 --ids meva-01-04
if errorlevel 1 goto fail
".venv-gpu\Scripts\python.exe" scripts\report_vlm.py --run-id "%RUN_ID%"
if errorlevel 1 goto fail
start "" "experiments\%RUN_ID%\review.html"
echo Results: experiments\%RUN_ID%
pause
exit /b 0
:setup
echo Run setup_gpu_windows.cmd first.
:fail
echo Failed. Keep the error text and experiments run.json for diagnosis.
pause
exit /b 1
