@echo off
setlocal
set "AICAFE_PACKAGE=%~dp0"
powershell.exe -NoProfile -Command "& ([ScriptBlock]::Create([IO.File]::ReadAllText((Join-Path $env:AICAFE_PACKAGE 'scripts\install_samples.ps1'))))"
set "AICAFE_UPDATE_EXIT=%ERRORLEVEL%"
pause
exit /b %AICAFE_UPDATE_EXIT%
