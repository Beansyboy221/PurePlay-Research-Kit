@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

rem ------------------------
rem Define paths
rem ------------------------
set "ENV_YML=%~dp0environment.yml"
set "CONDA_BAT_PATH=%USERPROFILE%\miniforge3\condabin\conda.bat"

rem ------------------------
rem Extract environment name from environment.yml
rem ------------------------
if not exist "%ENV_YML%" (
    call :REPORT_ERROR "environment.yml file not found. Please ensure it exists in the same directory as this script."
)

for /f "tokens=2 delims=: " %%A in ('findstr /B "name:" "%ENV_YML%"') do (
    set "ENV_NAME=%%A"
)

if "%ENV_NAME%"=="" (
    call :REPORT_ERROR "Could not detect environment name from environment.yml."
)

rem ------------------------
rem Activate and Run
rem ------------------------
echo Activating environment: "%ENV_NAME%"...
call "%CONDA_BAT_PATH%" activate "%ENV_NAME%" || (
    call :REPORT_ERROR "Failed to activate the environment '%ENV_NAME%'. Please check your installation."
)

echo Starting "%ENV_NAME%"...
start python source/main.py -use_gui || (
    call :REPORT_ERROR "Failed to start the application. Please check your installation and environment setup."
)

endlocal
exit /b 0

rem ------------------------
rem Functions
rem ------------------------
:REPORT_ERROR
:: %~1 is the error message passed to the function
powershell -c "(New-Object Media.SoundPlayer 'C:\Windows\Media\Windows Error.wav').PlaySync();"
echo.
echo [ERROR] %~1
pause
exit /b 1

:PLAY_NOTIFY
powershell -c "(New-Object Media.SoundPlayer 'C:\Windows\Media\notify.wav').PlaySync();"