@echo off
setlocal EnableDelayedExpansion

set "ENV_YML=%~dp0environment.yml"
set "CONDA_PATH=%USERPROFILE%\miniforge3\scripts\conda.exe"

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
rem Delete environment
rem ------------------------
echo Deleting conda environment...
call "%CONDA_PATH%" env remove -y -q -n %ENV_NAME% || (
    call :REPORT_ERROR "Failed to remove the environment '%ENV_NAME%'. Please check your installation and environment setup."
)

call :PLAY_NOTIFY
echo Environment removed successfully.
echo Please delete the Miniforge installation manually if desired.
pause

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