@echo off
cd /d "%~dp0"

echo Running PurePlay...
pixi run --manifest-path ..\pixi.toml cli-app || (
    call :REPORT_ERROR "An error occurred in the application."
)

exit /b 0

rem Functions
:REPORT_ERROR
:: %~1 is the error message passed to the function
powershell -c "(New-Object Media.SoundPlayer 'C:\Windows\Media\Windows Error.wav').PlaySync();"
echo.
echo [ERROR] %~1
pause
exit /b 1