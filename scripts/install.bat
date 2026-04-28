@echo off
cd /d "%~dp0"

echo Installing/updating Pixi package manager...
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex" || (
    call :REPORT_ERROR "Failed to download or install Pixi."
)

echo Installing project dependencies with Pixi...
pixi install --manifest-path ../pixi.toml || (
    call :REPORT_ERROR "Failed to install project dependencies with Pixi."
)

call :PLAY_NOTIFY
echo Setup complete.
echo Run start.bat to begin!
pause

exit /b 0

rem Functions
:REPORT_ERROR
:: %~1 is the error message passed to the function
powershell -c "(New-Object Media.SoundPlayer 'C:\Windows\Media\Windows Error.wav').PlaySync();" 2>nul
echo.
echo [ERROR] %~1
pause
exit /b 1

:PLAY_NOTIFY
powershell -c "(New-Object Media.SoundPlayer 'C:\Windows\Media\notify.wav').PlaySync();" 2>nul