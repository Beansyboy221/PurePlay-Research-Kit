@echo off
cd /d "%~dp0"

echo Removing Pixi environment...
pixi clean --manifest-path ..\pixi.toml || (
    call :REPORT_ERROR "Environment already uninstalled."
)

call :PLAY_NOTIFY
echo Pixi environment cleaned up successfully.
echo Please uninstall Pixi manually through control panel if desired.
pause

exit /b 0

rem Functions
:REPORT_ERROR
:: %~1 is the error message passed to the function
powershell -c "(New-Object Media.SoundPlayer 'C:\Windows\Media\Windows Error.wav').PlaySync();"
echo.
echo [ERROR] %~1
pause
exit /b 1

:PLAY_NOTIFY
powershell -c "(New-Object Media.SoundPlayer 'C:\Windows\Media\notify.wav').PlaySync();"