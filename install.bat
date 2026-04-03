@echo off
setlocal EnableDelayedExpansion

rem ------------------------
rem Define paths and URLs
rem ------------------------
set "ENV_YML=%~dp0environment.yml"
set "CONDA_PATH=%USERPROFILE%\miniforge3\scripts\conda.exe"
set "CONDA_BAT_PATH=%USERPROFILE%\miniforge3\condabin\conda.bat"

set "INSTALLER_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
set "INSTALLER_PATH=%~dp0\miniforge-installer.exe"

rem ------------------------
rem Extract environment name from environment.yml
rem ------------------------
if not exist "%ENV_YML%" (
    call :REPORT_ERROR "environment.yml not found."
)

for /f "tokens=2 delims=: " %%A in ('findstr /B "name:" "%ENV_YML%"') do (
    set "ENV_NAME=%%A"
)

if "%ENV_NAME%"=="" (
    call :REPORT_ERROR "Could not detect environment name from environment.yml."
)

rem ------------------------
rem Install Miniforge
rem ------------------------
if not exist "%CONDA_PATH%" (
    echo Miniforge not found. Checking for installer...
    if not exist "%INSTALLER_PATH%" (
        echo Installer not found. Downloading...
        powershell -Command "Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%INSTALLER_PATH%'" || (
            call :REPORT_ERROR "Failed to download the Miniforge installer."
        )
    )
    call :PLAY_NOTIFY
    echo Running the miniforge installer.
    echo Do not change the default options.
    echo After installation, please restart this script.
    start "" "%INSTALLER_PATH%"
    pause
    endlocal
    exit /b 0
)

rem ------------------------
rem Delete Miniforge installer
rem ------------------------
if exist "%INSTALLER_PATH%" del "%INSTALLER_PATH%"

rem ------------------------
rem Update Conda
rem ------------------------
echo Checking for Conda updates...
call "%CONDA_PATH%" update -n base -c conda-forge conda -y || (
    echo Conda update failed or already up to date. Continuing...
)

rem ------------------------
rem Create and set up the environment
rem ------------------------
echo Checking if the environment already exists...
call "%CONDA_PATH%" env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel%==0 (
    set /p OVERWRITE_ENV="Environment '%ENV_NAME%' already exists. Overwrite? (y/n): "
    if /i "!OVERWRITE_ENV!"=="y" (
        echo Deleting existing environment...
        call "%CONDA_PATH%" env remove -y -n "%ENV_NAME%" || (
            call :REPORT_ERROR "Failed to delete the existing environment '%ENV_NAME%'."
        )
        echo Creating environment from %ENV_YML%...
        call "%CONDA_PATH%" env create --file "%ENV_YML%" || (
            call :REPORT_ERROR "Failed to create the environment '%ENV_NAME%' from %ENV_YML%."
        )
    ) else (
        echo Updating existing environment from %ENV_YML%...
        call "%CONDA_PATH%" env update --file "%ENV_YML%" --prune -y || (
            call :REPORT_ERROR "Failed to update the environment '%ENV_NAME%' from %ENV_YML%."
        )
    )
) else (
    echo Creating new environment from %ENV_YML%...
    call "%CONDA_PATH%" env create --file "%ENV_YML%" || (
        call :REPORT_ERROR "Failed to create the environment '%ENV_NAME%' from %ENV_YML%."
    )
)

rem ------------------------
rem Final Activation Check
rem ------------------------
echo Activating environment to verify...
call "%CONDA_BAT_PATH%" activate %ENV_NAME% || (
    call :REPORT_ERROR "Failed to activate the environment '%ENV_NAME%'. Please check your installation."
)

call :PLAY_NOTIFY
echo Setup complete for environment: %ENV_NAME%
echo Run start.bat to begin!
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