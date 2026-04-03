@echo off
chcp 65001 >nul
title Manowzab Launcher

:MENU
cls
echo.
echo ==================================================
echo       Manowzab Vision - System Launcher
echo ==================================================
echo  1. Start System Hub (manowzab.live)
echo  2. Stop Server
echo  3. Exit
echo ==================================================
echo.
set /p choice="Select an option (1-3): "

if "%choice%"=="1" goto START_SERVER
if "%choice%"=="2" goto STOP_SERVER
if "%choice%"=="3" exit
goto MENU

:START_SERVER
tasklist /V | find "Manowzab_FastAPI_Server" >nul
if %errorlevel% equ 0 (
    echo.
    echo [INFO] Server is already running! Opening browser...
    start http://manowzab.live
    timeout /t 2 >nul
    goto MENU
) else (
    echo.
    echo [INFO] Starting Server...
    start "Manowzab_FastAPI_Server" cmd /c "python run.py"
    echo [INFO] Waiting for server to initialize...
    timeout /t 5 >nul
    start http://localhost:8000/
    goto MENU
)

:STOP_SERVER
echo.
echo [INFO] Stopping Server...
taskkill /FI "WINDOWTITLE eq Manowzab_FastAPI_Server*" /T /F >nul 2>&1
echo [INFO] Server stopped successfully.
timeout /t 2 >nul
goto MENU