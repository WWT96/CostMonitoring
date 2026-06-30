@echo off
setlocal EnableExtensions
chcp 65001 >nul

cd /d "%~dp0"

set "EXE_NAME=CostMonitoringLauncher"

where pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller was not found. Installing it with the current Python...
    python -m pip install pyinstaller
    if errorlevel 1 goto :fail
)

if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "%EXE_NAME%.spec" del /f /q "%EXE_NAME%.spec"

pyinstaller --onefile --name "%EXE_NAME%" exe_launcher.py
if errorlevel 1 goto :fail

copy /y "dist\%EXE_NAME%.exe" "%EXE_NAME%.exe" >nul
if errorlevel 1 goto :fail

if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "%EXE_NAME%.spec" del /f /q "%EXE_NAME%.spec"

echo Built %EXE_NAME%.exe
exit /b 0

:fail
echo.
echo Build failed.
pause
exit /b 1
