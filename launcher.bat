::[Bat To Exe Converter]
::
::YAwzoRdxOk+EWAnk
::fBw5plQjdG8=
::YAwzuBVtJxjWCl3EqQJgSA==
::ZR4luwNxJguZRRnk
::Yhs/ulQjdF+5
::cxAkpRVqdFKZSDk=
::cBs/ulQjdF+5
::ZR41oxFsdFKZSDk=
::eBoioBt6dFKZSDk=
::cRo6pxp7LAbNWATEpCI=
::egkzugNsPRvcWATEpCI=
::dAsiuh18IRvcCxnZtBJQ
::cRYluBh/LU+EWAnk
::YxY4rhs+aU+JeA==
::cxY6rQJ7JhzQF1fEqQJQ
::ZQ05rAF9IBncCkqN+0xwdVs0
::ZQ05rAF9IAHYFVzEqQJQ
::eg0/rx1wNQPfEVWB+kM9LVsJDGQ=
::fBEirQZwNQPfEVWB+kM9LVsJDGQ=
::cRolqwZ3JBvQF1fEqQJQ
::dhA7uBVwLU+EWDk=
::YQ03rBFzNR3SWATElA==
::dhAmsQZ3MwfNWATElA==
::ZQ0/vhVqMQ3MEVWAtB9wSA==
::Zg8zqx1/OA3MEVWAtB9wSA==
::dhA7pRFwIByZRRnk
::Zh4grVQjdCuDJN5qH8TGz5+JszgE1Js8+nmbOzN8NBUCfq2YuxMD/VM5X2xWpgjjbLko8m3cUJSdHxjXi8CnM5VChonIZemMd+N10fsZb+1kkpAPDru256nzfJJWYuy8swjvQTxTnXHwqXF6Q0eVCaB5tlmI6a9mOl3u1g6HLtEzs7aY1OIbH7jrc7iJ
::YB416Ek+ZG8=
::
::
::978f952a14a936cc963da21a135fa983
@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "PROJECT_ROOT=%~dp0"
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"

set "APP_URL=http://127.0.0.1:8501"
set "VENV_DIR=%PROJECT_ROOT%venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQ_MARKER=%VENV_DIR%\.requirements_installed"
set "WHEEL_DIR=%PROJECT_ROOT%wheels"
set "SYSTEM_PYTHON=python"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_SERVER_HEADLESS=true"
set "LOG_DIR=%PROJECT_ROOT%logs"
set "STREAMLIT_LOG=%LOG_DIR%\streamlit-last.log"

if not exist "%PROJECT_ROOT%app.py" (
    echo app.py was not found in %PROJECT_ROOT%
    goto :fail
)

if not exist "%PROJECT_ROOT%requirements.txt" (
    echo requirements.txt was not found in %PROJECT_ROOT%
    goto :fail
)

if not exist "%PYTHON_EXE%" (
    where python >nul 2>&1
    if errorlevel 1 (
        where py >nul 2>&1
        if errorlevel 1 (
            echo Python was not found. Install Python and add it to PATH, then retry.
            goto :fail
        )
        set "SYSTEM_PYTHON=py -3"
    )

    echo Creating local virtual environment...
    %SYSTEM_PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :fail
)

if not exist "%REQ_MARKER%" (
    if exist "%WHEEL_DIR%\" (
        echo Installing Python dependencies from local wheels...
        "%PYTHON_EXE%" -m pip install --no-index --find-links "%WHEEL_DIR%" -r "%PROJECT_ROOT%requirements.txt"
        if errorlevel 1 (
            echo Local wheel installation failed. Check that the installed Python version matches the wheel files.
            goto :fail
        )
    ) else (
        echo Installing Python dependencies from the internet...
        "%PYTHON_EXE%" -m pip install --upgrade pip
        if errorlevel 1 goto :fail

        "%PYTHON_EXE%" -m pip install -r "%PROJECT_ROOT%requirements.txt"
        if errorlevel 1 goto :fail
    )

    echo installed>"%REQ_MARKER%"
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%APP_URL%/_stcore/health'; if ($r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    start "" "%APP_URL%"
    exit /b 0
)

echo Starting Streamlit server...
start "CostMonitoring Server" /min cmd /c ""%PYTHON_EXE%" -m streamlit run app.py --server.port=8501 --server.headless=true --server.address=127.0.0.1 --browser.gatherUsageStats=false > "%STREAMLIT_LOG%" 2>&1"

for /L %%I in (1,1,180) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%APP_URL%/_stcore/health'; if ($r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 (
        start "" "%APP_URL%"
        exit /b 0
    )
    timeout /t 1 /nobreak >nul
)

echo Streamlit is still starting. Opening the browser anyway.
echo If the page is not ready yet, wait a moment and refresh it.
echo Server log: %STREAMLIT_LOG%
start "" "%APP_URL%"
exit /b 0

:fail
echo.
echo Startup failed. Review the messages above and retry.
pause
exit /b 1
