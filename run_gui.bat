@echo off
REM JP2 to TIFF Converter Pro - Desktop App Launcher
cd /d "%~dp0"
python app.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
