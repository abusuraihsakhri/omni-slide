@echo off
REM JP2 to TIFF Converter Pro - CLI Launcher
cd /d "%~dp0"
python -m jp2_tiff_converter %*
