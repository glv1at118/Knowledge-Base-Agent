@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

echo Cleaning logs and output folders...

del /Q "%SCRIPT_DIR%logs\*" 2>nul
del /Q "%SCRIPT_DIR%output\*" 2>nul

echo Done.
endlocal
