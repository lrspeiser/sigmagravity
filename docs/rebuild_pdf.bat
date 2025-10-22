@echo off
REM rebuild_pdf.bat — Quick regeneration of PDF from README.md
REM Usage: Run from root directory or double-click

cd /d "%~dp0\.."
echo Regenerating docs/sigmagravity_paper.pdf from README.md...
python scripts/md_to_latex.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Success! Opening PDF...
    start docs\sigmagravity_paper.pdf
) else (
    echo.
    echo ✗ Failed - check error messages above
    pause
)
