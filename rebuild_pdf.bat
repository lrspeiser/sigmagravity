@echo off
REM rebuild_pdf.bat — Quick regeneration of PDF from README.md
REM Usage: Double-click or run from command line

echo Regenerating sigmagravity_paper.pdf from README.md...
python scripts/md_to_latex.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Success! Opening PDF...
    start sigmagravity_paper.pdf
) else (
    echo.
    echo ✗ Failed - check error messages above
    pause
)
