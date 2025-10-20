# Install Pandoc + MiKTeX for proper PDF generation
# Run as Administrator if needed

Write-Host "Installing Pandoc + LaTeX for academic PDF generation" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Check if chocolatey is available
$chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue

if ($chocoInstalled) {
    Write-Host "✓ Chocolatey found. Installing Pandoc..." -ForegroundColor Green
    choco install pandoc -y
    
    Write-Host ""
    Write-Host "Installing MiKTeX (LaTeX distribution, ~200 MB download, ~1 GB installed)..." -ForegroundColor Yellow
    Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow
    choco install miktex -y
} else {
    Write-Host "Chocolatey not found. Manual installation required:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Install Pandoc:" -ForegroundColor Cyan
    Write-Host "   Download from: https://pandoc.org/installing.html" -ForegroundColor White
    Write-Host "   Or run: winget install --id JohnMacFarlane.Pandoc -e" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Install MiKTeX:" -ForegroundColor Cyan
    Write-Host "   Download from: https://miktex.org/download" -ForegroundColor White
    Write-Host "   Choose 'Basic MiKTeX Installer' (64-bit)" -ForegroundColor White
    Write-Host ""
}

Write-Host ""
Write-Host "After installation completes, close and reopen PowerShell, then run:" -ForegroundColor Green
Write-Host "  python scripts/make_pdf_latex.py" -ForegroundColor White
