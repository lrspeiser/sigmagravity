param([switch]$Force)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDirectory = Join-Path $projectRoot "data/raw/r1_ppxf_templates"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$templates = @(
    [ordered]@{
        family = "E-MILES"
        filename = "spectra_emiles_9.0.npz"
        source_url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/spectra_emiles_9.0.npz"
        usage = "Baseline SPS template grid for the residual-blind BCG reconstruction"
    },
    [ordered]@{
        family = "XSL"
        filename = "spectra_xsl_9.0.npz"
        source_url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/spectra_xsl_9.0.npz"
        usage = "Predeclared independent SPS-template systematic for RX J2129 covariance"
    }
)

$products = @()
foreach ($template in $templates) {
    $outputPath = Join-Path $outputDirectory $template.filename
    $partialPath = "$outputPath.partial"
    if ((Test-Path -LiteralPath $outputPath) -and -not $Force) {
        Write-Host "Already present: $outputPath"
    }
    else {
        if (Test-Path -LiteralPath $partialPath) {
            Remove-Item -LiteralPath $partialPath -Force
        }
        & curl.exe --fail --location --retry 4 --retry-delay 10 --output $partialPath $template.source_url
        if ($LASTEXITCODE -ne 0) {
            throw "Template download failed: $($template.family)"
        }
        Move-Item -LiteralPath $partialPath -Destination $outputPath -Force
    }

    $item = Get-Item -LiteralPath $outputPath
    $hash = Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
    $products += [ordered]@{
        family = $template.family
        source_url = $template.source_url
        local_path = "data/raw/r1_ppxf_templates/$($template.filename)"
        size_bytes = $item.Length
        sha256 = $hash.Hash
        usage = $template.usage
    }
    Write-Host "Template ready: $outputPath ($($item.Length) bytes; SHA256 $($hash.Hash))"
}

$provenance = [ordered]@{
    provenance_version = "R1B0-pPXF-templates-0.2"
    generated_utc = (Get-Date).ToUniversalTime().ToString("o")
    repository = "https://github.com/micappe/ppxf_data"
    products = $products
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$json = ($provenance | ConvertTo-Json -Depth 6) + [Environment]::NewLine
[System.IO.File]::WriteAllText((Join-Path $outputDirectory "provenance.json"), $json, $utf8NoBom)
