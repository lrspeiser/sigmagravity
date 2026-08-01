param([switch]$Force)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot "configs/r1_m1206_level2_products.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$sodaUrl = "https://dataportal.eso.org/dataPortal/soda/sync"
$records = @()

foreach ($product in $config.products) {
    $outputPath = Join-Path $projectRoot $product.local_path
    $outputDirectory = Split-Path -Parent $outputPath
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    if ((Test-Path -LiteralPath $outputPath) -and -not $Force) {
        Write-Host "Already present: $outputPath"
    }
    else {
        $partialPath = "$outputPath.partial"
        if (Test-Path -LiteralPath $partialPath) {
            Remove-Item -LiteralPath $partialPath -Force
        }
        $datasetId = "ivo://eso.org/ID?$($product.dp_id)"
        $circle = "$($config.cutout.center_ra_deg) $($config.cutout.center_dec_deg) $($config.cutout.radius_deg)"
        $band = "$($config.cutout.wavelength_min_m) $($config.cutout.wavelength_max_m)"
        Write-Host "Requesting $($product.dp_id)"
        & curl.exe --fail --location --retry 4 --retry-delay 30 --max-time 1800 `
            --get --data-urlencode "ID=$datasetId" --data-urlencode "CIRCLE=$circle" `
            --data-urlencode "BAND=$band" --output $partialPath $sodaUrl
        if ($LASTEXITCODE -ne 0) {
            throw "ESO SODA download failed for $($product.dp_id)"
        }
        Move-Item -LiteralPath $partialPath -Destination $outputPath -Force
    }
    $item = Get-Item -LiteralPath $outputPath
    $hash = Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
    $records += [ordered]@{
        dp_id = $product.dp_id
        exposure_seconds = $product.exposure_seconds
        local_path = $product.local_path
        size_bytes = $item.Length
        sha256 = $hash.Hash
    }
    Write-Host "Ready $($product.dp_id): $($item.Length) bytes"
}

$provenance = [ordered]@{
    provenance_version = "R1A2-M1206-level2-cutouts-0.1"
    generated_utc = (Get-Date).ToUniversalTime().ToString("o")
    config = "configs/r1_m1206_level2_products.json"
    soda_url = $sodaUrl
    files = $records
}
$provenancePath = Join-Path $projectRoot "data/raw/r1_muse_bcg_cubes/macs_j1206_level2/provenance.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$json = ($provenance | ConvertTo-Json -Depth 8) + [Environment]::NewLine
[System.IO.File]::WriteAllText($provenancePath, $json, $utf8NoBom)
Write-Host "Wrote $provenancePath"
