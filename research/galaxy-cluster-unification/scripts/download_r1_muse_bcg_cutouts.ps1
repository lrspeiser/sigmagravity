param(
    [ValidateSet("MACS J1206", "Abell S1063", "all")]
    [string]$System = "MACS J1206",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot "configs/r1_dynamics_public_data_targets.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$sodaUrl = "https://dataportal.eso.org/dataPortal/soda/sync"
$selected = $config.systems | Where-Object { $System -eq "all" -or $_.system -eq $System }

foreach ($entry in $selected) {
    foreach ($product in $entry.archive.products) {
        $cutout = $product.cutout
        $outputPath = Join-Path $projectRoot $cutout.local_path
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
            $circle = "$($cutout.center_ra_deg) $($cutout.center_dec_deg) $($cutout.radius_deg)"
            $band = "$($cutout.wavelength_min_m) $($cutout.wavelength_max_m)"
            Write-Host "Requesting $($entry.system) cutout $($product.dp_id)"
            & curl.exe --fail --location --retry 4 --retry-delay 30 --max-time 1800 `
                --get --data-urlencode "ID=$datasetId" --data-urlencode "CIRCLE=$circle" `
                --data-urlencode "BAND=$band" --output $partialPath $sodaUrl
            if ($LASTEXITCODE -ne 0) {
                throw "ESO SODA download failed for $($product.dp_id)"
            }
            Move-Item -LiteralPath $partialPath -Destination $outputPath -Force
            $hash = Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
            Write-Host "Downloaded $outputPath ($((Get-Item -LiteralPath $outputPath).Length) bytes; SHA256 $($hash.Hash))"
        }
    }
}

$records = @()
foreach ($entry in $config.systems) {
    foreach ($product in $entry.archive.products) {
        $cutout = $product.cutout
        $outputPath = Join-Path $projectRoot $cutout.local_path
        if (-not (Test-Path -LiteralPath $outputPath)) {
            continue
        }
        $item = Get-Item -LiteralPath $outputPath
        $hash = Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
        $records += [ordered]@{
            system = $entry.system
            dp_id = $product.dp_id
            source_download_url = $product.download_url
            soda_url = $sodaUrl
            circle = @($cutout.center_ra_deg, $cutout.center_dec_deg, $cutout.radius_deg)
            band_m = @($cutout.wavelength_min_m, $cutout.wavelength_max_m)
            local_path = $cutout.local_path
            size_bytes = $item.Length
            sha256 = $hash.Hash
        }
    }
}
$provenance = [ordered]@{
    provenance_version = "R1A2-MUSE-cutouts-0.1"
    generated_utc = (Get-Date).ToUniversalTime().ToString("o")
    config = "configs/r1_dynamics_public_data_targets.json"
    files = $records
}
$provenancePath = Join-Path $projectRoot "data/raw/r1_muse_bcg_cubes/provenance.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$provenanceJson = ($provenance | ConvertTo-Json -Depth 8) + [Environment]::NewLine
[System.IO.File]::WriteAllText($provenancePath, $provenanceJson, $utf8NoBom)
Write-Host "Wrote $provenancePath"
