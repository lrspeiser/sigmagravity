$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot "configs/r1_rxj2129_ppxf_protocol.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$cutout = $config.cutout
$outputPath = Join-Path $projectRoot $config.input.cube_path
$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

if (-not (Test-Path -LiteralPath $outputPath)) {
    $partialPath = "$outputPath.partial"
    if (Test-Path -LiteralPath $partialPath) {
        Remove-Item -LiteralPath $partialPath -Force
    }
    $datasetId = "ivo://eso.org/ID?$($config.input.archive_dp_id)"
    $circle = "$($cutout.center_ra_deg) $($cutout.center_dec_deg) $($cutout.radius_deg)"
    $band = "$($cutout.wavelength_min_m) $($cutout.wavelength_max_m)"
    & curl.exe --fail --location --retry 4 --retry-delay 30 --max-time 1800 `
        --get --data-urlencode "ID=$datasetId" --data-urlencode "CIRCLE=$circle" `
        --data-urlencode "BAND=$band" --output $partialPath `
        "https://dataportal.eso.org/dataPortal/soda/sync"
    if ($LASTEXITCODE -ne 0) { throw "ESO SODA download failed" }
    Move-Item -LiteralPath $partialPath -Destination $outputPath -Force
}

$item = Get-Item -LiteralPath $outputPath
$hash = Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
$record = [ordered]@{
    provenance_version = "R1A2-RXJ2129-MUSE-cutout-0.1"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    source_dataset = $config.input.archive_dp_id
    proposal_id = $config.input.archive_proposal_id
    source_url = "https://archive.eso.org/dataset/$($config.input.archive_dp_id)"
    soda_url = "https://dataportal.eso.org/dataPortal/soda/sync"
    circle = @($cutout.center_ra_deg, $cutout.center_dec_deg, $cutout.radius_deg)
    band_m = @($cutout.wavelength_min_m, $cutout.wavelength_max_m)
    local_path = $config.input.cube_path
    size_bytes = $item.Length
    sha256 = $hash.Hash
}
$provenancePath = Join-Path $outputDirectory "rxj2129_provenance.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    $provenancePath,
    (($record | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
    $utf8NoBom
)
Write-Host "Ready $outputPath ($($item.Length) bytes; SHA256 $($hash.Hash))"
