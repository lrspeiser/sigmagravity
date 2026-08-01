$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot "configs/r1_ms2137_muse_feasibility_protocol.json"
$reportPath = Join-Path $projectRoot "results/r1_ms2137_muse_feasibility/report.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json

if (-not $report.gates.metadata_feasibility_gate_passed) {
    throw "Frozen MS2137 metadata gate did not pass; acquisition is not authorized"
}
if (-not $report.authorization.download_frozen_cutout) {
    throw "Frozen MS2137 report does not authorize cutout acquisition"
}

$cutout = $config.frozen_cutout_request
$archive = $config.archive_product
$outputPath = Join-Path $projectRoot $cutout.local_path
$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

if (-not (Test-Path -LiteralPath $outputPath)) {
    $partialPath = "$outputPath.partial"
    if (Test-Path -LiteralPath $partialPath) {
        Remove-Item -LiteralPath $partialPath -Force
    }
    $circle = "$($cutout.center_ra_deg) $($cutout.center_dec_deg) $($cutout.radius_deg)"
    $band = "$($cutout.wavelength_min_m) $($cutout.wavelength_max_m)"
    & curl.exe --fail --location --retry 4 --retry-delay 30 --max-time 3600 `
        --get --data-urlencode "ID=$($archive.publisher_id)" `
        --data-urlencode "CIRCLE=$circle" --data-urlencode "BAND=$band" `
        --output $partialPath $archive.soda_url
    if ($LASTEXITCODE -ne 0) { throw "ESO SODA download failed" }
    Move-Item -LiteralPath $partialPath -Destination $outputPath -Force
}

$item = Get-Item -LiteralPath $outputPath
$hash = Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
$record = [ordered]@{
    provenance_version = "R1B2-MS2137-MUSE-acquisition-0.1"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    source_dataset = $archive.dp_id
    proposal_id = $archive.proposal_id
    source_url = "https://archive.eso.org/dataset/$($archive.dp_id)"
    datalink_url = $archive.datalink_url
    soda_url = $archive.soda_url
    circle = @($cutout.center_ra_deg, $cutout.center_dec_deg, $cutout.radius_deg)
    band_m = @($cutout.wavelength_min_m, $cutout.wavelength_max_m)
    local_path = $cutout.local_path
    size_bytes = $item.Length
    sha256 = $hash.Hash
    pixel_arrays_inspected = $false
}
$provenancePath = Join-Path $outputDirectory "ms2137_provenance.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    $provenancePath,
    (($record | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
    $utf8NoBom
)
Write-Host "Ready $outputPath ($($item.Length) bytes; SHA256 $($hash.Hash))"
