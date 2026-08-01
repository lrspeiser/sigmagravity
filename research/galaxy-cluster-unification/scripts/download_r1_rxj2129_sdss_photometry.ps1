param([switch]$Force)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDirectory = Join-Path $projectRoot "data/raw/r1_rxj2129_sdss"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$endpoint = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
$query = @"
SELECT TOP 10
  p.objID,p.run,p.rerun,p.camcol,p.field,p.ra,p.dec,p.type,p.parentID,p.nChild,p.flags,
  p.petroMag_u,p.petroMag_g,p.petroMag_r,p.petroMag_i,p.petroMag_z,
  p.petroMagErr_u,p.petroMagErr_g,p.petroMagErr_r,p.petroMagErr_i,p.petroMagErr_z,
  p.petroRad_r,p.petroR50_r,p.petroR90_r,p.cModelMag_r,p.deVRad_r,p.expRad_r,n.distance
FROM PhotoObjAll p
JOIN dbo.fGetNearbyObjEq(322.4165,0.08922,0.2) n ON p.objID=n.objID
WHERE p.mode=1
ORDER BY n.distance
"@ -replace "`r?`n", " "

$target = Join-Path $outputDirectory "sdss_dr18_rxj2129_bcg_photometry.csv"
if ((Test-Path -LiteralPath $target) -and -not $Force) {
    Write-Host "Already present: $target"
}
else {
    $partial = "$target.partial"
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
    & curl.exe --get --fail --location --retry 4 --retry-delay 10 `
        --data-urlencode "cmd=$query" --data-urlencode "format=csv" `
        --output $partial $endpoint
    if ($LASTEXITCODE -ne 0) {
        throw "SDSS SkyServer query failed"
    }
    Move-Item -LiteralPath $partial -Destination $target -Force
}

$item = Get-Item -LiteralPath $target
$hash = Get-FileHash -LiteralPath $target -Algorithm SHA256
$provenance = [ordered]@{
    provenance_version = "R1B1-RXJ2129-SDSS-0.1"
    generated_utc = (Get-Date).ToUniversalTime().ToString("o")
    endpoint = $endpoint
    query = $query
    query_center = [ordered]@{
        ra_deg = 322.4165
        dec_deg = 0.08922
        radius_arcmin = 0.2
    }
    selection_rule = "Retain the complete ordered 0.2-arcmin primary-object query. The BCG is the nearest row and is validated against the independent HST/Tian coordinate; do not choose a row from its photometric values."
    local_path = "data/raw/r1_rxj2129_sdss/sdss_dr18_rxj2129_bcg_photometry.csv"
    size_bytes = $item.Length
    sha256 = $hash.Hash
    aperture_definition_source = "https://classic.sdss.org/dr5/algorithms/photometry.php"
    aperture_definition = "SDSS Petrosian flux is measured in every band within 2 times the r-band Petrosian radius."
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$json = ($provenance | ConvertTo-Json -Depth 7) + [Environment]::NewLine
[System.IO.File]::WriteAllText((Join-Path $outputDirectory "provenance.json"), $json, $utf8NoBom)
Write-Host "Ready: $target ($($item.Length) bytes; SHA256 $($hash.Hash))"
