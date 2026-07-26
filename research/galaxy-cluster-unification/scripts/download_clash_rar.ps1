param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$destination = Join-Path $project "data\raw\clash_tian2020"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$catalogPage = "https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJ/896/70?format=html&tex=true"
$url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/896/70/fig2.dat"
$target = Join-Path $destination "fig2.dat"
$expectedBytes = 3612
$expectedSha256 = "1286f24d91a37e6c28903166d9493b3a84c9de9f82d36700ab13e9e2b2a0f85b"

$needsDownload = $Force -or -not (Test-Path -LiteralPath $target)
if (-not $needsDownload) {
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    $needsDownload = $actualHash -ne $expectedSha256
}

if ($needsDownload) {
    $partial = "$target.partial"
    curl.exe -L --fail --silent --show-error --output $partial $url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $url"
    }
    $actualBytes = (Get-Item -LiteralPath $partial).Length
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $partial).Hash.ToLowerInvariant()
    if ($actualBytes -ne $expectedBytes) {
        Remove-Item -LiteralPath $partial -Force
        throw "Unexpected byte count: $actualBytes (expected $expectedBytes)"
    }
    if ($actualHash -ne $expectedSha256) {
        Remove-Item -LiteralPath $partial -Force
        throw "Unexpected SHA-256: $actualHash"
    }
    Move-Item -LiteralPath $partial -Destination $target -Force
}

$manifest = [ordered]@{
    dataset = "Tian et al. 2020 CLASH radial acceleration catalog"
    downloaded_utc = (Get-Date).ToUniversalTime().ToString("o")
    citation = "Tian et al., Astrophysical Journal 896, 70 (2020)"
    citation_doi = "10.3847/1538-4357/ab8db9"
    catalog_page = $catalogPage
    independent_source = $url
    local_sigmagravity_copy_used_for_expected_hash_only = $true
    records = 84
    clusters = 20
    file = [ordered]@{
        path = "fig2.dat"
        bytes = (Get-Item -LiteralPath $target).Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination "provenance.json") -Encoding utf8
Write-Output "CLASH RAR catalog is ready at $destination"
