param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$destination = Join-Path $project "data\raw\manga_bcg_tian2024"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$url = "https://arxiv.org/e-print/2402.12016"
$archive = Join-Path $destination "arxiv_2402.12016_source.tar"
$tex = Join-Path $destination "RAR_BCG.tex"
$expectedBytes = 418438
$expectedSha256 = "04b7b619484eb4eaad7e168ced68253e788baa3121df76a77dad0c1c876c457c"

$needsDownload = $Force -or -not (Test-Path -LiteralPath $archive)
if (-not $needsDownload) {
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    $needsDownload = $actualHash -ne $expectedSha256
}

if ($needsDownload) {
    $partial = "$archive.partial"
    curl.exe -L --fail --silent --show-error --output $partial $url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $url"
    }
    $actualBytes = (Get-Item -LiteralPath $partial).Length
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $partial).Hash.ToLowerInvariant()
    if ($actualBytes -ne $expectedBytes -or $actualHash -ne $expectedSha256) {
        Remove-Item -LiteralPath $partial -Force
        throw "Unexpected arXiv source archive (bytes=$actualBytes, sha256=$actualHash)"
    }
    Move-Item -LiteralPath $partial -Destination $archive -Force
}

tar -xf $archive -C $destination RAR_BCG.tex
if ($LASTEXITCODE -ne 0) {
    throw "Could not extract RAR_BCG.tex from $archive"
}

$manifest = [ordered]@{
    dataset = "Tian et al. 2024 MaNGA brightest-cluster-galaxy acceleration table"
    downloaded_utc = (Get-Date).ToUniversalTime().ToString("o")
    citation = "Tian et al., Astronomy & Astrophysics 684, A180 (2024)"
    citation_doi = "10.1051/0004-6361/202347868"
    source_url = $url
    arxiv_id = "2402.12016"
    files = @(
        [ordered]@{
            path = "arxiv_2402.12016_source.tar"
            bytes = (Get-Item -LiteralPath $archive).Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
        },
        [ordered]@{
            path = "RAR_BCG.tex"
            bytes = (Get-Item -LiteralPath $tex).Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $tex).Hash.ToLowerInvariant()
        }
    )
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination "provenance.json") -Encoding utf8
Write-Output "MaNGA BCG source is ready at $destination"
