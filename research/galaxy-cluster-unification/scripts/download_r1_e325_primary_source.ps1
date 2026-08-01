$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$destination = Join-Path $projectRoot "data/raw/r1_e325_primary_source"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$url = "https://export.arxiv.org/e-print/1806.08300"
$archive = Join-Path $destination "collett2018_source.tar"
$extractPath = Join-Path $destination "1806.08300"
if (-not (Test-Path -LiteralPath $archive)) {
    & curl.exe --fail --location --silent --show-error --output $archive $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" }
}
New-Item -ItemType Directory -Force -Path $extractPath | Out-Null
if (-not (Get-ChildItem -LiteralPath $extractPath -Force | Select-Object -First 1)) {
    & tar.exe -xf $archive -C $extractPath
    if ($LASTEXITCODE -ne 0) { throw "Source extraction failed: $archive" }
}
$item = Get-Item -LiteralPath $archive
$provenance = [ordered]@{
    generated_utc = [DateTime]::UtcNow.ToString("o")
    purpose = "Residual-blind primary-source audit for ESO 325-G004 lens geometry, accepted dynamics support, and public archive identifiers"
    science_pixels_downloaded = $false
    arxiv_id = "1806.08300"
    url = $url
    archive_path = "collett2018_source.tar"
    extract_path = "1806.08300"
    archive_bytes = $item.Length
    archive_sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    extracted_file_count = (Get-ChildItem -LiteralPath $extractPath -File -Recurse).Count
}
$provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination "provenance.json") -Encoding utf8
$provenance | ConvertTo-Json -Depth 5
