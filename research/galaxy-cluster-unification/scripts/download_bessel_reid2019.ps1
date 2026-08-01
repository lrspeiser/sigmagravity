param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $Root "data\raw\bessel_reid2019"
}
$ResolvedParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $OutputDirectory))
$ResolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
if (-not $ResolvedOutput.StartsWith($ResolvedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing an output path outside its resolved parent"
}
New-Item -ItemType Directory -Force -Path $ResolvedOutput | Out-Null

$CatalogUri = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J%2FApJ%2F885%2F131%2Ftable1&-out.all&-out.max=unlimited"
$ReadmeUri = "http://cdsarc.u-strasbg.fr/ftp/cati/J_ApJ/885/131/ReadMe"
$CatalogPath = Join-Path $ResolvedOutput "table1.tsv"
$ReadmePath = Join-Path $ResolvedOutput "ReadMe"

& curl.exe -L --fail --silent --show-error $CatalogUri --output $CatalogPath
if ($LASTEXITCODE -ne 0) {
    throw "VizieR catalog download failed with exit code $LASTEXITCODE"
}
& curl.exe -L --fail --silent --show-error $ReadmeUri --output $ReadmePath
if ($LASTEXITCODE -ne 0) {
    throw "CDS ReadMe download failed with exit code $LASTEXITCODE"
}

$DataRows = Get-Content -LiteralPath $CatalogPath | Where-Object {
    $_ -and -not $_.StartsWith("#") -and $_ -match "^\s*\d+\s"
}
if ($DataRows.Count -ne 199) {
    throw "Expected 199 maser rows but found $($DataRows.Count)"
}

$CatalogHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CatalogPath).Hash.ToLowerInvariant()
$ReadmeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ReadmePath).Hash.ToLowerInvariant()
$Provenance = [ordered]@{
    catalog = "VizieR J/ApJ/885/131/table1"
    article = "Reid et al. 2019, ApJ 885, 131"
    article_doi = "10.3847/1538-4357/ab4a11"
    vizier_doi = "10.26093/cds/vizier.18850131"
    retrieved_utc = [DateTime]::UtcNow.ToString("o")
    rows = $DataRows.Count
    catalog_uri = $CatalogUri
    readme_uri = $ReadmeUri
    table1_sha256 = $CatalogHash
    readme_sha256 = $ReadmeHash
}
$Provenance | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ResolvedOutput "provenance.json") -Encoding utf8
Write-Output "Downloaded $($DataRows.Count) BeSSeL/VERA maser sources to $ResolvedOutput"
