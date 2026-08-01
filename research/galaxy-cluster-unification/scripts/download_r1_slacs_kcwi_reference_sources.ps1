$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$destination = Join-Path $projectRoot "data/raw/r1_slacs_kcwi_primary_source"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$references = @(
    [ordered]@{
        arxiv_id = "2311.09307"
        version = "v2"
        role = "Uniform Project Dinos lens-model reference for the 14-lens KCWI sample"
        url = "https://export.arxiv.org/e-print/2311.09307"
        archive_path = "dinos_source.tar"
        extract_path = "2311.09307"
    }
)

$ledger = @()
foreach ($reference in $references) {
    $archive = Join-Path $destination $reference.archive_path
    $extractPath = Join-Path $destination $reference.extract_path
    if (-not (Test-Path -LiteralPath $archive)) {
        & curl.exe --fail --location --silent --show-error --output $archive $reference.url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $($reference.url)" }
    }
    New-Item -ItemType Directory -Force -Path $extractPath | Out-Null
    if (-not (Get-ChildItem -LiteralPath $extractPath -Force | Select-Object -First 1)) {
        & tar.exe -xf $archive -C $extractPath
        if ($LASTEXITCODE -ne 0) { throw "Source extraction failed: $archive" }
    }
    $item = Get-Item -LiteralPath $archive
    $ledger += [ordered]@{
        arxiv_id = $reference.arxiv_id
        version = $reference.version
        role = $reference.role
        url = $reference.url
        archive_path = $reference.archive_path
        extract_path = $reference.extract_path
        archive_bytes = $item.Length
        archive_sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        extracted_file_count = (Get-ChildItem -LiteralPath $extractPath -File -Recurse).Count
    }
}

$provenance = [ordered]@{
    generated_utc = [DateTime]::UtcNow.ToString("o")
    purpose = "Residual-blind primary-reference audit for SLACS-KCWI lens-model reproducibility"
    science_pixels_downloaded = $false
    references = $ledger
}
$provenance | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $destination "reference_provenance.json") -Encoding utf8
$provenance | ConvertTo-Json -Depth 6
