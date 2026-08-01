$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$destination = Join-Path $projectRoot "data/raw/r1_j0946_primary_sources"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$sources = @(
    @{ Id = "2401.08771"; Name = "turner2024_source.tar" },
    @{ Id = "2004.00649"; Name = "collett_smith2020_source.tar" },
    @{ Id = "2104.12790"; Name = "smith_collett2021_source.tar" },
    @{ Id = "2309.04535"; Name = "ballard2024_source.tar" }
)

$records = @()
foreach ($source in $sources) {
    $url = "https://export.arxiv.org/e-print/$($source.Id)"
    $path = Join-Path $destination $source.Name
    if (-not (Test-Path -LiteralPath $path)) {
        & curl.exe --fail --location --silent --show-error --output $path $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" }
    }

    $extractPath = Join-Path $destination $source.Id
    New-Item -ItemType Directory -Force -Path $extractPath | Out-Null
    if (-not (Get-ChildItem -LiteralPath $extractPath -Force | Select-Object -First 1)) {
        & tar.exe -xf $path -C $extractPath
        if ($LASTEXITCODE -ne 0) { throw "Source extraction failed: $path" }
    }

    $item = Get-Item -LiteralPath $path
    $fileCount = (Get-ChildItem -LiteralPath $extractPath -File -Recurse).Count
    $records += [ordered]@{
        arxiv_id = $source.Id
        url = $url
        archive_path = $source.Name
        extract_path = $source.Id
        archive_bytes = $item.Length
        archive_sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        extracted_file_count = $fileCount
    }
}

$provenance = [ordered]@{
    generated_utc = [DateTime]::UtcNow.ToString("o")
    purpose = "Residual-blind primary-source audit for SDSS J0946+1006 geometry, dynamics support, archive identifiers, and observable-product availability"
    science_pixels_downloaded = $false
    files = $records
}
$provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination "provenance.json") -Encoding utf8
$records | Format-Table -AutoSize
