$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$destination = Join-Path $root "data\raw\li2018_rar"
$sourceDirectory = Join-Path $destination "source"
$archive = Join-Path $destination "1803.00022v2.tar.gz"
$pdf = Join-Path $destination "1803.00022v2.pdf"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
New-Item -ItemType Directory -Force -Path $sourceDirectory | Out-Null

if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri "https://arxiv.org/e-print/1803.00022v2" -OutFile $archive
}
if (-not (Test-Path -LiteralPath $pdf)) {
    Invoke-WebRequest -Uri "https://arxiv.org/pdf/1803.00022v2" -OutFile $pdf
}

tar -xzf $archive -C $sourceDirectory

$manifest = foreach ($file in Get-ChildItem -Recurse -File -LiteralPath $destination) {
    if ($file.Name -eq "provenance.json") {
        continue
    }
    [PSCustomObject]@{
        path = $file.FullName.Substring($destination.Length).TrimStart("\").Replace("\", "/")
        bytes = $file.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
    }
}

[PSCustomObject]@{
    acquired_utc = (Get-Date).ToUniversalTime().ToString("o")
    source = "Li, Lelli, McGaugh & Schombert (2018), arXiv:1803.00022v2"
    source_page = "https://arxiv.org/abs/1803.00022"
    files = @($manifest | Sort-Object path)
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $destination "provenance.json")

Write-Host "Li et al. 2018 benchmark source ready at $destination"
