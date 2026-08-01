$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$destination = Join-Path $root "data\raw\sparc_replica"
$archiveDirectory = Join-Path $destination "archives"
New-Item -ItemType Directory -Force -Path $archiveDirectory | Out-Null

$downloads = @(
    @{
        Name = "sfb_LTG.zip"
        Url = "https://astroweb.case.edu/SPARC/sfb_LTG.zip"
        Folder = "photometric_profiles"
    },
    @{
        Name = "BulgeDiskDec_LTG.zip"
        Url = "https://astroweb.case.edu/SPARC/BulgeDiskDec_LTG.zip"
        Folder = "bulge_disk_decompositions"
    }
)

foreach ($item in $downloads) {
    $archive = Join-Path $archiveDirectory $item.Name
    $expanded = Join-Path $destination $item.Folder
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -Uri $item.Url -OutFile $archive
    }
    New-Item -ItemType Directory -Force -Path $expanded | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $expanded -Force
}

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
    source = "Official SPARC data page, Lelli, McGaugh & Schombert"
    source_page = "https://astroweb.case.edu/SPARC/"
    files = @($manifest | Sort-Object path)
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $destination "provenance.json")

Write-Host "SPARC replica inputs ready at $destination"
