param(
    [string]$Target = (Join-Path $PSScriptRoot "..\data\raw\local_voids")
)

$ErrorActionPreference = "Stop"
$commit = "bbbc34594d92eeef32897d67d291d54eb384be6e"
$archiveUrl = "https://github.com/RosaMalandrino/LocalVoids/archive/$commit.zip"
$targetPath = [System.IO.Path]::GetFullPath($Target)

$resumeAfterInterruptedProvenance = $false
if (Test-Path -LiteralPath $targetPath) {
    $existing = Get-ChildItem -LiteralPath $targetPath -Force
    if ($existing.Count -gt 0) {
        if (Test-Path -LiteralPath (Join-Path $targetPath "provenance.json")) {
            throw "Refusing to overwrite completed target: $targetPath"
        }
        $resumeAfterInterruptedProvenance = $true
    }
} else {
    New-Item -ItemType Directory -Path $targetPath | Out-Null
}

$tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$tempPath = [System.IO.Path]::GetFullPath(
    (Join-Path $tempRoot ("void-screening-local-voids-" + [guid]::NewGuid().ToString("N")))
)
if (-not $tempPath.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Temporary path escaped the system temporary directory: $tempPath"
}
New-Item -ItemType Directory -Path $tempPath | Out-Null

try {
    $archivePath = Join-Path $tempPath "local_voids.zip"
    $extractPath = Join-Path $tempPath "extract"
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archivePath
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath
    $sourcePath = Join-Path $extractPath "LocalVoids-$commit"
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Expected archive root is missing: $sourcePath"
    }

    $selected = @(
        "README.md",
        "voids_catalog.csv",
        "Voronoi_cloud_all_voids_N64.npy",
        "truncation_vs_radius.npy",
        "VoronoiClouds"
    )
    foreach ($relative in $selected) {
        $sourceItem = Join-Path $sourcePath $relative
        $targetItem = Join-Path $targetPath $relative
        if ($resumeAfterInterruptedProvenance) {
            if (-not (Test-Path -LiteralPath $targetItem)) {
                throw "Interrupted target is missing expected item: $targetItem"
            }
        } else {
            Copy-Item -LiteralPath $sourceItem -Destination $targetPath -Recurse
        }
    }

    $records = Get-ChildItem -LiteralPath $targetPath -File -Recurse | Sort-Object FullName | ForEach-Object {
        $relativePath = $_.FullName.Substring($targetPath.TrimEnd("\").Length).TrimStart("\")
        $targetHash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($resumeAfterInterruptedProvenance) {
            $sourceFile = Join-Path $sourcePath $relativePath
            $sourceHash = (Get-FileHash -LiteralPath $sourceFile -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($targetHash -ne $sourceHash) {
                throw "Interrupted download hash mismatch: $relativePath"
            }
        }
        [ordered]@{
            path = $relativePath.Replace("\", "/")
            bytes = $_.Length
            sha256 = $targetHash
        }
    }
    $provenance = [ordered]@{
        dataset = "Malandrino et al. Bayesian catalog of 100 high-significance Local Universe voids"
        paper = "https://arxiv.org/abs/2507.06866"
        repository = "https://github.com/RosaMalandrino/LocalVoids"
        commit = $commit
        archive_url = $archiveUrl
        archive_sha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        downloaded_utc = [DateTime]::UtcNow.ToString("o")
        boundary_rule = "individual Voronoi-cloud overlap > 0.37"
        files = @($records)
    }
    $provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (
        Join-Path $targetPath "provenance.json"
    ) -Encoding utf8
    Write-Host "Downloaded $($records.Count) catalog files to $targetPath"
} finally {
    $resolvedTemp = [System.IO.Path]::GetFullPath($tempPath)
    if ($resolvedTemp.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
    }
}
