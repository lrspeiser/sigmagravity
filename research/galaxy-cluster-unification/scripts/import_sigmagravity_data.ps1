param(
    [string]$SourceRoot = "C:\Users\henry\Documents\Codex\2026-07-18\sigmagravity-frontiers-main"
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path -LiteralPath $SourceRoot).Path
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$destination = Join-Path $project "data\raw\sparc"
$rotmodDestination = Join-Path $destination "rotmod"

New-Item -ItemType Directory -Force -Path $rotmodDestination | Out-Null

$rotmodSource = Join-Path $source "data\Rotmod_LTG"
$tableSource = Join-Path $source "data\sparc\Table1_SPARC.dat"
$readmeSource = Join-Path $source "data\sparc\ReadMe_SPARC.txt"
$coordinateSource = Join-Path $source "resources\photometry\sparc_coordinates.csv"

if (-not (Test-Path -LiteralPath $rotmodSource)) {
    throw "Missing SPARC mass-model directory: $rotmodSource"
}

$sourceFiles = @()
$sourceFiles += Get-ChildItem -LiteralPath $rotmodSource -File -Filter "*_rotmod.dat" | Sort-Object Name
$sourceFiles += Get-Item -LiteralPath $tableSource
$sourceFiles += Get-Item -LiteralPath $readmeSource
$sourceFiles += Get-Item -LiteralPath $coordinateSource

foreach ($file in $sourceFiles) {
    if ($file.DirectoryName -eq $rotmodSource) {
        $target = Join-Path $rotmodDestination $file.Name
    } elseif ($file.FullName -eq $tableSource) {
        $target = Join-Path $destination "table1.dat"
    } elseif ($file.FullName -eq $readmeSource) {
        $target = Join-Path $destination "SPARC_ReadMe.txt"
    } else {
        $target = Join-Path $destination "coordinates.csv"
    }
    Copy-Item -LiteralPath $file.FullName -Destination $target -Force
}

$commit = git -C $source rev-parse HEAD 2>$null
$records = Get-ChildItem -LiteralPath $destination -Recurse -File |
    Where-Object { $_.Name -ne "provenance.json" } |
    Sort-Object FullName |
    ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($destination.Length + 1).Replace("\", "/")
            bytes = $_.Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        }
    }

$manifest = [ordered]@{
    dataset = "SPARC mass models and metadata"
    citation = "Lelli, McGaugh & Schombert, AJ 152, 157 (2016)"
    imported_utc = (Get-Date).ToUniversalTime().ToString("o")
    source_checkout = $source
    source_git_commit = "$commit".Trim()
    source_is_read_only = $true
    rotmod_file_count = @($records | Where-Object { $_.path -like "rotmod/*_rotmod.dat" }).Count
    files = @($records)
}

$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $destination "provenance.json") -Encoding utf8
Write-Output "Imported $($manifest.rotmod_file_count) SPARC curves into $destination"

