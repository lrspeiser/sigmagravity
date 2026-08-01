$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $Root "configs\p0637_little_things_photometric_metadata.json"
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$Output = Join-Path $Root $Config.raw_directory
New-Item -ItemType Directory -Path $Output -Force | Out-Null

$ReadMe = Join-Path $Output "ReadMe.txt"
if (-not (Test-Path -LiteralPath $ReadMe)) {
    Invoke-WebRequest -Uri "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/162/49/ReadMe" -OutFile $ReadMe
}

foreach ($Table in $Config.catalog.tables) {
    $Destination = Join-Path $Output "$Table.tsv"
    $Url = "$($Config.catalog.base_url)?-source=J%2FApJS%2F162%2F49%2F$Table&-out.all&-out.max=500"
    Invoke-WebRequest -Uri $Url -OutFile $Destination
    $FirstLine = Get-Content -LiteralPath $Destination -TotalCount 1
    if ($FirstLine -like "*doctype html*") {
        throw "VizieR returned HTML instead of table data for $Table"
    }
    Write-Host "${Table}: downloaded from the VizieR catalog service"
}
