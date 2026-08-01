param(
    [string]$Protocol = "configs/p0554_macs1931_wide_field_acquisition_protocol.json"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $root $Protocol
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$outputDirectory = Join-Path $root $config.outputs.directory
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

function Invoke-DownloadWithRetry {
    param([string]$Uri, [string]$OutFile, [int]$Attempts = 4)
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $OutFile -TimeoutSec 180
            return
        }
        catch {
            if ($attempt -eq $Attempts) { throw }
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}

$query = [uri]::EscapeDataString([string]$config.catalog_query)
$catalogUrl = [string]$config.official_source.tap_endpoint + "?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=" + $query
$catalogPath = Join-Path $root $config.outputs.catalog
Invoke-DownloadWithRetry -Uri $catalogUrl -OutFile $catalogPath

$field = $config.field
$cutout = $config.cutout
$cutoutUrl = [string]$config.official_source.image_cutout_documentation +
    "?ra=$($field.center_ra_deg)&dec=$($field.center_dec_deg)&layer=$($cutout.layer)" +
    "&pixscale=$($cutout.pixel_scale_arcsec)&bands=$($cutout.bands)&size=$($cutout.size_pixels)"
$jpegPath = Join-Path $root $config.outputs.jpeg
$jpegDownloaded = $true
$jpegError = $null
try {
    Invoke-DownloadWithRetry -Uri $cutoutUrl -OutFile $jpegPath
}
catch {
    if ([bool]$cutout.required) { throw }
    $jpegDownloaded = $false
    $jpegError = $_.Exception.Message
    if (Test-Path -LiteralPath $jpegPath) {
        Remove-Item -LiteralPath $jpegPath -Force
    }
}

$catalogRows = (Get-Content -LiteralPath $catalogPath | Measure-Object -Line).Lines - 1
if ($catalogRows -ne [int]$field.preflight_object_count) {
    throw "Catalog row count $catalogRows differs from frozen preflight count $($field.preflight_object_count)"
}
$catalogHash = (Get-FileHash -LiteralPath $catalogPath -Algorithm SHA256).Hash.ToLowerInvariant()
$jpegHash = if ($jpegDownloaded) { (Get-FileHash -LiteralPath $jpegPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$provenance = [ordered]@{
    protocol = $Protocol.Replace("\", "/")
    protocol_sha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
    acquired_utc = [DateTime]::UtcNow.ToString("o")
    catalog = [ordered]@{
        url = $catalogUrl
        rows = $catalogRows
        bytes = (Get-Item -LiteralPath $catalogPath).Length
        sha256 = $catalogHash
    }
    jpeg = [ordered]@{
        url = $cutoutUrl
        downloaded = $jpegDownloaded
        error = $jpegError
        bytes = if ($jpegDownloaded) { (Get-Item -LiteralPath $jpegPath).Length } else { 0 }
        sha256 = $jpegHash
    }
}
$provenancePath = Join-Path $root $config.outputs.provenance
$provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $provenancePath -Encoding utf8
$provenance | ConvertTo-Json -Depth 8
