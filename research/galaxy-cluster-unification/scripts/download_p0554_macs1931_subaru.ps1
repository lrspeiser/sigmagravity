param(
    [string]$Protocol = "configs/p0554_macs1931_subaru_acquisition_protocol.json"
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
            Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $OutFile -TimeoutSec 300
            return
        }
        catch {
            if ($attempt -eq $Attempts) { throw }
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}

$downloaded = @()
foreach ($product in $config.products) {
    $uri = [string]$config.official_source.directory + [string]$product.filename
    $path = Join-Path $outputDirectory ([string]$product.filename)
    Invoke-DownloadWithRetry -Uri $uri -OutFile $path
    $bytes = (Get-Item -LiteralPath $path).Length
    if ($bytes -ne [long]$product.expected_bytes) {
        throw "$($product.filename) has $bytes bytes; expected $($product.expected_bytes)"
    }
    $lines = (Get-Content -LiteralPath $path | Measure-Object -Line).Lines
    $downloaded += [ordered]@{
        role = [string]$product.role
        filename = [string]$product.filename
        url = $uri
        bytes = $bytes
        lines = $lines
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$provenance = [ordered]@{
    protocol = $Protocol.Replace("\", "/")
    protocol_sha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
    acquired_utc = [DateTime]::UtcNow.ToString("o")
    products = $downloaded
}
$provenancePath = Join-Path $root $config.outputs.provenance
$provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $provenancePath -Encoding utf8
$provenance | ConvertTo-Json -Depth 8
