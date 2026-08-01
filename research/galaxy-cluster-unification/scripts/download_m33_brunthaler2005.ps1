param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$output = Join-Path $Root "data\raw\m33_brunthaler2005"
New-Item -ItemType Directory -Force -Path $output | Out-Null

$pdfUrl = "https://arxiv.org/pdf/astro-ph/0503058"
$pdfPath = Join-Path $output "brunthaler2005_m33_science.pdf"
Invoke-WebRequest -Uri $pdfUrl -OutFile $pdfPath

$hash = (Get-FileHash -LiteralPath $pdfPath -Algorithm SHA256).Hash.ToLowerInvariant()
$bytes = (Get-Item -LiteralPath $pdfPath).Length
$retrieved = (Get-Date).ToUniversalTime().ToString("o")
$provenance = [ordered]@{
    dataset = "Brunthaler et al. 2005 M33 proper motion"
    article = "The Geometric Distance and Proper Motion of the Triangulum Galaxy (M33)"
    doi = "10.1126/science.1108342"
    arxiv = "astro-ph/0503058"
    source_url = $pdfUrl
    retrieved_utc = $retrieved
    files = @(
        [ordered]@{
            path = "brunthaler2005_m33_science.pdf"
            bytes = $bytes
            sha256 = $hash
        }
    )
}
$provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output "provenance.json") -Encoding utf8

Write-Host "Downloaded M33 paper to $pdfPath"
