param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "data\raw\r1_replacement_search_sources\loubser2018_bcg_kinematics"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$AllowedRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "data\raw\r1_replacement_search_sources"))
if (-not $OutputDirectory.StartsWith($AllowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must remain inside $AllowedRoot"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$SourceArchive = Join-Path $OutputDirectory "arxiv_1802.07745_source.tar.gz"
$PaperPdf = Join-Path $OutputDirectory "loubser2018_bcg_kinematics.pdf"
$SourceDirectory = Join-Path $OutputDirectory "source"

$Downloads = @(
    @{ Url = "https://export.arxiv.org/e-print/1802.07745"; Path = $SourceArchive },
    @{ Url = "https://arxiv.org/pdf/1802.07745"; Path = $PaperPdf }
)
foreach ($Download in $Downloads) {
    if (-not (Test-Path -LiteralPath $Download.Path)) {
        $Partial = "$($Download.Path).partial"
        Invoke-WebRequest -Uri $Download.Url -OutFile $Partial -UserAgent "sigmagravity-observable-audit/1.0"
        Move-Item -LiteralPath $Partial -Destination $Download.Path
    }
}

if (-not (Test-Path -LiteralPath $SourceDirectory)) {
    New-Item -ItemType Directory -Path $SourceDirectory | Out-Null
    tar -xzf $SourceArchive -C $SourceDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Could not extract the arXiv source archive."
    }
}

$Records = foreach ($Path in @($SourceArchive, $PaperPdf)) {
    $Item = Get-Item -LiteralPath $Path
    [ordered]@{
        local_path = $Item.FullName.Substring($ProjectRoot.Length + 1).Replace('\', '/')
        size_bytes = $Item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Item.FullName).Hash
    }
}
$Provenance = [ordered]@{
    provenance_version = "R1A2-cycle3-loubser2018-0.1"
    downloaded_utc = [DateTime]::UtcNow.ToString("o")
    source = "Loubser et al. 2018, MNRAS 477, 335; arXiv:1802.07745"
    doi = "10.1093/mnras/sty498"
    urls = $Downloads.Url
    records = $Records
}
$Provenance | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "provenance.json")
Write-Output "Downloaded and extracted Loubser et al. 2018 to $OutputDirectory"
