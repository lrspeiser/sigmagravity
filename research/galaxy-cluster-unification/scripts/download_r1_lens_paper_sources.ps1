$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$destination = Join-Path $projectRoot "data/raw/r1_lens_paper_sources"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$sources = @(
    @{ Id = "1710.09329"; Name = "cerny2018_a2537_source.tar" },
    @{ Id = "1810.13439"; Name = "mahler2019_macs0417_relics_source.tar" },
    @{ Id = "1811.02505"; Name = "jauzac2019_macs0417_source.tar" },
    @{ Id = "2207.10520"; Name = "allingham2023_macs0949_source.tar" }
)

$records = @()
foreach ($source in $sources) {
    $url = "https://export.arxiv.org/e-print/$($source.Id)"
    $path = Join-Path $destination $source.Name
    if (-not (Test-Path -LiteralPath $path)) {
        & curl.exe --fail --location --silent --show-error --output $path $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" }
    }
    $item = Get-Item -LiteralPath $path
    $records += [ordered]@{
        arxiv_id = $source.Id
        url = $url
        path = $source.Name
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$provenance = [ordered]@{
    generated_utc = [DateTime]::UtcNow.ToString("o")
    purpose = "Primary-source audit for BCG centers, lens constraints, and rerunnable-model availability"
    files = $records
}
$provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination "provenance.json") -Encoding utf8
$records | Format-Table -AutoSize
