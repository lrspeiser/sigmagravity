param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "data\raw\r1_a2261_lens_observables"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$AllowedRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "data\raw"))
if (-not $OutputDirectory.StartsWith($AllowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must remain inside $AllowedRoot"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$Downloads = @(
    @{ Url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/757/22/ReadMe"; Name = "cds_ReadMe.txt"; Role = "CDS machine-readable schema and provenance" },
    @{ Url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/757/22/table3.dat"; Name = "coe2012_table3_multiple_images.dat"; Role = "30 measured multiple-image coordinates and photometric-redshift summaries" },
    @{ Url = "https://export.arxiv.org/e-print/1201.1616"; Name = "arxiv_1201.1616_source.tar.gz"; Role = "primary paper source" },
    @{ Url = "https://arxiv.org/pdf/1201.1616"; Name = "coe2012_a2261.pdf"; Role = "primary paper PDF" }
)

foreach ($Download in $Downloads) {
    $Path = Join-Path $OutputDirectory $Download.Name
    if (-not (Test-Path -LiteralPath $Path)) {
        $Partial = "$Path.partial"
        Invoke-WebRequest -Uri $Download.Url -OutFile $Partial -UserAgent "sigmagravity-observable-audit/1.0"
        Move-Item -LiteralPath $Partial -Destination $Path
    }
}

$Records = foreach ($Download in $Downloads) {
    $Path = Join-Path $OutputDirectory $Download.Name
    $Item = Get-Item -LiteralPath $Path
    [ordered]@{
        local_path = $Item.FullName.Substring($ProjectRoot.Length + 1).Replace('\', '/')
        source_url = $Download.Url
        role = $Download.Role
        size_bytes = $Item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Item.FullName).Hash
    }
}
$Provenance = [ordered]@{
    provenance_version = "R1A2-a2261-observable-0.1"
    downloaded_utc = [DateTime]::UtcNow.ToString("o")
    primary_source = "Coe et al. 2012, ApJ 757, 22; arXiv:1201.1616"
    catalog = "CDS J/ApJ/757/22 table3"
    later_spectroscopic_update = [ordered]@{
        source = "Rydberg et al. 2017, MNRAS 467, 768"
        doi = "10.1093/mnras/stx157"
        family = 4
        image = "4a"
        redshift = 3.377
        note = "Published MOSFIRE [O II] doublet and [Ne III] line redshift; no raw spectrum is included in this acquisition."
    }
    records = $Records
}
$Provenance | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "provenance.json")
Write-Output "Downloaded A2261 observable-level strong-lens inputs to $OutputDirectory"
