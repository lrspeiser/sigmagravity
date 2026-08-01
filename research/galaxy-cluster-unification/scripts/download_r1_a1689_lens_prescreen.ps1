param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "data\raw\r1_a1689_lens_prescreen"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$AllowedRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "data\raw"))
if (-not $OutputDirectory.StartsWith($AllowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must remain inside $AllowedRoot"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$Downloads = @(
    @{ Url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/723/1678/ReadMe"; Name = "cds_ReadMe.txt"; Role = "CDS machine-readable schema and provenance" },
    @{ Url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/723/1678/table5.dat"; Name = "coe2010_table5_multiple_images.dat"; Role = "corrected coordinates and independent image redshift summaries for 135 observed images" },
    @{ Url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/723/1678/table7.dat"; Name = "coe2010_table7_multiple_image_systems.dat"; Role = "42-family audit table containing input and LensPerfect output redshifts" },
    @{ Url = "https://export.arxiv.org/e-print/1005.0398"; Name = "arxiv_1005.0398_source.tar.gz"; Role = "primary paper source" },
    @{ Url = "https://arxiv.org/pdf/1005.0398"; Name = "coe2010_a1689_lensperfect.pdf"; Role = "primary paper PDF" }
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
    provenance_version = "R1B1-A1689-lens-prescreen-0.1"
    downloaded_utc = [DateTime]::UtcNow.ToString("o")
    primary_source = "Coe et al. 2010, ApJ 723, 1678; arXiv:1005.0398"
    catalog = "CDS J/ApJ/723/1678 tables 5 and 7"
    records = $Records
}
$Provenance | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "provenance.json")
Write-Output "Downloaded A1689 lens-prescreen products to $OutputDirectory"
