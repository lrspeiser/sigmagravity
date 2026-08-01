param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "data\raw\r1_clash_zitrin2015"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$AllowedRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "data\raw"))
if (-not $OutputDirectory.StartsWith($AllowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must remain inside $AllowedRoot"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$Downloads = @(
    @{ Url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/801/44/ReadMe"; Name = "cds_ReadMe.txt"; Role = "CDS machine-readable schema and provenance" },
    @{ Url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/801/44/table2.dat"; Name = "zitrin2015_table2_multiple_images.dat"; Role = "complete CLASH multiple-image and candidate coordinate catalog" },
    @{ Url = "https://export.arxiv.org/e-print/1411.1414v4"; Name = "arxiv_1411.1414v4_source.tar.gz"; Role = "primary paper source" },
    @{ Url = "https://arxiv.org/pdf/1411.1414v4"; Name = "zitrin2015_clash_lensing.pdf"; Role = "primary paper PDF" },
    @{ Url = "https://archive.stsci.edu/prepds/clash/"; Name = "mast_clash_release_page.html"; Role = "official release statement that the RXJ1532 model uses one unconfirmed candidate family" }
)

$Systems = @("a209", "macs0647", "macs0717", "macs0744", "macs1149", "macs1720", "rxj1532")
foreach ($System in $Systems) {
    $BaseUrl = "https://archive.stsci.edu/missions/hlsp/clash/$System/models/zitrin/ltm-gauss/v2"
    $Stem = "hlsp_clash_model_${System}_zitrin-ltm-gauss_v2"
    $Downloads += @(
        @{ Url = "$BaseUrl/${Stem}_readme.txt"; Name = "mast_${System}_zitrin_ltm_v2_readme.txt"; Role = "archived model documentation and likelihood convention for $System" },
        @{ Url = "$BaseUrl/${Stem}_params.txt"; Name = "mast_${System}_zitrin_ltm_v2_params.txt"; Role = "archived model parameter summary for $System" }
    )
}

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
    provenance_version = "R1-CLASH-Zitrin2015-0.1"
    downloaded_utc = [DateTime]::UtcNow.ToString("o")
    primary_source = "Zitrin et al. 2015, ApJ 801, 44; arXiv:1411.1414v4"
    catalog = "CDS J/ApJ/801/44 table2"
    archive = "MAST CLASH Zitrin LTM-Gauss v2"
    intentionally_excluded = @("convergence maps", "deflection maps", "shear maps", "magnification maps", "MCMC range maps")
    records = $Records
}
$Provenance | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "provenance.json")
Write-Output "Downloaded Zitrin2015 observable catalog and seven MAST audit packages to $OutputDirectory"
