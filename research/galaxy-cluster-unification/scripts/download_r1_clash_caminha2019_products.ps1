param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "data\raw\r1_clash_caminha2019"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$AllowedRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "data\raw"))
if (-not $OutputDirectory.StartsWith($AllowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must remain inside $AllowedRoot"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$Base = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/632/A36"
$Downloads = @(
    @{ Relative = "ReadMe"; Local = "cds_ReadMe.txt"; Role = "catalog schema" },
    @{ Relative = "list.dat"; Local = "cds_list.dat"; Role = "published product manifest" },
    @{ Relative = "table2.dat"; Local = "table2_full_redshift_catalog.dat"; Role = "full MUSE redshift catalog" },
    @{ Relative = "tablea2.dat"; Local = "tablea2_multiple_images.dat"; Role = "150 spectroscopically identified multiple-image rows" }
)
$Models = @(
    @{ Folder = "MACSJ0329-P2-shear_mcmc"; Members = "members_lenstool_v2.dat"; Arcs = "obs_arcs_v3.dat" },
    @{ Folder = "MACSJ0429-P1_mcmc"; Members = "members_lenstool_v1.dat"; Arcs = "obs_arcs_v1.dat" },
    @{ Folder = "MACSJ1115-P1_mcmc"; Members = "members_lenstool_v2.dat"; Arcs = "obs_arcs_v1.dat" },
    @{ Folder = "MACSJ1311-P1_mcmc"; Members = "members_lenstool_v1.dat"; Arcs = "obs_arcs_v3.dat" },
    @{ Folder = "MACSJ1931-P2_circular_mcmc"; Members = "members_lenstool_v2.dat"; Arcs = "obs_arcs_v3.dat" },
    @{ Folder = "MACSJ2129-P2_mcmc"; Members = "members_lenstool_v2.dat"; Arcs = "obs_arcs_v02.dat" },
    @{ Folder = "RXJ1347-P2-shear_mcmc"; Members = "members_lenstool_v2.dat"; Arcs = "obs_arcs_v4.dat" },
    @{ Folder = "RXJ2129-P1_mcmc"; Members = "members_lenstool_v2.dat"; Arcs = "obs_arcs_v3.dat" }
)
foreach ($Model in $Models) {
    foreach ($File in @("bayes.dat", "confidence_levels.png", "convergence_test.png", "lenstool_in.par", $Model.Members, $Model.Arcs)) {
        $Downloads += @{
            Relative = "$($Model.Folder)/$File"
            Local = "$($Model.Folder)/$File"
            Role = if ($File -eq $Model.Arcs) { "observed multiple-image catalog" } elseif ($File -eq "bayes.dat") { "model-dependent nuisance MCMC chain" } else { "rerunnable model input or diagnostic" }
        }
    }
}
$Downloads += @(
    @{ Relative = "https://export.arxiv.org/e-print/1903.05103"; Local = "arxiv_1903.05103_source.tar.gz"; Role = "primary paper source" },
    @{ Relative = "https://arxiv.org/pdf/1903.05103"; Local = "caminha2019.pdf"; Role = "primary paper PDF" }
)

$Records = foreach ($Download in $Downloads) {
    $Url = if ($Download.Relative.StartsWith("https://")) { $Download.Relative } else { "$Base/$($Download.Relative)" }
    $Path = Join-Path $OutputDirectory $Download.Local
    $Parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    if (-not (Test-Path -LiteralPath $Path)) {
        $Partial = "$Path.partial"
        & curl.exe -L --fail --silent --show-error --retry 4 --retry-delay 2 --continue-at - --user-agent "sigmagravity-observable-audit/1.0" --output $Partial $Url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed for $Url"
        }
        Move-Item -LiteralPath $Partial -Destination $Path
    }
    $Item = Get-Item -LiteralPath $Path
    [ordered]@{
        local_path = $Item.FullName.Substring($ProjectRoot.Length + 1).Replace('\', '/')
        source_url = $Url
        role = $Download.Role
        size_bytes = $Item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Item.FullName).Hash
    }
}
$Provenance = [ordered]@{
    provenance_version = "R1-CLASH-Caminha2019-0.1"
    downloaded_utc = [DateTime]::UtcNow.ToString("o")
    primary_source = "Caminha et al. 2019, A&A 632, A36; arXiv:1903.05103"
    catalog = "CDS J/A+A/632/A36"
    downloaded_model_packages = $Models.Folder
    excluded_derived_products = @("convergence", "deflection", "shear", "magnification", "best-fit maps")
    records = $Records
}
$Provenance | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "provenance.json")
Write-Output "Downloaded Caminha et al. observable and nuisance products to $OutputDirectory"
