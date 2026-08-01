param([switch]$Force)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDirectory = Join-Path $projectRoot "data/raw/r1_rxj2129_baryons"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$hstBase = "https://archive.stsci.edu/pub/hlsp/clash/rxj2129/data/hst"
$products = @(
    [ordered]@{
        name = "clash_hst_readme.txt"
        url = "$hstBase/hlsp_clash_hst_rxj2129_readme.txt"
        role = "CLASH HST reduction, units, astrometry, and product metadata"
    },
    [ordered]@{
        name = "hlsp_clash_hst_wfc3ir_rxj2129_f125w_v1_drz.fits"
        url = "$hstBase/scale_65mas/hlsp_clash_hst_wfc3ir_rxj2129_f125w_v1_drz.fits"
        role = "rest-frame approximately 1-micron BCG plus ICL surface-brightness image"
    },
    [ordered]@{
        name = "hlsp_clash_hst_wfc3ir_rxj2129_f125w_v1_wht.fits"
        url = "$hstBase/scale_65mas/hlsp_clash_hst_wfc3ir_rxj2129_f125w_v1_wht.fits"
        role = "inverse-variance weight image for F125W"
    },
    [ordered]@{
        name = "hlsp_clash_hst_acs_rxj2129_f814w_v1_drz.fits"
        url = "$hstBase/scale_65mas/hlsp_clash_hst_acs_rxj2129_f814w_v1_drz.fits"
        role = "second-band BCG, ICL, and color-systematic image"
    },
    [ordered]@{
        name = "hlsp_clash_hst_acs_rxj2129_f814w_v1_wht.fits"
        url = "$hstBase/scale_65mas/hlsp_clash_hst_acs_rxj2129_f814w_v1_wht.fits"
        role = "inverse-variance weight image for F814W"
    },
    [ordered]@{
        name = "hlsp_clash_hst_ir_rxj2129_cat-molino.txt"
        url = "https://archive.stsci.edu/pub/hlsp/clash/rxj2129/catalogs/molino/hlsp_clash_hst_ir_rxj2129_cat-molino.txt"
        role = "ICL-subtracted 16-band photometry, photometric redshifts, and published stellar-mass estimates for satellite selection"
    },
    [ordered]@{
        name = "tian2020_cds_readme.txt"
        url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/896/70/ReadMe"
        role = "machine-readable definitions and uncertainty notes for the published RX J2129 BCG stellar and gas mass anchor"
    },
    [ordered]@{
        name = "tian2020_arxiv_source.tar.gz"
        url = "https://export.arxiv.org/e-print/2001.08340"
        role = "primary-source equations and method text for the published Hernquist BCG baseline and gas-profile lineage"
    },
    [ordered]@{
        name = "donahue2014_arxiv_source.tar.gz"
        url = "https://export.arxiv.org/e-print/1405.7876"
        role = "primary-source CLASH-X gas-density model and RX J2129 profile figure; no machine-readable gas profile or covariance is included"
    },
    [ordered]@{
        name = "cooke2016_arxiv_source.tar.gz"
        url = "https://export.arxiv.org/e-print/1610.05310"
        role = "primary-source photometric-aperture and MAGPHYS lineage for the published RX J2129 BCG stellar-mass normalization"
    }
)

$records = @()
foreach ($product in $products) {
    $target = Join-Path $outputDirectory $product.name
    if ((Test-Path -LiteralPath $target) -and -not $Force) {
        Write-Host "Already present: $target"
    }
    else {
        $partial = "$target.partial"
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force
        }
        & curl.exe --fail --location --retry 4 --retry-delay 10 --output $partial $product.url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $($product.url)"
        }
        Move-Item -LiteralPath $partial -Destination $target -Force
    }
    $item = Get-Item -LiteralPath $target
    $hash = Get-FileHash -LiteralPath $target -Algorithm SHA256
    $records += [ordered]@{
        source_url = $product.url
        local_path = "data/raw/r1_rxj2129_baryons/$($product.name)"
        size_bytes = $item.Length
        sha256 = $hash.Hash
        role = $product.role
    }
    Write-Host "Ready: $($product.name) ($($item.Length) bytes)"
}

$provenance = [ordered]@{
    provenance_version = "R1B0-RXJ2129-baryons-0.3"
    generated_utc = (Get-Date).ToUniversalTime().ToString("o")
    selection_rule = "Products were selected from required baryonic components and the frozen 0-5 arcsec support before fitting a gravity residual."
    records = $records
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$json = ($provenance | ConvertTo-Json -Depth 7) + [Environment]::NewLine
[System.IO.File]::WriteAllText((Join-Path $outputDirectory "provenance.json"), $json, $utf8NoBom)
