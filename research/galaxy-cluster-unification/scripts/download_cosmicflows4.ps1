param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$destination = Join-Path $project "data\raw\cosmicflows4"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$landingPage = "https://projets.ip2i.in2p3.fr/cosmicflows/"
$files = @(
    [ordered]@{
        name = "CF4gp_new_64-z008_delta.fits"
        url = "${landingPage}CF4gp_new_64-z008_delta.fits"
        expected_bytes = 1054080
        role = "Grouped Cosmicflows-4 density contrast grid"
    },
    [ordered]@{
        name = "CF4gp_new_64-z008_delta_error.fits"
        url = "${landingPage}CF4gp_new_64-z008_delta_error.fits"
        expected_bytes = 37440
        role = "Grouped Cosmicflows-4 published delta error product (2-D)"
    },
    [ordered]@{
        name = "CF4_new_64-z008_delta.fits"
        url = "${landingPage}CF4_new_64-z008_delta.fits"
        expected_bytes = 1054080
        role = "Ungrouped Cosmicflows-4 density contrast grid"
    },
    [ordered]@{
        name = "CF4_new_64-z008_delta_error.fits"
        url = "${landingPage}CF4_new_64-z008_delta_error.fits"
        expected_bytes = 37440
        role = "Ungrouped Cosmicflows-4 published delta error product (2-D)"
    },
    [ordered]@{
        name = "CF4_catalog_ReadMe.txt"
        url = "https://cdsarc.cds.unistra.fr/ftp/cats/J/ApJ/944/94/ReadMe"
        expected_bytes = 19918
        role = "VizieR J/ApJ/944/94 machine-readable catalog documentation"
    },
    [ordered]@{
        name = "CF4_table4_groups.dat.gz"
        url = "https://cdsarc.cds.unistra.fr/ftp/cats/J/ApJ/944/94/table4.dat.gz"
        expected_bytes = 2528953
        role = "Cosmicflows-4 group distances and peculiar velocities (38,053 rows)"
    },
    [ordered]@{
        name = "CF4_new_128-z008_delta.fits"
        url = "https://zenodo.org/records/20653238/files/CF4_new_128-z008_delta.fits?download=1"
        expected_bytes = 8392320
        role = "Official 2026 ungrouped 128^3 density grid for sensitivity and convention checks"
    }
)

foreach ($entry in $files) {
    $target = Join-Path $destination $entry.name
    $needsDownload = $Force -or -not (Test-Path -LiteralPath $target)
    if (-not $needsDownload) {
        $needsDownload = (Get-Item -LiteralPath $target).Length -ne $entry.expected_bytes
    }
    if ($needsDownload) {
        Write-Output "Downloading $($entry.name)"
        $partial = "$target.partial"
        curl.exe -L --fail --silent --show-error --output $partial $entry.url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $($entry.url)"
        }
        $actualBytes = (Get-Item -LiteralPath $partial).Length
        if ($actualBytes -ne $entry.expected_bytes) {
            throw "Unexpected byte count for $($entry.name): $actualBytes (expected $($entry.expected_bytes))"
        }
        Move-Item -LiteralPath $partial -Destination $target -Force
    }
}

$records = foreach ($entry in $files) {
    $target = Join-Path $destination $entry.name
    [ordered]@{
        path = $entry.name
        role = $entry.role
        url = $entry.url
        bytes = (Get-Item -LiteralPath $target).Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    }
}

$manifest = [ordered]@{
    dataset = "Cosmicflows-4 three-dimensional density reconstructions"
    downloaded_utc = (Get-Date).ToUniversalTime().ToString("o")
    official_landing_page = $landingPage
    citation = "Courtois et al., Astronomy & Astrophysics 670, L15 (2023)"
    citation_doi = "10.1051/0004-6361/202245331"
    zenodo_record = "https://doi.org/10.5281/zenodo.20653238"
    coordinate_order = "The FITS files have no WCS. The official page declares FITS axes (SGZ,SGY,SGX); scripts/build_cf4_environment.py applies FITS-to-NumPy axis reversal, the published 7.8125 h^-1 Mpc voxel scale, and validates the sky-to-supergalactic transform against the companion catalog."
    files = @($records)
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $destination "provenance.json") -Encoding utf8
Write-Output "Cosmicflows-4 density snapshot is ready at $destination"
