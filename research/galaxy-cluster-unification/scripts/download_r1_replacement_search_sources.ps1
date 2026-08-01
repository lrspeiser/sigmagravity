$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$destination = Join-Path $projectRoot "data/raw/r1_replacement_search_sources"
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$sources = @(
    @{
        Id = "astro-ph/0310703"
        Slug = "sand2004_six_cluster_bcg_lensing"
        Role = "legacy six-host conference summary of resolved-BCG dynamics plus arc sample"
    },
    @{
        Id = "astro-ph/0309465"
        Slug = "sand2004_six_cluster_full_analysis"
        Role = "legacy six-host full resolved-BCG dynamics plus arc analysis"
    },
    @{
        Id = "2003.08475"
        Slug = "sartoris2020_abell_s1063_dynamics"
        Role = "40-kpc resolved BCG dynamics and full baryonic dynamics model"
    },
    @{
        Id = "2307.06804"
        Slug = "biviano2023_macs_j1206_dynamics"
        Role = "resolved BCG dynamics and full baryonic dynamics model"
    },
    @{
        Id = "2208.14020"
        Slug = "bergamini2023_macs_j0416_lensing"
        Role = "public spectroscopic strong-lens image model and nuisance inputs"
    },
    @{
        Id = "1707.00690"
        Slug = "caminha2017_macs_j1206_lensing"
        Role = "82 spectroscopic images including 11 within the 50-kpc BCG dynamics aperture"
    },
    @{
        Id = "1512.04555"
        Slug = "caminha2016_abell_s1063_lensing"
        Role = "47-image spectroscopic strong-lens catalog for the resolved-BCG dynamics host"
    },
    @{
        Id = "2411.07289"
        Slug = "bolamperti2024_sdss_j0100_group_lens"
        Role = "six-bin brightest-group-galaxy kinematics plus 18 spectroscopic multiple images for cycle 2"
    },
    @{
        Id = "2006.10700"
        Slug = "jauzac2021_three_muse_cluster_lenses"
        Role = "spectroscopic strong-lens image tables and public-MUSE candidate coverage for RX J2129, MS 0451, and MACS J2129 in cycle 2"
    }
)

$records = @()
foreach ($source in $sources) {
    $url = "https://export.arxiv.org/e-print/$($source.Id)"
    $archiveName = "$($source.Slug)_source.tar"
    $archivePath = Join-Path $destination $archiveName
    $extractPath = Join-Path $destination $source.Slug
    if (-not (Test-Path -LiteralPath $archivePath)) {
        & curl.exe --fail --location --silent --show-error --output $archivePath $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" }
    }
    New-Item -ItemType Directory -Force -Path $extractPath | Out-Null
    if (-not (Get-ChildItem -LiteralPath $extractPath -Force | Select-Object -First 1)) {
        & tar.exe -xf $archivePath -C $extractPath
        if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $archivePath" }
    }
    $item = Get-Item -LiteralPath $archivePath
    $records += [ordered]@{
        arxiv_id = $source.Id
        role = $source.Role
        url = $url
        archive_path = $archiveName
        extracted_path = $source.Slug
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$provenance = [ordered]@{
    generated_utc = [DateTime]::UtcNow.ToString("o")
    purpose = "Residual-blind R1A.2 replacement-host discovery and observable-availability audit"
    selection_note = "Sources were selected from resolved-BCG and strong-lens coverage, before inspecting any alternative-gravity residual."
    files = $records
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$provenanceJson = ($provenance | ConvertTo-Json -Depth 6) + [Environment]::NewLine
[System.IO.File]::WriteAllText((Join-Path $destination "provenance.json"), $provenanceJson, $utf8NoBom)
$records | Format-Table -AutoSize
