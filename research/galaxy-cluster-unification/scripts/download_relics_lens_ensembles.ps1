$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$destinationRoot = Join-Path $root "data/raw/relics_lens_models"

$models = @(
    @{
        System = "A2537"
        Slug = "abell2537"
        Version = "v1"
        BaseUrl = "https://archive.stsci.edu/hlsps/relics/abell2537/models/lenstool"
    },
    @{
        System = "MACS_J0417"
        Slug = "macs0417m11"
        Version = "v2"
        BaseUrl = "https://archive.stsci.edu/hlsps/relics/macs0417m11/models/lenstool/v2"
    },
    @{
        System = "MACS_J0949"
        Slug = "rxc0949p17"
        Version = "v1"
        BaseUrl = "https://archive.stsci.edu/hlsps/relics/rxc0949p17/models/lenstool"
    }
)

New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null

$readmeUrl = "https://archive.stsci.edu/hlsps/relics/docs/hlsp_relics_hst_multi_lens-models_multi_v3_readme.pdf"
$readmePath = Join-Path $destinationRoot "hlsp_relics_hst_multi_lens-models_multi_v3_readme.pdf"
if (-not (Test-Path -LiteralPath $readmePath)) {
    & curl.exe --fail --location --silent --show-error --output $readmePath $readmeUrl
    if ($LASTEXITCODE -ne 0) { throw "RELICS README download failed" }
}

foreach ($model in $models) {
    $systemDirectory = Join-Path $destinationRoot $model.System
    $rangeDirectory = Join-Path $systemDirectory "range"
    New-Item -ItemType Directory -Force -Path $rangeDirectory | Out-Null

    $bestName = "hlsp_relics_model_model_$($model.Slug)_lenstool_$($model.Version)_kappa.fits"
    $bestPath = Join-Path $systemDirectory $bestName
    if (-not (Test-Path -LiteralPath $bestPath)) {
        & curl.exe --fail --location --silent --show-error --output $bestPath "$($model.BaseUrl)/$bestName"
        if ($LASTEXITCODE -ne 0) { throw "Best-map download failed for $($model.System)" }
    }

    $urls = @()
    foreach ($index in 0..99) {
        $mapId = $index.ToString("000")
        $name = "hlsp_relics_model_model_$($model.Slug)_lenstool-map$($mapId)_$($model.Version)_kappa.fits"
        $path = Join-Path $rangeDirectory $name
        if (-not (Test-Path -LiteralPath $path)) {
            $urls += "$($model.BaseUrl)/range/$name"
        }
    }
    if ($urls.Count -gt 0) {
        & curl.exe --fail --location --silent --show-error --parallel --parallel-max 8 --remote-name-all --output-dir $rangeDirectory $urls
        if ($LASTEXITCODE -ne 0) { throw "Range-map download failed for $($model.System)" }
    }

    $downloaded = @(Get-ChildItem -LiteralPath $rangeDirectory -Filter "*_kappa.fits")
    if ($downloaded.Count -ne 100) {
        throw "Expected 100 range maps for $($model.System); found $($downloaded.Count)"
    }
    Write-Host "$($model.System): best map plus $($downloaded.Count) MCMC range maps"
}
