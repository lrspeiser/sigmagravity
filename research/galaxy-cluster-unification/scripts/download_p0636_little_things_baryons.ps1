$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $Root "configs\p0636_little_things_baryon_acquisition.json"
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$Output = Join-Path $Root $Config.raw_directory
New-Item -ItemType Directory -Path $Output -Force | Out-Null

foreach ($Target in $Config.targets) {
    $TargetOutput = Join-Path $Output $Target.id
    New-Item -ItemType Directory -Path $TargetOutput -Force | Out-Null
    $Products = @(
        @{
            Name = $Target.hi_filename
            Url = "https://things.cv.nrao.edu/littlethings/$($Target.archive_directory)/HI/$($Target.hi_filename)"
        },
        @{
            Name = "$($Target.optical_prefix)b.fits"
            Url = "https://science.nrao.edu/science/surveys/littlethings/data/$($Target.archive_directory)/$($Target.optical_prefix)b.fits"
        },
        @{
            Name = "$($Target.optical_prefix)v.fits"
            Url = "https://science.nrao.edu/science/surveys/littlethings/data/$($Target.archive_directory)/$($Target.optical_prefix)v.fits"
        },
        @{
            Name = "$($Target.optical_prefix)_ubvcalib.txt"
            Url = "https://science.nrao.edu/science/surveys/littlethings/data/$($Target.archive_directory)/$($Target.optical_prefix)_ubvcalib.txt"
        }
    )
    foreach ($Product in $Products) {
        $Lower = $Product.Name.ToLower()
        foreach ($Fragment in $Config.forbidden_filename_fragments) {
            if ($Lower.Contains($Fragment.ToLower())) {
                throw "Refusing forbidden P0633 product: $($Product.Name)"
            }
        }
        $Destination = Join-Path $TargetOutput $Product.Name
        if (-not (Test-Path -LiteralPath $Destination)) {
            Invoke-WebRequest -Uri $Product.Url -OutFile $Destination
        }
    }
    Write-Host "$($Target.id): four permitted baryonic products present"
}
