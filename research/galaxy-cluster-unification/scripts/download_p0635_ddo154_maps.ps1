$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Output = Join-Path $Root "data\raw\p0635_commissioning_ddo154"
New-Item -ItemType Directory -Path $Output -Force | Out-Null

$Products = @(
    @{
        Name = "DDO154_NA_X0_P_R.FITS"
        Url = "https://things.cv.nrao.edu/littlethings/ddo154/HI/DDO154_NA_X0_P_R.FITS"
        Sha256 = "7dc505fffa9e35530c7d556b6e84357e73bd7be4c17624eab82d5f4c8874b911"
    },
    @{
        Name = "d154b.fits"
        Url = "https://science.nrao.edu/science/surveys/littlethings/data/ddo154/d154b.fits"
        Sha256 = "34d67c96d5b8649ba6c265ef3d71ab272aae1856ea0de8ff7443279e5e72c23b"
    },
    @{
        Name = "d154v.fits"
        Url = "https://science.nrao.edu/science/surveys/littlethings/data/ddo154/d154v.fits"
        Sha256 = "8e88d7db013d09a7bcdfd21a5f6c51c2d1a7a732af36131eb7d7ec1803d00181"
    },
    @{
        Name = "d154_ubvcalib.txt"
        Url = "https://science.nrao.edu/science/surveys/littlethings/data/ddo154/d154_ubvcalib.txt"
        Sha256 = "75b925dcd3277948d4ce14a4526fd4950860024a17225da0cf47860369ace5c8"
    }
)

foreach ($Product in $Products) {
    $Destination = Join-Path $Output $Product.Name
    $Valid = Test-Path -LiteralPath $Destination
    if ($Valid) {
        $Valid = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLower() -eq $Product.Sha256
    }
    if (-not $Valid) {
        Invoke-WebRequest -Uri $Product.Url -OutFile $Destination
    }
    $Actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLower()
    if ($Actual -ne $Product.Sha256) {
        throw "SHA-256 mismatch for $($Product.Name): $Actual"
    }
    Write-Host "$($Product.Name): verified"
}
