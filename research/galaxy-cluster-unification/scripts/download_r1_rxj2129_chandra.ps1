$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDirectory = Join-Path $projectRoot "data/raw/r1_rxj2129_chandra"
$archiveRoot = "https://cxc.cfa.harvard.edu/cdaftp/byobsid"
$observations = @(
    @{ obsid = 9370; shard = 0; exposure_ks = 29.642256754205; target = "RXCJ2129.6+0005" },
    @{ obsid = 552; shard = 2; exposure_ks = 9.961839418835199; target = "RXJ2129.6+0005" }
)

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$resolvedOutput = [System.IO.Path]::GetFullPath($outputDirectory)
$records = [System.Collections.Generic.List[object]]::new()

function Get-CdaFiles {
    param(
        [Parameter(Mandatory = $true)][string]$BaseUrl,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$RelativePath
    )
    $url = if ($RelativePath) { "$BaseUrl/$RelativePath" } else { "$BaseUrl/" }
    if (-not $url.EndsWith("/")) { $url += "/" }
    $html = (Invoke-WebRequest -UseBasicParsing -Uri $url).Content
    $links = [regex]::Matches($html, 'href="([^"]+)"') | ForEach-Object {
        $_.Groups[1].Value
    }
    $files = [System.Collections.Generic.List[object]]::new()
    foreach ($link in $links) {
        if ($link -eq "../" -or $link.StartsWith("?")) { continue }
        if ($link.EndsWith("/")) {
            $child = if ($RelativePath) { "$RelativePath/$($link.TrimEnd('/'))" } else { $link.TrimEnd('/') }
            foreach ($item in (Get-CdaFiles -BaseUrl $BaseUrl -RelativePath $child)) {
                $files.Add($item)
            }
            continue
        }
        if (
            $link -eq "00README" -or $link -eq "oif.fits" -or
            $link.EndsWith(".fits.gz") -or $link.EndsWith(".pdf") -or
            $link.EndsWith(".pdf.gz")
        ) {
            $relativeFile = if ($RelativePath) { "$RelativePath/$link" } else { $link }
            $files.Add(@{ relative = $relativeFile; url = "$url$link" })
        }
    }
    return $files
}

foreach ($observation in $observations) {
    $obsid = [int]$observation.obsid
    $baseUrl = "$archiveRoot/$($observation.shard)/$obsid"
    $obsDirectory = Join-Path $outputDirectory $obsid
    New-Item -ItemType Directory -Force -Path $obsDirectory | Out-Null
    $files = Get-CdaFiles -BaseUrl $baseUrl -RelativePath ""
    foreach ($file in $files) {
        $relativeNative = $file.relative.Replace("/", [System.IO.Path]::DirectorySeparatorChar)
        $destination = Join-Path $obsDirectory $relativeNative
        $resolvedDestination = [System.IO.Path]::GetFullPath($destination)
        if (-not $resolvedDestination.StartsWith($resolvedOutput, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing archive path outside the output directory: $resolvedDestination"
        }
        $parent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        if (-not (Test-Path -LiteralPath $destination)) {
            $partial = "$destination.partial"
            Invoke-WebRequest -UseBasicParsing -Uri $file.url -OutFile $partial
            Move-Item -LiteralPath $partial -Destination $destination
        }
        $item = Get-Item -LiteralPath $destination
        $hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        $records.Add([ordered]@{
            obsid = $obsid
            target = $observation.target
            exposure_ks = [double]$observation.exposure_ks
            source_url = $file.url
            local_path = "data/raw/r1_rxj2129_chandra/$obsid/$($file.relative)"
            size_bytes = [long]$item.Length
            sha256 = $hash
        })
        Write-Host "$obsid $($file.relative) $($item.Length) bytes"
    }
}

$provenance = [ordered]@{
    provenance_version = "R1B3-RXJ2129-Chandra-archive-0.1"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    selection_rule = "All public standard archive FITS products and validation reports for the two Chandra observations within 0.15 degrees of the frozen RX J2129 center, selected before reconstructing a gas profile or reading a gravity residual."
    archive_query = [ordered]@{
        service = "Chandra Observation Catalog TAP"
        endpoint = "https://cda.cfa.harvard.edu/cxctap/sync"
        center_ra_deg = 322.41651
        center_dec_deg = 0.08923
        result_obsids = @(9370, 552)
        total_public_exposure_ks = 39.6040961730402
    }
    records = $records
}
$provenancePath = Join-Path $outputDirectory "provenance.json"
$provenance | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $provenancePath
Write-Host "Wrote $provenancePath with $($records.Count) file records."
