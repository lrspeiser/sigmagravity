$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$protocolPath = Join-Path $projectRoot "configs/p0554_all_baryon_route_acquisition_protocol.json"
$protocol = Get-Content -Raw -LiteralPath $protocolPath | ConvertFrom-Json
if (-not $protocol.status.StartsWith("frozen_")) {
    throw "The all-baryon acquisition protocol is not frozen."
}

$outputDirectory = Join-Path $projectRoot $protocol.outputs.directory
$hstDirectory = Join-Path $projectRoot $protocol.outputs.hst_directory
$chandraDirectory = Join-Path $projectRoot $protocol.outputs.chandra_directory
New-Item -ItemType Directory -Force -Path $hstDirectory, $chandraDirectory | Out-Null
$resolvedOutput = [System.IO.Path]::GetFullPath($outputDirectory)
$records = [System.Collections.Generic.List[object]]::new()

function Save-VerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][string]$SystemLabel,
        [int]$Obsid = 0,
        [long]$ExpectedLength = 0
    )
    $resolvedDestination = [System.IO.Path]::GetFullPath($Destination)
    if (-not $resolvedDestination.StartsWith($resolvedOutput, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing archive path outside the acquisition directory: $resolvedDestination"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    if (-not (Test-Path -LiteralPath $Destination)) {
        $partial = "$Destination.partial"
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial
        Move-Item -LiteralPath $partial -Destination $Destination
    }
    $item = Get-Item -LiteralPath $Destination
    if ($ExpectedLength -gt 0 -and $item.Length -ne $ExpectedLength) {
        throw "Length mismatch for ${Destination}: expected $ExpectedLength, found $($item.Length)"
    }
    $records.Add([ordered]@{
        kind = $Kind
        system_label = $SystemLabel
        obsid = if ($Obsid -gt 0) { $Obsid } else { $null }
        source_url = $Url
        local_path = $resolvedDestination.Substring($projectRoot.Length + 1).Replace('\', '/')
        size_bytes = [long]$item.Length
        sha256 = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    })
    Write-Host "$SystemLabel $Kind $($item.Name) $($item.Length) bytes"
}

$newHst = $protocol.hst.new_system
foreach ($productName in "science", "weight") {
    $product = $newHst.$productName
    Save-VerifiedDownload -Url $product.url `
        -Destination (Join-Path $hstDirectory $product.filename) `
        -Kind "hst_f160w_$productName" -SystemLabel $newHst.label `
        -ExpectedLength ([long]$product.content_length)
}

$archiveRoot = $protocol.official_sources.chandra_download_root
foreach ($system in $protocol.chandra.systems) {
    if ($system.source -and $system.source.StartsWith("reused:")) { continue }
    foreach ($obsidValue in $system.obsids) {
        $obsid = [int]$obsidValue
        $baseUrl = "$archiveRoot/$($obsid % 10)/$obsid/primary/"
        $html = (Invoke-WebRequest -UseBasicParsing -Uri $baseUrl).Content
        $links = [regex]::Matches($html, 'href="([^"]+)"') | ForEach-Object {
            $_.Groups[1].Value
        } | Where-Object {
            $_ -match '(evt2|cntr_img2|full_img2)\.fits\.gz$'
        }
        if (($links | Where-Object { $_ -match 'evt2\.fits\.gz$' }).Count -ne 1) {
            throw "Expected exactly one primary evt2 product for ObsID $obsid."
        }
        foreach ($filename in $links) {
            Save-VerifiedDownload -Url ($baseUrl + $filename) `
                -Destination (Join-Path $chandraDirectory "$($system.label)/$obsid/$filename") `
                -Kind "chandra_primary" -SystemLabel $system.label -Obsid $obsid
        }
    }
}

$reused = [System.Collections.Generic.List[object]]::new()
$reusedHst = Get-Content -Raw -LiteralPath (Join-Path $projectRoot $protocol.hst.reused_acquisition_protocol) | ConvertFrom-Json
foreach ($label in $protocol.hst.reused_labels) {
    $system = $reusedHst.systems | Where-Object { $_.label -eq $label }
    foreach ($productName in "science", "weight") {
        $product = $system.$productName
        $path = Join-Path $projectRoot (Join-Path $reusedHst.outputs.directory $product.filename)
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing reused HST product: $path" }
        $item = Get-Item -LiteralPath $path
        $reused.Add([ordered]@{
            kind = "hst_f160w_$productName"
            system_label = $label
            local_path = ([System.IO.Path]::GetFullPath($path)).Substring($projectRoot.Length + 1).Replace('\', '/')
            size_bytes = [long]$item.Length
            sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        })
    }
}
$rxSource = ($protocol.chandra.systems | Where-Object { $_.label -eq "RXJ2129" }).source.Substring(7)
$rxProvenance = Join-Path $projectRoot "$rxSource/provenance.json"
if (-not (Test-Path -LiteralPath $rxProvenance)) { throw "Missing reused RX J2129 Chandra provenance." }

$provenance = [ordered]@{
    provenance_version = "P0554-ALL-BARYON-ROUTE-ACQUISITION-RESULTS-0.1.0"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    protocol_path = "configs/p0554_all_baryon_route_acquisition_protocol.json"
    protocol_sha256 = (Get-FileHash -LiteralPath $protocolPath -Algorithm SHA256).Hash.ToLowerInvariant()
    downloaded_records = $records
    reused_records = $reused
    reused_rxj2129_chandra_provenance = [ordered]@{
        local_path = ([System.IO.Path]::GetFullPath($rxProvenance)).Substring($projectRoot.Length + 1).Replace('\', '/')
        sha256 = (Get-FileHash -LiteralPath $rxProvenance -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$provenancePath = Join-Path $projectRoot $protocol.outputs.provenance
$provenance | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $provenancePath
Write-Host "Wrote $provenancePath with $($records.Count) downloaded and $($reused.Count) reused records."
