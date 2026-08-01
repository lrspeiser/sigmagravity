param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

$downloads = @(
    [ordered]@{
        dataset = "SDSS DR17 MaNGA DRPall v3_1_1"
        citation = "SDSS DR17 MaNGA Data Reduction Pipeline summary catalog"
        url = "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits"
        directory = "manga_dr17"
        filename = "drpall-v3_1_1.fits"
        expected_bytes = 75360960
        expected_sha256 = "9d34371c157bf7d1b98eb389afd8553243af1f37f2efd2527d02fd8698d862b4"
    },
    [ordered]@{
        dataset = "eROSITA eRASS1 primary galaxy groups and clusters catalog v3.2"
        citation = "Bulbul et al., Astronomy & Astrophysics 685, A106 (2024)"
        url = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/685/A106/fits/erass1cl_main_v3.2.fits"
        directory = "erass1_clusters"
        filename = "erass1cl_main_v3.2.fits"
        expected_bytes = 50636160
        expected_sha256 = "e79c392e2d7b9e26105ead7d56787713465aa853feb040a227891fb9571b9a54"
    },
    [ordered]@{
        dataset = "eRASS1 optical cluster and BCG catalog"
        citation = "Kluge et al., Astronomy & Astrophysics 688, A210 (2024)"
        url = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/688/A210/tablee1.dat.gz"
        directory = "erass1_optical_clusters"
        filename = "tablee1.dat.gz"
        manifest_filename = "provenance_catalog.json"
        expected_bytes = 1469608
        expected_sha256 = "89b64b588301787b5eb6388bf82214ed83288b7b6d8f7d30309e5266d9f456d8"
    },
    [ordered]@{
        dataset = "eRASS1 optical cluster and BCG catalog data dictionary"
        citation = "CDS catalog J/A+A/688/A210 ReadMe"
        url = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/688/A210/ReadMe"
        directory = "erass1_optical_clusters"
        filename = "ReadMe"
        manifest_filename = "provenance_readme.json"
        expected_bytes = 12848
        expected_sha256 = "51d7edb4385ec8f3c5572e2c83f13cfc7f7eeda8a3dbae33e9a6fd3f250f978c"
    },
    [ordered]@{
        dataset = "SDSS DR17 GEMA-VAC 2.0.2"
        citation = "Galaxy Environment for MaNGA value-added catalog"
        url = "https://data.sdss.org/sas/dr17/env/MANGA_GEMA/2.0.2/GEMA_2.0.2.fits"
        directory = "manga_gema_dr17"
        filename = "GEMA_2.0.2.fits"
        expected_bytes = 7223040
        expected_sha256 = "244e9286f225b9e1dbef73bb93e1101d840a2a62ec5dabf3aa16ec88cfef1597"
    },
    [ordered]@{
        dataset = "SPIDERS DR14 X-ray cluster BCG catalog v2.0"
        citation = "SDSS SPIDERS Brightest Cluster Galaxies value-added catalog"
        url = "https://data.sdss.org/sas/dr16/eboss/spiders/analysis/SpidersXclusterBCGs-v2.0.fits"
        directory = "spiders_clusters"
        filename = "SpidersXclusterBCGs-v2.0.fits"
        manifest_filename = "provenance_bcg_catalog.json"
        expected_bytes = 362880
        expected_sha256 = "31a3a3ca41443f9c6279b48d0ed4c02030b7e69aac80ff0bcd3b2a08f8d045d0"
    },
    [ordered]@{
        dataset = "SPIDERS DR16 RASS cluster catalog v3.0"
        citation = "SDSS SPIDERS spectroscopically confirmed X-ray clusters"
        url = "https://data.sdss.org/sas/dr16/eboss/spiders/analysis/catCluster-SPIDERS_RASS_CLUS-v3.0.fits"
        directory = "spiders_clusters"
        filename = "catCluster-SPIDERS_RASS_CLUS-v3.0.fits"
        manifest_filename = "provenance_cluster_catalog.json"
        expected_bytes = 708480
        expected_sha256 = "b8c716b561f6984f64ea6a2536344ffad2ddd69aff1c51f3052f3aba3a8247d0"
    },
    [ordered]@{
        dataset = "SDSS DR17 SPIDERS RASS cluster target/member catalog v1.1"
        citation = "Clerc et al., MNRAS 497, 3976 (2020); SDSS DR17 SPIDERS target catalog"
        url = "https://data.sdss.org/sas/dr17/eboss/spiders/target/spiderstargetClusters-SPIDERS_RASS_CLUS-v1.1.fits"
        directory = "spiders_clusters"
        filename = "spiderstargetClusters-SPIDERS_RASS_CLUS-v1.1.fits"
        manifest_filename = "provenance_target_catalog.json"
        expected_bytes = 9884160
        expected_sha256 = "fe8924ddf795ef92d42c80db3e76d364c8634623a9aaaa6d370883262413391d"
    },
    [ordered]@{
        dataset = "SDSS DR17 SPIDERS SEQUELS RASS cluster target/member catalog v1.0"
        citation = "Clerc et al., MNRAS 497, 3976 (2020); SDSS DR17 SPIDERS target catalog"
        url = "https://data.sdss.org/sas/dr17/eboss/spiders/target/spiderstargetSequelsClus-SPIDERS_RASS_CLUS-v1.0.fits"
        directory = "spiders_clusters"
        filename = "spiderstargetSequelsClus-SPIDERS_RASS_CLUS-v1.0.fits"
        manifest_filename = "provenance_sequels_target_catalog.json"
        expected_bytes = 696960
        expected_sha256 = "f7d593beca33dfd1cc005c0acefcc49660fa75016b55dd43fdf6de2088df635c"
    },
    [ordered]@{
        dataset = "MaNGA DynPop I JAM catalog"
        citation = "Zhu et al., MNRAS 522, 6326 (2023); Zenodo 10.5281/zenodo.17518315"
        url = "https://zenodo.org/api/records/17518315/files/SDSSDR17_MaNGA_JAM.fits/content"
        directory = "manga_dynpop"
        filename = "SDSSDR17_MaNGA_JAM.fits"
        manifest_filename = "provenance_catalog.json"
        expected_bytes = 19157760
        expected_sha256 = "04ca6385c646d1554f254ec7fb3227fcadb08d326d02a67e5863cac28bb8ce91"
    },
    [ordered]@{
        dataset = "MaNGA DynPop JAM v2 data model"
        citation = "MaNGA DynPop catalog data model; Zenodo 10.5281/zenodo.17518315"
        url = "https://zenodo.org/api/records/17518315/files/SDSSDR17_MaNGA_JAM_v2_datamodel.pdf/content"
        directory = "manga_dynpop"
        filename = "SDSSDR17_MaNGA_JAM_v2_datamodel.pdf"
        manifest_filename = "provenance_datamodel.json"
        expected_bytes = 131067
        expected_sha256 = "2587917a62624d47a897323787719af69c960efd070cb323f5508f5d2693f204"
    }
)

foreach ($item in $downloads) {
    $destination = Join-Path $project ("data\raw\" + $item.directory)
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    $target = Join-Path $destination $item.filename
    $needsDownload = $Force -or -not (Test-Path -LiteralPath $target)
    if (-not $needsDownload) {
        $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
        $needsDownload = (
            (Get-Item -LiteralPath $target).Length -ne $item.expected_bytes -or
            $existingHash -ne $item.expected_sha256
        )
    }
    if ($needsDownload) {
        $partial = "$target.partial"
        curl.exe -L --fail --silent --show-error --output $partial $item.url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $($item.url)"
        }
        $actualBytes = (Get-Item -LiteralPath $partial).Length
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $partial).Hash.ToLowerInvariant()
        if ($actualBytes -ne $item.expected_bytes -or $actualHash -ne $item.expected_sha256) {
            Remove-Item -LiteralPath $partial -Force
            throw "Unexpected file for $($item.filename): bytes=$actualBytes sha256=$actualHash"
        }
        Move-Item -LiteralPath $partial -Destination $target -Force
    }
    $manifest = [ordered]@{
        dataset = $item.dataset
        citation = $item.citation
        source_url = $item.url
        downloaded_utc = (Get-Date).ToUniversalTime().ToString("o")
        file = [ordered]@{
            path = $item.filename
            bytes = (Get-Item -LiteralPath $target).Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
        }
    }
    $manifestFilename = "provenance.json"
    if ($item.Contains("manifest_filename")) {
        $manifestFilename = $item.manifest_filename
    }
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (
        Join-Path $destination $manifestFilename
    ) -Encoding utf8
    Write-Output "$($item.dataset) is ready at $target"
}
