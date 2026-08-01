param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\data\raw\r1_lens_nuisance_inputs")
)

$ErrorActionPreference = "Stop"
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null

$downloads = @(
    @{
        System = "A383"; Kind = "lenstool_params";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/a383/models/lenstool/hlsp_clash_model_a383_lenstool_v1_params.txt";
        File = "a383/hlsp_clash_model_a383_lenstool_v1_params.txt"
    },
    @{
        System = "A383"; Kind = "lenstool_readme";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/a383/models/lenstool/hlsp_clash_model_a383_lenstool_v1_readme.txt";
        File = "a383/hlsp_clash_model_a383_lenstool_v1_readme.txt"
    },
    @{
        System = "A383"; Kind = "arc_catalog";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/a383/models/lenstool/catalogs/hlsp_clash_model_a383_lenstool_v1-arcs.txt";
        File = "a383/hlsp_clash_model_a383_lenstool_v1-arcs.txt"
    },
    @{
        System = "A383"; Kind = "arc_barycenters";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/a383/models/lenstool/catalogs/hlsp_clash_model_a383_lenstool_v1_arcs-barycenter.txt";
        File = "a383/hlsp_clash_model_a383_lenstool_v1_arcs-barycenter.txt"
    },
    @{
        System = "A383"; Kind = "cluster_member_catalog";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/a383/models/lenstool/catalogs/hlsp_clash_model_a383_lenstool_v1_cluster-members.txt";
        File = "a383/hlsp_clash_model_a383_lenstool_v1_cluster-members.txt"
    },
    @{
        System = "A383"; Kind = "source_barycenters";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/a383/models/lenstool/catalogs/hlsp_clash_model_a383_lenstool_v1_sources-barycenter.txt";
        File = "a383/hlsp_clash_model_a383_lenstool_v1_sources-barycenter.txt"
    },
    @{
        System = "A383"; Kind = "hst_photometric_catalog";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/a383/catalogs/hst/hlsp_clash_hst_acs-ir_a383_cat.txt";
        File = "a383/hlsp_clash_hst_acs-ir_a383_cat.txt"
    },
    @{
        System = "A383"; Kind = "hst_catalog_readme";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/a383/catalogs/hst/hlsp_clash_hst_catalog_readme.txt";
        File = "a383/hlsp_clash_hst_catalog_readme.txt"
    },
    @{
        System = "MS2137"; Kind = "zitrin_ltm_params";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/ms2137/models/zitrin/ltm-gauss/v2/hlsp_clash_model_ms2137_zitrin-ltm-gauss_v2_params.txt";
        File = "ms2137/hlsp_clash_model_ms2137_zitrin-ltm-gauss_v2_params.txt"
    },
    @{
        System = "MS2137"; Kind = "zitrin_ltm_readme";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/ms2137/models/zitrin/ltm-gauss/v2/hlsp_clash_model_ms2137_zitrin-ltm-gauss_v2_readme.txt";
        File = "ms2137/hlsp_clash_model_ms2137_zitrin-ltm-gauss_v2_readme.txt"
    },
    @{
        System = "MS2137"; Kind = "hst_photometric_catalog";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/ms2137/catalogs/hst/hlsp_clash_hst_acs-ir_ms2137_cat.txt";
        File = "ms2137/hlsp_clash_hst_acs-ir_ms2137_cat.txt"
    },
    @{
        System = "MS2137"; Kind = "hst_catalog_readme";
        Url = "https://archive.stsci.edu/missions/hlsp/clash/ms2137/catalogs/hst/hlsp_clash_hst_catalog_readme.txt";
        File = "ms2137/hlsp_clash_hst_catalog_readme.txt"
    }
)

$records = @()
foreach ($item in $downloads) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $resolvedOutput $item.File))
    if (-not $target.StartsWith($resolvedOutput, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Resolved download target escapes output directory: $target"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    if (-not (Test-Path -LiteralPath $target)) {
        Invoke-WebRequest -Uri $item.Url -OutFile $target
    }
    $file = Get-Item -LiteralPath $target
    $records += [ordered]@{
        system = $item.System
        product_kind = $item.Kind
        url = $item.Url
        local_path = $target.Substring($resolvedOutput.Length).TrimStart('\', '/')
        bytes = $file.Length
        sha256 = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$provenance = [ordered]@{
    audit_version = "R1A1-nuisance-inputs-0.1"
    source_archive = "MAST CLASH HLSP"
    source_index = "https://archive.stsci.edu/prepds/clash/"
    downloaded_utc = [DateTime]::UtcNow.ToString("o")
    records = $records
}
$provenance | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $resolvedOutput "provenance.json") -Encoding utf8
$provenance | ConvertTo-Json -Depth 6
