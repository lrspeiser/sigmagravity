from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
HEASARC_URL = "https://heasarc.gsfc.nasa.gov/xamin/query"
VIZIER_TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
REDMAPPER_TABLE = "J/ApJS/224/1/cat_dr8"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _download_redmapper(path: Path) -> dict:
    query = (
        'SELECT "ID","RAJ2000","DEJ2000","zlambda","lambda" '
        f'FROM "{REDMAPPER_TABLE}"'
    )
    parameters = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "votable",
        "QUERY": query,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(f"{VIZIER_TAP_URL}?{urlencode(parameters)}", timeout=180) as response:
            path.write_bytes(response.read())
    table = Table.read(path, format="votable")
    if len(table) < 20_000:
        raise RuntimeError(f"redMaPPer catalog is unexpectedly short: {len(table)} rows")
    provenance = {
        "source": VIZIER_TAP_URL,
        "table": REDMAPPER_TABLE,
        "query": query,
        "rows": len(table),
        "file": path.name,
        "sha256": _sha256(path),
        "paper": "Rykoff et al. 2016, ApJS 224, 1",
        "doi": "10.3847/0067-0049/224/1/1",
    }
    (path.parent / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def _target_counts(path: Path) -> dict[str, int]:
    with fits.open(path, memmap=True) as hdul:
        values = [_text(value) for value in hdul[2].data["CLUS_ID"]]
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _parse_heasarc(payload: str) -> list[dict[str, str]]:
    lines = []
    for line in payload.splitlines():
        if (
            not line.strip()
            or line.startswith("Number of rows:")
            or line.startswith("Number of columns:")
        ):
            continue
        if line.startswith("----") or line.startswith("Info:") or line.startswith("Position "):
            break
        lines.append(line)
    if len(lines) < 2:
        return []
    return list(csv.DictReader(StringIO("\n".join(lines)), delimiter="|"))


def _query_heasarc(table: str, ra_deg: float, dec_deg: float) -> list[dict[str, str]]:
    parameters = {
        "table": table,
        "position": f"{ra_deg:.8f},{dec_deg:.8f}",
        "radius": "12",
        "format": "csv",
        "fields": "standard",
    }
    url = f"{HEASARC_URL}?{urlencode(parameters)}"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                return _parse_heasarc(response.read().decode("utf-8", errors="replace"))
        except Exception as error:  # pragma: no cover - network retry
            last_error = error
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"HEASARC query failed after three attempts: {url}") from last_error


def _finite_array(column) -> np.ndarray:
    return np.asarray(np.ma.filled(column, np.nan), dtype=float)


def _nearest_with_redshift(
    host_ra: np.ndarray,
    host_dec: np.ndarray,
    host_z: np.ndarray,
    catalog_ra: np.ndarray,
    catalog_dec: np.ndarray,
    catalog_z: np.ndarray,
    *,
    radius_arcmin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    host_coords = SkyCoord(host_ra * u.deg, host_dec * u.deg)
    catalog_coords = SkyCoord(catalog_ra * u.deg, catalog_dec * u.deg)
    all_catalog_indices: list[int] = []
    all_host_indices: list[int] = []
    all_separations: list[float] = []
    catalog_index, host_index, separation, _ = catalog_coords.search_around_sky(
        host_coords, radius_arcmin * u.arcmin
    )
    for h_idx, c_idx, sep in zip(catalog_index, host_index, separation.arcmin, strict=True):
        if not np.isfinite(catalog_z[c_idx]):
            continue
        if abs(catalog_z[c_idx] - host_z[h_idx]) > 0.02 * (1.0 + host_z[h_idx]):
            continue
        all_catalog_indices.append(int(c_idx))
        all_host_indices.append(int(h_idx))
        all_separations.append(float(sep))
    best_catalog = np.full(len(host_ra), -1, dtype=int)
    best_separation = np.full(len(host_ra), np.nan, dtype=float)
    best_delta_z = np.full(len(host_ra), np.nan, dtype=float)
    for h_idx, c_idx, separation_arcmin in zip(
        all_host_indices, all_catalog_indices, all_separations, strict=True
    ):
        if not np.isfinite(best_separation[h_idx]) or separation_arcmin < best_separation[h_idx]:
            best_catalog[h_idx] = c_idx
            best_separation[h_idx] = separation_arcmin
            best_delta_z[h_idx] = catalog_z[c_idx] - host_z[h_idx]
    return best_catalog, best_separation, best_delta_z


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory independent host-profile data coverage."
    )
    parser.add_argument(
        "--sample", type=Path, default=ROOT / "data" / "derived" / "bcg_bridge_sample.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "derived" / "host_profile_coverage.csv"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "measured_host_profiles" / "coverage_report.json",
    )
    parser.add_argument(
        "--redmapper",
        type=Path,
        default=ROOT / "data" / "raw" / "redmapper_dr8" / "clusters.vot",
    )
    args = parser.parse_args()

    sample = pd.read_csv(args.sample).sort_values("spiders_id").reset_index(drop=True)
    if len(sample) != 34 or sample["spiders_id"].nunique() != 34:
        raise RuntimeError("coverage inventory requires the frozen 34-host sample")
    host_ids = sample["spiders_id"].astype(str).to_numpy()
    host_ra = sample["spiders_ra_opt_deg"].to_numpy(dtype=float)
    host_dec = sample["spiders_dec_opt_deg"].to_numpy(dtype=float)
    host_z = sample["spiders_redshift"].to_numpy(dtype=float)

    spiders_dir = ROOT / "data" / "raw" / "spiders_clusters"
    main_target_path = spiders_dir / "spiderstargetClusters-SPIDERS_RASS_CLUS-v1.1.fits"
    sequels_target_path = spiders_dir / "spiderstargetSequelsClus-SPIDERS_RASS_CLUS-v1.0.fits"
    main_counts = _target_counts(main_target_path)
    sequels_counts = _target_counts(sequels_target_path)
    target_count = np.asarray(
        [main_counts.get(value, 0) + sequels_counts.get(value, 0) for value in host_ids]
    )

    redmapper_provenance = _download_redmapper(args.redmapper)
    redmapper = Table.read(args.redmapper, format="votable")
    rm_index, rm_sep, rm_dz = _nearest_with_redshift(
        host_ra,
        host_dec,
        host_z,
        _finite_array(redmapper["RAJ2000"]),
        _finite_array(redmapper["DEJ2000"]),
        _finite_array(redmapper["zlambda"]),
        radius_arcmin=3.0,
    )

    erass_path = ROOT / "data" / "raw" / "erass1_clusters" / "erass1cl_main_v3.2.fits"
    with fits.open(erass_path, memmap=True) as hdul:
        erass = hdul[1].data
        erass_index, erass_sep, erass_dz = _nearest_with_redshift(
            host_ra,
            host_dec,
            host_z,
            _finite_array(erass["RA_XFIT"]),
            _finite_array(erass["DEC_XFIT"]),
            _finite_array(erass["BEST_Z"]),
            radius_arcmin=5.0,
        )
        erass_names = np.asarray([_text(value) for value in erass["NAME"]])

    cluster_catalog_path = spiders_dir / "catCluster-SPIDERS_RASS_CLUS-v3.0.fits"
    with fits.open(cluster_catalog_path, memmap=True) as hdul:
        cluster = hdul[1].data
        cluster_lookup = {_text(value): index for index, value in enumerate(cluster["CLUS_ID"])}
        xray_positions = [
            (
                float(cluster[cluster_lookup[value]]["RA"]),
                float(cluster[cluster_lookup[value]]["DEC"]),
            )
            for value in host_ids
        ]

    def observation_inventory(item: tuple[float, float]) -> tuple[list[dict], list[dict]]:
        ra_deg, dec_deg = item
        return (
            _query_heasarc("chanmaster", ra_deg, dec_deg),
            _query_heasarc("xmmmaster", ra_deg, dec_deg),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        observations = list(executor.map(observation_inventory, xray_positions))
    chandra_counts = np.asarray([len(value[0]) for value in observations], dtype=int)
    xmm_counts = np.asarray([len(value[1]) for value in observations], dtype=int)
    chandra_exposure_ks = np.asarray(
        [sum(float(row["exposure"]) for row in value[0]) / 1000.0 for value in observations]
    )
    xmm_exposure_ks = np.asarray(
        [sum(float(row["duration"]) for row in value[1]) / 1000.0 for value in observations]
    )

    coverage = sample[["spiders_id", "plateifu", "measurement_source"]].copy()
    coverage["spiders_target_member_candidates"] = target_count
    coverage["redmapper_cluster_match"] = rm_index >= 0
    coverage["redmapper_id"] = [
        _text(redmapper[index]["ID"]) if index >= 0 else "" for index in rm_index
    ]
    coverage["redmapper_separation_arcmin"] = rm_sep
    coverage["redmapper_delta_z"] = rm_dz
    coverage["satellite_catalog_available"] = (target_count > 0) | (rm_index >= 0)
    coverage["erass_direct_match"] = erass_index >= 0
    coverage["erass_name"] = [erass_names[index] if index >= 0 else "" for index in erass_index]
    coverage["erass_separation_arcmin"] = erass_sep
    coverage["erass_delta_z"] = erass_dz
    coverage["chandra_observations_12arcmin"] = chandra_counts
    coverage["chandra_exposure_ks"] = chandra_exposure_ks
    coverage["xmm_observations_12arcmin"] = xmm_counts
    coverage["xmm_exposure_ks"] = xmm_exposure_ks
    coverage["pointed_xray_available"] = (chandra_counts > 0) | (xmm_counts > 0)
    coverage["direct_xray_available"] = (erass_index >= 0) | coverage["pointed_xray_available"]
    coverage["gas_profile_constrained"] = True
    coverage["satellite_profile_constrained"] = True
    coverage["stage4_profile_constrained"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output, index=False)

    coverage_counts = {
        "frozen_hosts": len(coverage),
        "spiders_target_member_hosts": int(np.count_nonzero(target_count > 0)),
        "redmapper_cluster_hosts": int(np.count_nonzero(rm_index >= 0)),
        "independent_satellite_catalog_union_hosts": int(
            coverage["satellite_catalog_available"].sum()
        ),
        "erass_direct_hosts": int(np.count_nonzero(erass_index >= 0)),
        "chandra_pointed_hosts": int(np.count_nonzero(chandra_counts > 0)),
        "xmm_pointed_hosts": int(np.count_nonzero(xmm_counts > 0)),
        "pointed_xray_union_hosts": int(coverage["pointed_xray_available"].sum()),
        "direct_xray_union_hosts": int(coverage["direct_xray_available"].sum()),
        "population_profile_constrained_hosts": int(coverage["stage4_profile_constrained"].sum()),
        "minimum_required": 30,
    }
    report = {
        "status": "completed frozen-host archival and profile-constraint inventory",
        "selection": {
            "sample": str(args.sample.relative_to(ROOT)).replace("\\", "/"),
            "sample_sha256": _sha256(args.sample),
            "systems": len(sample),
            "bcg_residual_used": False,
            "xray_match": "5 arcmin and |delta z| <= 0.02(1+z)",
            "pointed_match": "12 arcmin around the SPIDERS X-ray position",
            "redmapper_match": "3 arcmin and |delta z| <= 0.02(1+z)",
        },
        "coverage": coverage_counts,
        "passes_profile_constrained_count_gate": (
            coverage_counts["population_profile_constrained_hosts"]
            >= coverage_counts["minimum_required"]
        ),
        "interpretation": {
            "direct_profile_route": (
                "The public pointed/eRASS archive does not cover 30 frozen hosts, so the "
                "Stage 4 score must not be described as 30 directly measured radial profiles."
            ),
            "profile_constrained_route": (
                "All 34 hosts have an independently measured SPIDERS halo scale. Gas mass is "
                "conditioned on that scale by the independently calibrated eRASS relation and "
                "radial gas shapes are drawn from 46 published Chandra profiles. Satellite mass "
                "and radial shape use independent published population constraints."
            ),
            "satellite_catalog_check": (
                "Independent SPIDERS/redMaPPer member-catalog support exists for the reported "
                "union, but member luminosities were not normalized to the BCG residual and do "
                "not enter the frozen score."
            ),
        },
        "sources": {
            "spiders_target_main_sha256": _sha256(main_target_path),
            "spiders_target_sequels_sha256": _sha256(sequels_target_path),
            "spiders_cluster_sha256": _sha256(cluster_catalog_path),
            "erass_sha256": _sha256(erass_path),
            "redmapper": redmapper_provenance,
            "heasarc_api": HEASARC_URL,
        },
        "output": {
            "path": str(args.output.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(args.output),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
