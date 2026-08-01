#!/usr/bin/env python3
"""Audit the residual-blind SDSS J0946+1006 rank-three feasibility gate."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_j0946_jackpot_feasibility_protocol.json"
SOURCE_ROOT = ROOT / "data/raw/r1_j0946_primary_sources"
PROVENANCE_PATH = SOURCE_ROOT / "provenance.json"
INVENTORY_PATH = ROOT / "data/derived/r1_j0946_archive_inventory.csv"
QUEUE_PATH = ROOT / "data/derived/r1_new_rank3_candidate_queue.csv"
REPORT_PATH = ROOT / "results/r1_j0946_jackpot_feasibility/report.json"
QUEUE_COLUMNS = [
    "system",
    "alias",
    "selection_blind",
    "distinct_spectroscopic_source_planes",
    "published_ring_scales_arcsec",
    "accepted_dynamics_outer_radius_arcsec",
    "ring_scales_inside_accepted_support",
    "pre_fit_ring_scale_rank_upper_bound",
    "extended_surface_brightness_constraints_published",
    "published_radial_magnification_sensitivity",
    "pre_pixel_structural_rank_upper_bound",
    "full_image_level_structural_rank",
    "raw_archives_public",
    "observable_level_normalized_likelihood_public",
    "counts_toward_ten_system_target",
    "disposition",
    "primary_blocker",
    "next_authorized_stage",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upsert_candidate_queue(row: dict[str, object]) -> None:
    if QUEUE_PATH.exists():
        queue = pd.read_csv(QUEUE_PATH, keep_default_na=False)
        queue = queue.loc[queue["system"] != row["system"]].copy()
    else:
        queue = pd.DataFrame(columns=QUEUE_COLUMNS)
    for column in QUEUE_COLUMNS:
        if column not in queue:
            queue[column] = ""
    queue = pd.concat([queue, pd.DataFrame([row])], ignore_index=True, sort=False)
    queue = queue.reindex(columns=QUEUE_COLUMNS).fillna("")
    queue = queue.sort_values("system", kind="stable").reset_index(drop=True)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(QUEUE_PATH, index=False, lineterminator="\n")


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "sigmagravity-observable-audit/0.1"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def query_eso_muse(proposal_id: str) -> pd.DataFrame:
    fields = [
        "dp_id",
        "target_name",
        "s_ra",
        "s_dec",
        "t_exptime",
        "dataproduct_type",
        "calib_level",
        "obs_collection",
        "instrument_name",
        "proposal_id",
        "access_url",
        "access_estsize",
        "em_min",
        "em_max",
    ]
    query = (
        f"SELECT {','.join(fields)} FROM ivoa.ObsCore "
        f"WHERE proposal_id LIKE '{proposal_id}%'"
    )
    url = "https://archive.eso.org/tap_obs/sync?" + urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    )
    with urlopen(url, timeout=60) as response:
        table = pd.read_csv(io.BytesIO(response.read()))
    return table.sort_values(["calib_level", "dp_id"], kind="stable").reset_index(drop=True)


def query_hst(config: dict) -> pd.DataFrame:
    target = config["target"]
    coordinate = SkyCoord(target["ra_deg"] * u.deg, target["dec_deg"] * u.deg)
    table = Observations.query_region(coordinate, radius=5 * u.arcsec)
    table = table[table["obs_collection"] == "HST"]
    rows: list[dict[str, object]] = []
    for expected in config["archive_targets"]["hst"]:
        matches = table[
            (table["proposal_id"].astype(str) == expected["proposal_id"])
            & (table["instrument_name"].astype(str) == expected["instrument"])
            & (table["filters"].astype(str) == expected["filter"])
        ]
        if len(matches) == 0:
            rows.append(
                {
                    "archive": "HST_MAST",
                    "product_id": "",
                    "proposal_id": expected["proposal_id"],
                    "target_name": "",
                    "instrument": expected["instrument"],
                    "filter": expected["filter"],
                    "calibration_level": "",
                    "exposure_seconds": float("nan"),
                    "public": False,
                    "metadata_match": False,
                }
            )
            continue
        row = matches.to_pandas().sort_values(["t_exptime", "obsid"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "archive": "HST_MAST",
                "product_id": str(row["obsid"]),
                "proposal_id": str(row["proposal_id"]),
                "target_name": str(row["target_name"]),
                "instrument": str(row["instrument_name"]),
                "filter": str(row["filters"]),
                "calibration_level": str(row["calib_level"]),
                "exposure_seconds": float(row["t_exptime"]),
                "public": str(row["dataRights"]).upper() == "PUBLIC",
                "metadata_match": True,
            }
        )
    return pd.DataFrame(rows)


def source_checks(config: dict) -> tuple[dict[str, bool], dict[str, dict[str, object]]]:
    files = {
        "turner2024": SOURCE_ROOT / "2401.08771/main.tex",
        "collett_smith2020": SOURCE_ROOT / "2004.00649/triplewhammy.tex",
        "smith_collett2021": SOURCE_ROOT / "2104.12790/jackpot_s2_redshift.tex",
        "ballard2024": SOURCE_ROOT / "2309.04535/main.tex",
    }
    text = {key: path.read_text(encoding="utf-8", errors="replace") for key, path in files.items()}
    checks = {
        "turner_muse_program_and_5p2h": "0102.A-0950" in text["turner2024"] and "5.2 hour" in text["turner2024"],
        "turner_measured_to_2p7_arcsec": "2.7\\,arcsec" in text["turner2024"],
        "turner_53_voronoi_bins": "binned into 53 bins" in text["turner2024"],
        "turner_excludes_nine_outer_bins_and_stops_at_1p95": "exclusion of our nine outermost bins" in text["turner2024"] and "to 1.95\\,arcsec" in text["turner2024"],
        "three_published_ring_scales": all(token in text["collett_smith2020"] for token in ("1.4\\,arcsec", "2.1\\,arcsec", "2.5\\,arcsec")),
        "third_source_image_radii_1p35_and_3p56": "3.56\\,arcsec" in text["collett_smith2020"] and "1.35\\,arcsec" in text["collett_smith2020"],
        "second_source_spectroscopic_z_2p035": "2.035" in text["smith_collett2021"] and "spectroscopic determination" in text["smith_collett2021"],
        "third_source_spectroscopic_z_5p975": "5.975" in text["collett_smith2020"],
        "hst_programs_10886_and_11202": "Programme 10886" in text["collett_smith2020"] and "Programme 11202" in text["collett_smith2020"],
        "ballard_supporting_products_request_or_archives_only": "Supporting research data are available on request" in text["ballard2024"] and "HST and VLT archives" in text["ballard2024"],
    }
    metadata = {
        key: {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for key, path in files.items()
    }
    return checks, metadata


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8-sig"))
    checks, source_metadata = source_checks(config)

    provenance_checks = {}
    for entry in provenance["files"]:
        archive = SOURCE_ROOT / entry["archive_path"]
        provenance_checks[entry["arxiv_id"]] = bool(
            archive.exists()
            and archive.stat().st_size == int(entry["archive_bytes"])
            and sha256(archive) == entry["archive_sha256"]
        )

    eso = query_eso_muse(config["archive_targets"]["eso_muse"]["proposal_id"])
    eso_inventory = pd.DataFrame(
        {
            "archive": "ESO",
            "product_id": eso["dp_id"].astype(str),
            "proposal_id": eso["proposal_id"].astype(str),
            "target_name": eso["target_name"].astype(str),
            "instrument": eso["instrument_name"].astype(str),
            "filter": "",
            "calibration_level": eso["calib_level"].astype(str),
            "exposure_seconds": eso["t_exptime"].astype(float),
            "public": True,
            "metadata_match": (
                (eso["instrument_name"].astype(str) == "MUSE")
                & (eso["dataproduct_type"].astype(str) == "cube")
            ),
        }
    )
    hst_inventory = query_hst(config)
    inventory = pd.concat([eso_inventory, hst_inventory], ignore_index=True)
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(INVENTORY_PATH, index=False)

    zenodo = fetch_json("https://zenodo.org/api/records/17014179")
    zenodo_files = [
        {"name": item["key"], "bytes": int(item["size"]), "checksum": item["checksum"]}
        for item in zenodo["files"]
    ]
    github = fetch_json("https://api.github.com/repos/astroskylee/Jackpot_GNFW")
    github_commit = fetch_json("https://api.github.com/repos/astroskylee/Jackpot_GNFW/commits/main")

    support = config["published_dynamics_support"]["accepted_axisymmetric_model_outer_radius_arcsec"]
    ring_radii = [
        item["ring_radius_arcsec"]
        for item in config["published_lens_geometry"]["source_planes_sorted_by_ring_radius"]
    ]
    inside = [radius for radius in ring_radii if radius <= support]
    outside = [radius for radius in ring_radii if radius > support]
    geometry = config["frozen_geometry_gate"]
    geometry_gate = bool(
        len(inside) >= geometry["minimum_rings_inside_accepted_dynamics_support"]
        and len(ring_radii) >= geometry["minimum_distinct_ring_radii"]
        and config["published_lens_geometry"]["distinct_source_planes"]
        >= geometry["minimum_distinct_spectroscopic_source_planes"]
    )
    archive_gate = bool(
        len(eso_inventory) >= 1
        and int((eso["calib_level"] == 3).sum()) >= 1
        and len(hst_inventory) == len(config["archive_targets"]["hst"])
        and hst_inventory["metadata_match"].all()
        and hst_inventory["public"].all()
    )
    source_gate = all(checks.values()) and all(provenance_checks.values())

    no_normalized_likelihood = bool(
        checks["ballard_supporting_products_request_or_archives_only"]
        and "posterior" not in {item["name"].lower() for item in zenodo_files}
    )
    upsert_candidate_queue(
        {
            "system": config["system"],
            "alias": config["alias"],
            "selection_blind": True,
            "distinct_spectroscopic_source_planes": config["published_lens_geometry"]["distinct_source_planes"],
            "published_ring_scales_arcsec": ";".join(str(value) for value in ring_radii),
            "accepted_dynamics_outer_radius_arcsec": support,
            "ring_scales_inside_accepted_support": len(inside),
            "pre_fit_ring_scale_rank_upper_bound": len(inside),
            "extended_surface_brightness_constraints_published": False,
            "published_radial_magnification_sensitivity": False,
            "pre_pixel_structural_rank_upper_bound": len(inside),
            "full_image_level_structural_rank": "not_established",
            "raw_archives_public": archive_gate,
            "observable_level_normalized_likelihood_public": not no_normalized_likelihood,
            "counts_toward_ten_system_target": False,
            "disposition": "rank_one_repair_candidate_not_rank_three_promotion",
            "primary_blocker": "published accepted dynamics stop at 1.95 arcsec; 2.1 and 2.5 arcsec ring scales are outside support",
            "next_authorized_stage": "none_without_outer_dynamics_repair",
        }
    )

    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "science_pixels_downloaded_or_inspected": False,
        "system": config["system"],
        "inputs": {
            "protocol": {"path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(CONFIG_PATH)},
            "source_provenance": {"path": str(PROVENANCE_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(PROVENANCE_PATH)},
            "primary_source_files": source_metadata,
        },
        "primary_source_checks": checks,
        "primary_source_archive_hash_checks": provenance_checks,
        "archive_inventory": {
            "eso_muse_product_count": int(len(eso_inventory)),
            "eso_muse_calibration_level_2_count": int((eso["calib_level"] == 2).sum()),
            "eso_muse_calibration_level_3_count": int((eso["calib_level"] == 3).sum()),
            "eso_level_3_exposure_seconds": [float(value) for value in eso.loc[eso["calib_level"] == 3, "t_exptime"]],
            "hst_expected_public_observations_found": int(hst_inventory["metadata_match"].sum()),
            "hst_expected_public_observations_required": len(config["archive_targets"]["hst"]),
            "output": str(INVENTORY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "output_sha256": sha256(INVENTORY_PATH),
        },
        "public_supporting_products": {
            "ballard2024_disposition": "supporting research data on request; raw HST and VLT data in archives",
            "zenodo_17014179_title": zenodo["metadata"]["title"],
            "zenodo_files": zenodo_files,
            "zenodo_disposition": "frozen PyAutoLens code and supplementary figures; no normalized observable likelihood or chains listed",
            "jackpot_gnfw_repository": {
                "url": github["html_url"],
                "public": not github["private"],
                "default_branch": github["default_branch"],
                "latest_commit_sha": github_commit["sha"],
                "repository_size_kib": github["size"],
                "disposition": "public 2026 analysis scripts; paper says posterior samples and MCMC chains remain available only on request",
            },
            "normalized_theory_neutral_likelihood_publicly_identified": not no_normalized_likelihood,
        },
        "geometry": {
            "accepted_dynamics_outer_radius_arcsec": support,
            "published_ring_scales_arcsec": ring_radii,
            "ring_scales_inside_accepted_support_arcsec": inside,
            "ring_scales_outside_accepted_support_arcsec": outside,
            "current_pre_fit_ring_scale_rank_upper_bound": len(inside),
            "full_image_level_structural_rank": "not_established",
            "minimum_required_rank": geometry["minimum_pre_fit_structural_radial_rank_upper_bound"],
        },
        "gates": {
            "primary_sources_and_hashes_passed": source_gate,
            "public_raw_archive_metadata_passed": archive_gate,
            "three_ring_scales_inside_accepted_dynamics_support_passed": geometry_gate,
            "public_normalized_observable_likelihood_identified": not no_normalized_likelihood,
            "rank_three_candidate_admission_passed": source_gate and archive_gate and geometry_gate and not no_normalized_likelihood,
        },
        "decision": "retain_as_rank_one_repair_candidate_not_a_ten_system_promotion",
        "ten_system_effect": {
            "previous_structural_ceiling": 3,
            "updated_structural_ceiling": 3,
            "minimum_new_rank_three_systems_still_required": 7,
        },
        "next_action": "Do not download J0946 science pixels under this protocol. Either freeze a separate outer-dynamics repair whose validity support reaches at least 2.5 arcsec, or replace J0946 with another residual-blind candidate whose accepted dynamics already spans at least three lensing radii.",
        "outputs": {
            "archive_inventory": str(INVENTORY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "candidate_queue": str(QUEUE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "candidate_queue_sha256": sha256(QUEUE_PATH),
        },
        "authorization": {
            "download_science_pixels": False,
            "inspect_science_pixels": False,
            "count_toward_ten_system_target": False,
            "freeze_ten_system_sample": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
