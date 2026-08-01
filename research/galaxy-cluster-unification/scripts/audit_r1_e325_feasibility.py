#!/usr/bin/env python3
"""Audit ESO 325-G004 before any HST or MUSE science-array download."""

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
CONFIG_PATH = ROOT / "configs/r1_e325_feasibility_protocol.json"
SOURCE_ROOT = ROOT / "data/raw/r1_e325_primary_source"
PROVENANCE_PATH = SOURCE_ROOT / "provenance.json"
INVENTORY_PATH = ROOT / "data/derived/r1_e325_archive_inventory.csv"
QUEUE_PATH = ROOT / "data/derived/r1_new_rank3_candidate_queue.csv"
REPORT_PATH = ROOT / "results/r1_e325_feasibility/report.json"
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


def query_hst(config: dict) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    target = config["target"]
    coordinate = SkyCoord(target["ra_deg"] * u.deg, target["dec_deg"] * u.deg)
    table = Observations.query_region(coordinate, radius=8 * u.arcsec).to_pandas()
    table = table.loc[table["obs_collection"].astype(str) == "HST"].copy()
    inventory_rows: list[dict[str, object]] = []
    group_summary: list[dict[str, object]] = []
    for expected in config["archive_targets"]["hst"]:
        matches = table.loc[
            (table["proposal_id"].astype(str) == expected["proposal_id"])
            & table["instrument_name"].astype(str).str.contains(
                expected["instrument_contains"], regex=False
            )
            & (table["filters"].astype(str) == expected["filter"])
        ].copy()
        if len(matches):
            matches = matches.sort_values(["obsid", "t_exptime"], kind="stable")
            matches = matches.drop_duplicates(subset=["obsid"], keep="last")
        public = bool(
            len(matches)
            and matches["dataRights"].astype(str).str.upper().eq("PUBLIC").all()
        )
        representative = matches.sort_values("t_exptime", ascending=False, kind="stable")
        selected_indices: list[int] = []
        total_exposure = 0.0
        for index, row in representative.iterrows():
            if total_exposure >= expected["minimum_public_exposure_seconds"]:
                break
            selected_indices.append(index)
            total_exposure += float(row["t_exptime"])
        representative_ids = representative.loc[selected_indices, "obsid"].astype(str).tolist()
        group_id = f"{expected['proposal_id']}:{expected['filter']}"
        group_summary.append(
            {
                "group": group_id,
                "proposal_id": expected["proposal_id"],
                "filter": expected["filter"],
                "metadata_rows_found": int(len(matches)),
                "representative_observation_count": int(len(representative_ids)),
                "representative_observation_ids": representative_ids,
                "total_exposure_seconds": total_exposure,
                "minimum_public_exposure_seconds": expected["minimum_public_exposure_seconds"],
                "public": public,
                "metadata_and_exposure_gate_pass": bool(
                    public and total_exposure >= expected["minimum_public_exposure_seconds"]
                ),
            }
        )
        if len(matches) == 0:
            inventory_rows.append(
                {
                    "archive": "HST_MAST",
                    "product_id": "",
                    "proposal_id": expected["proposal_id"],
                    "target_name": "",
                    "instrument": expected["instrument_contains"],
                    "filter": expected["filter"],
                    "calibration_level": "",
                    "exposure_seconds": 0.0,
                    "public": False,
                    "metadata_match": False,
                    "expected_group": group_id,
                    "science_role": expected["purpose"],
                }
            )
            continue
        for _, row in matches.iterrows():
            inventory_rows.append(
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
                    "expected_group": group_id,
                    "science_role": expected["purpose"],
                }
            )
    return pd.DataFrame(inventory_rows), group_summary


def source_checks() -> tuple[dict[str, bool], dict[str, object]]:
    source = SOURCE_ROOT / "1806.08300/ScienceFormat3.tex"
    text = source.read_text(encoding="utf-8", errors="replace")
    checks = {
        "coordinates_lens_and_source_redshifts": all(
            token in text for token in ("z_l=0.035", "13:43:33.2", "38:10:34", "z_s =2.1")
        ),
        "einstein_radius_2p95_arcsec": "2.95 arcsecond radius" in text,
        "extended_arcs_constrain_radial_magnification": (
            "additional constraints on the radial magnification across the image plane" in text
        ),
        "muse_program_and_public_archive": (
            "097.A-0987(A)" in text and "This data is available at the ESO archive" in text
        ),
        "muse_exposure_pixel_psf_and_kinematic_sampling": all(
            token in text
            for token in (
                "Five on-source exposures and two sky exposures each of 330 seconds",
                "0.2$''$ spatial pixels",
                "0.57''$",
                "binned to $0.6''$ pixels",
            )
        ),
        "accepted_dynamics_support_central_4_arcsec": (
            "model only the central 4 arcseconds of kinematic data" in text
        ),
        "hst_programs_filters_and_exposures": all(
            token in text
            for token in (
                "18900 sec observation in F814W",
                "4800 sec observation in F475W",
                "GO 10429",
                "GO 10710",
            )
        ),
        "extended_arc_pixels_and_source_grid": (
            "fluxes in thousands of pixels" in text and "adaptive grid of 80 by 80 square pixels" in text
        ),
        "public_pylens_code_declared": "github.com/tcollett/pylens" in text,
        "published_model_is_not_theory_neutral": (
            "simultaneously fit a 20 parameter model" in text
            and "final parameter of our model is $\\gamma$" in text
        ),
    }
    metadata = {
        "path": str(source.relative_to(ROOT)).replace("\\", "/"),
        "bytes": source.stat().st_size,
        "sha256": sha256(source),
    }
    return checks, metadata


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


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8-sig"))
    checks, source_metadata = source_checks()
    archive = SOURCE_ROOT / provenance["archive_path"]
    source_archive_hash_pass = bool(
        archive.exists()
        and archive.stat().st_size == int(provenance["archive_bytes"])
        and sha256(archive) == provenance["archive_sha256"]
        and provenance["archive_sha256"] == config["primary_source"]["source_archive_sha256"]
    )

    eso = query_eso_muse(config["archive_targets"]["eso_muse"]["proposal_id"])
    science_target = ~eso["target_name"].astype(str).str.upper().str.startswith("SKY_")
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
            "metadata_match": eso["instrument_name"].astype(str).eq("MUSE"),
            "expected_group": "097.A-0987:MUSE",
            "science_role": science_target.map(
                {True: "spatially resolved stellar kinematics", False: "paired sky background"}
            ),
        }
    )
    hst_inventory, hst_groups = query_hst(config)
    inventory = pd.concat([eso_inventory, hst_inventory], ignore_index=True)
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(INVENTORY_PATH, index=False, lineterminator="\n")

    repo = fetch_json("https://api.github.com/repos/tcollett/pylens")
    commit = fetch_json(
        f"https://api.github.com/repos/tcollett/pylens/commits/{repo['default_branch']}"
    )
    repo_public = not bool(repo["private"])
    normalized_likelihood_public = False

    dynamics_support = config["published_dynamics_support"]["accepted_model_outer_radius_arcsec"]
    ring_radius = config["published_lens_geometry"]["einstein_radius_arcsec"]
    ring_inside = ring_radius <= dynamics_support
    source_gate = bool(all(checks.values()) and source_archive_hash_pass)
    eso_gate = bool(
        len(eso) == config["archive_targets"]["eso_muse"]["expected_public_products"]
        and int(science_target.sum())
        == config["archive_targets"]["eso_muse"]["expected_public_science_cubes"]
        and int((eso["calib_level"] == 3).sum())
        == config["archive_targets"]["eso_muse"]["expected_public_level3_products"]
    )
    hst_gate = bool(len(hst_groups) == 2 and all(group["metadata_and_exposure_gate_pass"] for group in hst_groups))
    archive_gate = eso_gate and hst_gate
    extended_arc_sensitivity = bool(
        checks["extended_arcs_constrain_radial_magnification"]
        and checks["extended_arc_pixels_and_source_grid"]
    )
    pre_pixel_protocol_gate = bool(
        source_gate and archive_gate and ring_inside and extended_arc_sensitivity and repo_public
    )

    upsert_candidate_queue(
        {
            "system": config["system"],
            "alias": config["alias"],
            "selection_blind": True,
            "distinct_spectroscopic_source_planes": 1,
            "published_ring_scales_arcsec": str(ring_radius),
            "accepted_dynamics_outer_radius_arcsec": dynamics_support,
            "ring_scales_inside_accepted_support": int(ring_inside),
            "pre_fit_ring_scale_rank_upper_bound": 1,
            "extended_surface_brightness_constraints_published": True,
            "published_radial_magnification_sensitivity": True,
            "pre_pixel_structural_rank_upper_bound": "potentially_at_least_3_pending_jacobian",
            "full_image_level_structural_rank": "not_established",
            "raw_archives_public": archive_gate,
            "observable_level_normalized_likelihood_public": normalized_likelihood_public,
            "counts_toward_ten_system_target": False,
            "disposition": "authorize_pre_pixel_acquisition_protocol_not_structural_promotion",
            "primary_blocker": "nuisance-marginalized extended-arc response rank is not established",
            "next_authorized_stage": "freeze_exact_acquisition_and_image_level_jacobian_protocol",
        }
    )

    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "science_pixels_downloaded_or_inspected": False,
        "system": config["system"],
        "inputs": {
            "protocol": {
                "path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(CONFIG_PATH),
            },
            "source_provenance": {
                "path": str(PROVENANCE_PATH.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(PROVENANCE_PATH),
            },
            "primary_source_file": source_metadata,
        },
        "primary_source_checks": checks,
        "primary_source_archive_hash_check_passed": source_archive_hash_pass,
        "archive_inventory": {
            "eso_muse_product_count": int(len(eso)),
            "eso_muse_science_cube_count": int(science_target.sum()),
            "eso_muse_sky_cube_count": int((~science_target).sum()),
            "eso_muse_calibration_level_2_count": int((eso["calib_level"] == 2).sum()),
            "eso_muse_calibration_level_3_count": int((eso["calib_level"] == 3).sum()),
            "eso_science_product_ids": eso.loc[science_target, "dp_id"].astype(str).tolist(),
            "hst_expected_groups": hst_groups,
            "hst_expected_public_groups_found": int(
                sum(group["metadata_and_exposure_gate_pass"] for group in hst_groups)
            ),
            "hst_expected_public_groups_required": len(config["archive_targets"]["hst"]),
            "output": str(INVENTORY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "output_sha256": sha256(INVENTORY_PATH),
        },
        "public_supporting_products": {
            "pylens_repository": {
                "url": repo["html_url"],
                "public": repo_public,
                "default_branch": repo["default_branch"],
                "latest_commit_sha": commit["sha"],
                "archived": bool(repo["archived"]),
                "disposition": "general public lens-modelling code; no E325 observable likelihood, source reconstruction, covariance, or chains identified by the primary source",
            },
            "normalized_theory_neutral_likelihood_publicly_identified": normalized_likelihood_public,
            "published_joint_posterior_reusable_as_theory_neutral_data": False,
        },
        "geometry": {
            "accepted_dynamics_outer_radius_arcsec": dynamics_support,
            "published_einstein_radius_arcsec": ring_radius,
            "einstein_radius_inside_accepted_support": ring_inside,
            "ring_only_radial_rank_upper_bound": 1,
            "extended_surface_brightness_constraints_published": True,
            "published_radial_magnification_sensitivity": extended_arc_sensitivity,
            "pre_pixel_extended_arc_structural_rank_upper_bound": "potentially_at_least_3",
            "full_image_level_nuisance_marginalized_rank": "not_established",
            "minimum_required_rank": config["frozen_pre_pixel_gate"]["minimum_pre_fit_structural_radial_rank_upper_bound"],
            "pixel_count_used_as_rank": False,
        },
        "gates": {
            "primary_source_and_hash_passed": source_gate,
            "public_raw_archive_metadata_passed": archive_gate,
            "einstein_radius_inside_accepted_dynamics_support_passed": ring_inside,
            "published_extended_arc_radial_sensitivity_passed": extended_arc_sensitivity,
            "public_general_lens_code_identified": repo_public,
            "public_normalized_observable_likelihood_identified": normalized_likelihood_public,
            "pre_pixel_acquisition_and_jacobian_protocol_authorized": pre_pixel_protocol_gate,
            "rank_three_candidate_admission_passed": False,
        },
        "decision": "authorize_pre_pixel_acquisition_protocol_not_structural_promotion",
        "ten_system_effect": {
            "previous_structural_ceiling": 3,
            "updated_structural_ceiling": 3,
            "minimum_new_rank_three_systems_still_required": 7,
        },
        "next_action": "Freeze exact HST/MUSE product acquisition and an image-level lens-response Jacobian. Download or inspect science arrays only under that new checksum-locked protocol. Promote E325 only if at least three response singular directions survive source-light, lens-light, PSF, shear, mass-sheet, baryonic, and regularization nuisance projection and perturbation-stability tests.",
        "outputs": {
            "archive_inventory": str(INVENTORY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "candidate_queue": str(QUEUE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "candidate_queue_sha256": sha256(QUEUE_PATH),
        },
        "authorization": {
            "freeze_acquisition_and_image_level_jacobian_protocol": pre_pixel_protocol_gate,
            "download_science_pixels_under_current_protocol": False,
            "inspect_science_pixels_under_current_protocol": False,
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
