#!/usr/bin/env python3
"""Normalize A2261 image positions and apply the frozen same-radius gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/r1_a2261_lens_observable_protocol.json"
RAW_ROOT = ROOT / "data/raw/r1_a2261_lens_observables"
CATALOG_PATH = RAW_ROOT / "coe2012_table3_multiple_images.dat"
PROVENANCE_PATH = RAW_ROOT / "provenance.json"
OUTPUT_PATH = ROOT / "data/derived/r1_a2261_lens_observables.csv"
FAMILY_PATH = ROOT / "data/derived/r1_a2261_lens_families.csv"
REPORT_PATH = ROOT / "results/r1_a2261_lens_observables/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sexagesimal_ra_degrees(value: str) -> float:
    hours, minutes, seconds = (float(item) for item in value.replace(":", " ").split())
    return 15.0 * (hours + minutes / 60.0 + seconds / 3600.0)


def sexagesimal_dec_degrees(value: str) -> float:
    sign = -1.0 if value.strip().startswith("-") else 1.0
    degrees, minutes, seconds = (
        float(item) for item in value.lstrip("+-").replace(":", " ").split()
    )
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def leading_float(value: str) -> float:
    match = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", value)
    if match is None:
        raise ValueError(f"No numeric value in {value!r}")
    return float(match.group(1))


def parse_catalog() -> pd.DataFrame:
    records = []
    for line in CATALOG_PATH.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 10:
            raise RuntimeError(f"Expected 10 pipe-delimited fields, got {len(fields)}: {line!r}")
        image_id = fields[0].strip()
        match = re.fullmatch(r"([0-9]+)([a-d])", image_id)
        if match is None:
            raise RuntimeError(f"Invalid image identifier {image_id!r}")
        family = int(match.group(1))
        ra_deg = sexagesimal_ra_degrees(fields[1].strip())
        dec_deg = sexagesimal_dec_degrees(fields[2].strip())
        records.append(
            {
                "system": "Abell 2261",
                "family_id": family,
                "image_id": image_id,
                "ra_hms": fields[1].strip(),
                "dec_dms": fields[2].strip(),
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "hst_magnitude_ab": leading_float(fields[3]),
                "hst_magnitude_error": leading_float(fields[4]),
                "continuous_arc_or_other_filter_flag": bool(re.search(r"[cd]", fields[4])),
                "best_isolated_redshift_anchor": fields[5].strip() == "*",
                "image_photometric_redshift": leading_float(fields[6]),
                "image_photometric_redshift_error_plus_95": leading_float(fields[7]),
                "image_photometric_redshift_error_minus_95": leading_float(fields[8]),
                "lensperfect_model_redshift_forbidden_as_input": leading_float(fields[9]),
                "position_uncertainty_published": False,
                "position_uncertainty_arcsec": math.nan,
                "coordinate_covariance_published": False,
            }
        )
    frame = pd.DataFrame(records)
    if frame["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image identifiers in A2261 catalog")
    if frame[["ra_deg", "dec_deg"]].duplicated().any():
        raise RuntimeError("Duplicate image coordinates in A2261 catalog")
    return frame


def build_audit() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8-sig"))
    for record in provenance["records"]:
        path = ROOT / record["local_path"]
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Provenance mismatch for {path}")

    frame = parse_catalog()
    center_ra = sexagesimal_ra_degrees(protocol["center"]["ra_hms"])
    center_dec = sexagesimal_dec_degrees(protocol["center"]["dec_dms"])
    cos_dec = math.cos(math.radians(center_dec))
    frame["delta_ra_arcsec"] = (frame["ra_deg"] - center_ra) * cos_dec * 3600.0
    frame["delta_dec_arcsec"] = (frame["dec_deg"] - center_dec) * 3600.0
    frame["radius_arcsec"] = (frame["delta_ra_arcsec"] ** 2 + frame["delta_dec_arcsec"] ** 2) ** 0.5
    frame["radius_kpc"] = frame["radius_arcsec"] * protocol["distance_conversion"]["kpc_per_arcsec"]
    dynamics_radius = protocol["dynamics_support"]["one_sided_radius_kpc"]
    frame["inside_dynamics_support"] = frame["radius_kpc"] <= dynamics_radius

    anchors = frame.loc[frame["best_isolated_redshift_anchor"]].copy()
    if anchors["family_id"].duplicated().any() or set(anchors["family_id"]) != set(frame["family_id"]):
        raise RuntimeError("Expected exactly one best-isolated photometric-redshift anchor per family")
    family_redshift = {
        int(row.family_id): {
            "family_redshift": float(row.image_photometric_redshift),
            "family_redshift_error_plus_95": float(row.image_photometric_redshift_error_plus_95),
            "family_redshift_error_minus_95": float(row.image_photometric_redshift_error_minus_95),
            "family_redshift_kind": "independent_photometric_summary",
            "family_redshift_source": f"Coe2012 best-isolated image {row.image_id}",
        }
        for row in anchors.itertuples()
    }
    spec = provenance["later_spectroscopic_update"]
    family_redshift[int(spec["family"])] = {
        "family_redshift": float(spec["redshift"]),
        "family_redshift_error_plus_95": math.nan,
        "family_redshift_error_minus_95": math.nan,
        "family_redshift_kind": "independent_spectroscopic_redshift",
        "family_redshift_source": "Rydberg2017 MOSFIRE image 4a",
    }
    for column in next(iter(family_redshift.values())):
        frame[column] = frame["family_id"].map(lambda family: family_redshift[int(family)][column])
    frame["lens_independent_family_redshift"] = True
    frame["strict_coordinate_likelihood_row"] = False
    frame["gravity_target_used"] = False

    family = (
        frame.groupby("family_id", as_index=False)
        .agg(
            image_count=("image_id", "size"),
            images_inside_dynamics_support=("inside_dynamics_support", "sum"),
            minimum_radius_kpc=("radius_kpc", "min"),
            maximum_radius_kpc=("radius_kpc", "max"),
            family_redshift=("family_redshift", "first"),
            family_redshift_kind=("family_redshift_kind", "first"),
            family_redshift_source=("family_redshift_source", "first"),
        )
        .sort_values("family_id")
    )
    family["family_has_image_inside_dynamics_support"] = family["images_inside_dynamics_support"] > 0
    family["family_wide_position_dof_after_source_position"] = 2 * (family["image_count"] - 1)
    family["gravity_target_used"] = False

    catalog_gate = protocol["pre_registered_gates"]["catalog_integrity"]
    image_count = int(len(frame))
    family_count = int(frame["family_id"].nunique())
    catalog_pass = bool(
        image_count == catalog_gate["exact_image_count"]
        and family_count == catalog_gate["exact_family_count"]
        and not frame[["ra_deg", "dec_deg"]].duplicated().any()
    )
    likelihood_gate = protocol["pre_registered_gates"]["observable_likelihood"]
    independent_families = int(frame.loc[frame["lens_independent_family_redshift"], "family_id"].nunique())
    images_with_independent_family_redshift = int(frame["lens_independent_family_redshift"].sum())
    stated_errors = bool(frame["position_uncertainty_published"].all())
    coordinate_covariance = bool(frame["coordinate_covariance_published"].all())
    likelihood_pass = bool(
        independent_families >= likelihood_gate["minimum_families_with_lens_independent_redshift"]
        and images_with_independent_family_redshift >= likelihood_gate["minimum_images_in_those_families"]
        and stated_errors
        and coordinate_covariance
    )
    bridge_gate = protocol["pre_registered_gates"]["same_radius_bridge"]
    inner_images = int(frame["inside_dynamics_support"].sum())
    inner_families = int(family["family_has_image_inside_dynamics_support"].sum())
    radial_rank_upper = min(inner_images, inner_families)
    bridge_pass = bool(
        inner_images >= bridge_gate["minimum_images_inside_dynamics_support"]
        and inner_families >= bridge_gate["minimum_families_inside_dynamics_support"]
        and radial_rank_upper >= bridge_gate["minimum_structural_radial_rank_upper_bound"]
    )
    nearest = frame.sort_values("radius_kpc").iloc[0]
    strict_ready = bool(
        catalog_pass
        and likelihood_pass
        and bridge_pass
        and protocol["dynamics_support"]["machine_readable_profile"]
        and protocol["dynamics_support"]["published_covariance"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAMILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["family_id", "image_id"]).to_csv(OUTPUT_PATH, index=False)
    family.to_csv(FAMILY_PATH, index=False)
    report = {
        "report_version": "R1A2-a2261-observable-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "system": "Abell 2261",
        "selection_blind": True,
        "catalog": {
            "images": image_count,
            "families": family_count,
            "family_wide_position_dof_after_source_positions": int(2 * (image_count - family_count)),
            "lens_independent_redshift_families": independent_families,
            "images_with_lens_independent_family_redshift": images_with_independent_family_redshift,
            "spectroscopic_redshift_families": int((family["family_redshift_kind"] == "independent_spectroscopic_redshift").sum()),
            "published_position_errors": stated_errors,
            "published_coordinate_covariance": coordinate_covariance,
            "lensperfect_redshifts_retained_for_audit_only": True,
            "lensperfect_redshifts_used_as_inputs": False,
        },
        "radial_overlap": {
            "dynamics_support_kpc": dynamics_radius,
            "images_inside_dynamics_support": inner_images,
            "families_inside_dynamics_support": inner_families,
            "structural_radial_rank_upper_bound": radial_rank_upper,
            "nearest_image_id": str(nearest["image_id"]),
            "nearest_image_radius_arcsec": float(nearest["radius_arcsec"]),
            "nearest_image_radius_kpc": float(nearest["radius_kpc"]),
            "gap_beyond_dynamics_support_kpc": float(nearest["radius_kpc"] - dynamics_radius),
        },
        "gates": {
            "catalog_integrity_passed": catalog_pass,
            "observable_coordinate_likelihood_passed": likelihood_pass,
            "same_radius_bridge_passed": bridge_pass,
            "strict_r1_readiness_passed": strict_ready,
        },
        "failed_requirements": [
            name
            for name, passed in {
                "published_or_reconstructed_position_errors": stated_errors,
                "published_coordinate_covariance": coordinate_covariance,
                "three_images_inside_15_kpc": inner_images >= bridge_gate["minimum_images_inside_dynamics_support"],
                "two_families_inside_15_kpc": inner_families >= bridge_gate["minimum_families_inside_dynamics_support"],
                "three_structural_radial_modes_inside_15_kpc": radial_rank_upper >= bridge_gate["minimum_structural_radial_rank_upper_bound"],
                "machine_readable_dynamics_profile": protocol["dynamics_support"]["machine_readable_profile"],
                "dynamics_covariance": protocol["dynamics_support"]["published_covariance"],
                "complete_baryonic_profiles": False,
            }.items()
            if not passed
        ],
        "status": "observable_catalog_ingested_same_radius_and_likelihood_gates_failed",
        "authorization": {
            "count_as_clash_observable_catalog_acquired": catalog_pass,
            "promote_to_same_system_response_sample": strict_ready,
            "fit_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "outputs": {
            "image_catalog": str(OUTPUT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "family_catalog": str(FAMILY_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = build_audit()
    print(json.dumps(result, indent=2))
