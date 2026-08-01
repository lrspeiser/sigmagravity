#!/usr/bin/env python3
"""Pre-screen A1689 lens geometry against the frozen Loubser dynamics support."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/r1_a1689_lens_prescreen"
CONFIG_PATH = ROOT / "configs/r1_a1689_lens_prescreen_protocol.json"
PROVENANCE_PATH = RAW / "provenance.json"
TABLE5_PATH = RAW / "coe2010_table5_multiple_images.dat"
IMAGE_OUTPUT = ROOT / "data/derived/r1_a1689_lens_prescreen_images.csv"
FAMILY_OUTPUT = ROOT / "data/derived/r1_a1689_lens_prescreen_families.csv"
REPORT_PATH = ROOT / "results/r1_a1689_lens_prescreen/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def optional_float(value: str) -> float | None:
    value = value.strip()
    return float(value) if value else None


def hms_degrees(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return 15.0 * (float(hours) + float(minutes) / 60.0 + float(seconds) / 3600.0)


def dms_degrees(value: str) -> float:
    sign = -1.0 if value.startswith("-") else 1.0
    degrees, minutes, seconds = value[1:].split(":")
    return sign * (float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0)


def angular_separation_arcsec(ra_1: float, dec_1: float, ra_2: float, dec_2: float) -> float:
    ra1, dec1, ra2, dec2 = map(math.radians, [ra_1, dec_1, ra_2, dec_2])
    cosine = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3600.0


def parse_table5() -> pd.DataFrame:
    records = []
    for line in TABLE5_PATH.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        line = line.ljust(107)
        ra = 15.0 * (
            int(line[19:21]) + int(line[22:24]) / 60.0 + float(line[25:31]) / 3600.0
        )
        sign = -1.0 if line[32:33] == "-" else 1.0
        dec = sign * (int(line[33:34]) + int(line[35:37]) / 60.0 + float(line[38:43]) / 3600.0)
        records.append(
            {
                "family_id": int(line[0:2]),
                "image_letter": line[2:3],
                "image_id": f"{int(line[0:2])}{line[2:3]}",
                "ra_deg": ra,
                "dec_deg": dec,
                "spectroscopic_redshift": optional_float(line[78:82]),
                "spectroscopic_redshift_reference": line[83:86].strip(),
                "photometric_redshift": optional_float(line[87:91]),
                "photometric_redshift_95_upper_delta": optional_float(line[92:96]),
                "photometric_redshift_95_lower_delta": optional_float(line[97:101]),
                "lensperfect_magnification_audit_only": optional_float(line[55:60]),
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != 135 or frame["family_id"].nunique() != 42:
        raise RuntimeError("A1689 catalog integrity mismatch")
    return frame


def build_audit() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8-sig"))
    for record in provenance["records"]:
        path = ROOT / record["local_path"]
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Provenance mismatch for {path}")

    images = parse_table5()
    if images["image_id"].duplicated().any() or images[["ra_deg", "dec_deg"]].duplicated().any():
        raise RuntimeError("Duplicate A1689 image identifier or coordinate")
    center_ra = hms_degrees(config["center"]["ra_hms"])
    center_dec = dms_degrees(config["center"]["dec_dms"])
    images["radius_arcsec"] = images.apply(
        lambda row: angular_separation_arcsec(center_ra, center_dec, row["ra_deg"], row["dec_deg"]), axis=1
    )
    images["radius_kpc"] = images["radius_arcsec"] * config["distance_conversion"]["kpc_per_arcsec"]

    family_records = []
    for family, group in images.groupby("family_id"):
        spectroscopy = group["spectroscopic_redshift"].dropna()
        photometry = group.loc[
            group["photometric_redshift"].notna()
            & group["photometric_redshift_95_upper_delta"].notna()
            & group["photometric_redshift_95_lower_delta"].notna()
        ]
        if not spectroscopy.empty:
            independent = True
            kind = "spectroscopic"
            redshift = float(spectroscopy.median())
        elif not photometry.empty:
            independent = True
            kind = "row_level_bayesian_photometric_95pct_intervals"
            redshift = None
        else:
            independent = False
            kind = ""
            redshift = None
        family_records.append(
            {
                "family_id": int(family),
                "images": int(len(group)),
                "independent_redshift_anchor": independent,
                "independent_redshift_kind": kind,
                "family_spectroscopic_redshift": redshift,
            }
        )
    families = pd.DataFrame(family_records)
    family_anchor = families.set_index("family_id")["independent_redshift_anchor"]
    family_kind = families.set_index("family_id")["independent_redshift_kind"]
    family_spec = families.set_index("family_id")["family_spectroscopic_redshift"]
    images["family_independent_redshift_anchor"] = images["family_id"].map(family_anchor)
    images["family_independent_redshift_kind"] = images["family_id"].map(family_kind)
    images["family_spectroscopic_redshift"] = images["family_id"].map(family_spec)
    support = float(config["dynamics_support"]["one_sided_radius_kpc"])
    images["inside_frozen_dynamics_support"] = images["radius_kpc"] <= support
    images["metric_neutral_geometry_row"] = (
        images["family_independent_redshift_anchor"] & images["inside_frozen_dynamics_support"]
    )
    images["published_position_error"] = False
    images["published_coordinate_covariance"] = False
    images["lensperfect_magnification_used_as_input"] = False
    images["gravity_target_used"] = False

    inner = images.loc[images["metric_neutral_geometry_row"]]
    distinct_radii = int(inner["radius_kpc"].round(6).nunique())
    gates = config["pre_registered_gates"]
    integrity = bool(
        len(images) == gates["catalog_integrity"]["exact_image_count"]
        and images["family_id"].nunique() == gates["catalog_integrity"]["exact_family_count"]
    )
    bridge = gates["same_radius_bridge"]
    same_radius = bool(
        len(inner) >= bridge["minimum_independently_redshift_anchored_images_inside_dynamics_support"]
        and inner["family_id"].nunique()
        >= bridge["minimum_independently_redshift_anchored_families_inside_dynamics_support"]
        and distinct_radii >= bridge["minimum_distinct_image_radii_inside_dynamics_support"]
    )
    raw_authorized = integrity and same_radius

    IMAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FAMILY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    images.sort_values(["radius_kpc", "family_id", "image_id"]).to_csv(IMAGE_OUTPUT, index=False)
    families.sort_values("family_id").to_csv(FAMILY_OUTPUT, index=False)
    report = {
        "report_version": "R1B1-A1689-lens-prescreen-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "catalog": {
            "images": int(len(images)),
            "families": int(images["family_id"].nunique()),
            "independently_redshift_anchored_families": int(families["independent_redshift_anchor"].sum()),
            "published_position_errors": False,
            "published_coordinate_covariance": False,
        },
        "radial_overlap": {
            "dynamics_support_kpc": support,
            "independently_redshift_anchored_images_inside_support": int(len(inner)),
            "independently_redshift_anchored_families_inside_support": int(inner["family_id"].nunique()),
            "distinct_image_radii_inside_support": distinct_radii,
            "nearest_image_id": str(images.iloc[images["radius_kpc"].argmin()]["image_id"]),
            "nearest_image_radius_kpc": float(images["radius_kpc"].min()),
        },
        "gates": {
            "catalog_integrity_passed": integrity,
            "same_radius_bridge_passed": same_radius,
            "raw_gemini_reconstruction_authorized": raw_authorized,
            "observable_coordinate_likelihood_ready": False,
            "gravity_response_fit_authorized": False,
        },
        "authorization": {
            "download_and_reduce_raw_gemini_spectra": raw_authorized,
            "derive_astrometric_covariance_from_raw_hst_before_lens_residuals": raw_authorized,
            "fit_lens_mass_model": False,
            "infer_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "outputs": {
            "images": str(IMAGE_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "families": str(FAMILY_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_audit(), indent=2))
