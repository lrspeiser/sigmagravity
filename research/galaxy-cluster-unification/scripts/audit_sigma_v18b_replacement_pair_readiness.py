#!/usr/bin/env python3
"""Audit a spent replacement pair for collisionless-member-stress readiness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v18b_replacement_pair_readiness.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v18b_replacement_pair_readiness"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def angular_separation_arcsec(
    ra_deg: float, dec_deg: float, center_ra_deg: float, center_dec_deg: float
) -> float:
    """Stable small-angle separation for a sub-degree cluster field."""

    mean_dec = math.radians(0.5 * (dec_deg + center_dec_deg))
    east = math.radians(ra_deg - center_ra_deg) * math.cos(mean_dec)
    north = math.radians(dec_deg - center_dec_deg)
    return math.degrees(math.hypot(east, north)) * 3600.0


def ab_magnitude(row: dict[str, str], cluster: dict[str, Any]) -> float:
    value = float(row[cluster["photometry_column"]])
    if cluster["photometry_kind"] == "ab_magnitude":
        return value
    if cluster["photometry_kind"] == "cgs_fnu":
        if value <= 0.0:
            return math.nan
        return -2.5 * math.log10(value) - 48.6
    raise RuntimeError("unsupported photometry representation")


def stellar_mass_msun(
    magnitude: float, cluster: dict[str, Any], weight: dict[str, Any]
) -> float:
    absolute = (
        magnitude
        - float(cluster["planck18_distance_modulus"])
        + 2.5 * math.log10(1.0 + float(cluster["redshift"]))
    )
    luminosity = 10.0 ** (
        -0.4 * (absolute - float(weight["solar_absolute_ab_magnitude"]))
    )
    return float(weight["mass_to_light_solar"]) * luminosity


def audit_cluster(
    cluster_name: str,
    cluster: dict[str, Any],
    selection: dict[str, Any],
    weight: dict[str, Any],
) -> dict[str, Any]:
    rows = read_csv(ROOT / cluster["member_catalog"])
    required = {
        cluster["ra_column"],
        cluster["dec_column"],
        cluster["redshift_column"],
        cluster["photometry_column"],
    }
    if cluster["quality_column"] is not None:
        required.add(cluster["quality_column"])
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"{cluster_name} member schema is incomplete")

    quality_rows = []
    for row in rows:
        if cluster["quality_column"] is not None and float(
            row[cluster["quality_column"]]
        ) < float(cluster["minimum_quality"]):
            continue
        quality_rows.append(row)
    redshifts = [float(row[cluster["redshift_column"]]) for row in quality_rows]
    if not redshifts or not all(math.isfinite(value) for value in redshifts):
        raise RuntimeError(f"{cluster_name} has no finite quality-approved redshifts")
    median_redshift = float(statistics.median(redshifts))

    selected = []
    photometry_rejected = 0
    aperture_rejected = 0
    velocity_rejected = 0
    for row in quality_rows:
        magnitude = ab_magnitude(row, cluster)
        mass = stellar_mass_msun(magnitude, cluster, weight)
        if not math.isfinite(magnitude) or not math.isfinite(mass) or mass <= 0.0:
            photometry_rejected += 1
            continue
        separation_arcsec = angular_separation_arcsec(
            float(row[cluster["ra_column"]]),
            float(row[cluster["dec_column"]]),
            float(cluster["center_ra_deg"]),
            float(cluster["center_dec_deg"]),
        )
        radius_kpc = separation_arcsec * float(cluster["planck18_kpc_per_arcsec"])
        velocity = float(selection["speed_of_light_km_s"]) * (
            float(row[cluster["redshift_column"]]) - median_redshift
        ) / (1.0 + median_redshift)
        if radius_kpc > float(selection["projected_aperture_kpc"]):
            aperture_rejected += 1
            continue
        if abs(velocity) > float(
            selection["maximum_absolute_rest_frame_velocity_km_s"]
        ):
            velocity_rejected += 1
            continue
        selected.append((radius_kpc, velocity, mass))

    minimum = int(selection["minimum_secure_members_inside_aperture"])
    return {
        "cluster": cluster_name,
        "input_rows": len(rows),
        "quality_approved_rows": len(quality_rows),
        "median_redshift": median_redshift,
        "photometry_rejected_rows": photometry_rejected,
        "aperture_rejected_rows": aperture_rejected,
        "velocity_rejected_rows": velocity_rejected,
        "selected_secure_members": len(selected),
        "minimum_required": minimum,
        "member_gate_passed": len(selected) >= minimum,
        "maximum_selected_radius_kpc": max(value[0] for value in selected),
        "velocity_range_km_s": [
            min(value[1] for value in selected),
            max(value[1] for value in selected),
        ],
        "stellar_mass_weight_range_msun": [
            min(value[2] for value in selected),
            max(value[2] for value in selected),
        ],
        "summed_stellar_mass_weight_msun": sum(value[2] for value in selected),
        "common_f160w_weight_rule": True,
    }


def validate_inputs(config: dict[str, Any]) -> dict[str, str]:
    hashes = {"config": sha256(DEFAULT_CONFIG)}
    for name, parent_path_key, parent_hash_key in (
        ("dynamical_stress_gate", "dynamical_stress_gate", "dynamical_stress_gate_sha256"),
        ("thermal_failure_report", "thermal_failure_report", "thermal_failure_report_sha256"),
    ):
        path = ROOT / config["parents"][parent_path_key]
        actual = sha256(path)
        if actual != config["parents"][parent_hash_key]:
            raise RuntimeError(f"frozen {name} changed")
        hashes[name] = actual
    for cluster_name, cluster in config["clusters"].items():
        for key in ("member_catalog", "baryon_report"):
            path = ROOT / cluster[key]
            actual = sha256(path)
            if actual != cluster[f"{key}_sha256"]:
                raise RuntimeError(f"frozen {cluster_name} {key} changed")
            hashes[f"{cluster_name}_{key}"] = actual
        baryon_key = "baryon_sources" if "baryon_sources" in cluster else "baryon_map"
        path = ROOT / cluster[baryon_key]
        actual = sha256(path)
        if actual != cluster[f"{baryon_key}_sha256"]:
            raise RuntimeError(f"frozen {cluster_name} {baryon_key} changed")
        hashes[f"{cluster_name}_{baryon_key}"] = actual
    return hashes


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config_path.resolve()
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if not config["status"].startswith("frozen after v17E failed"):
        raise RuntimeError("v18B readiness protocol is not frozen")
    if not config["universal_selection"]["minimum_is_inherited_unchanged"]:
        raise RuntimeError("the inherited member gate was relaxed")
    if config["authorization"]["formula_or_spatial_kernel_selection_authorized"]:
        raise RuntimeError("readiness audit cannot select a formula or kernel")

    hashes = validate_inputs(config)
    clusters = [
        audit_cluster(name, cluster, config["universal_selection"], config["universal_stellar_weight"])
        for name, cluster in config["clusters"].items()
    ]
    ready = all(item["member_gate_passed"] for item in clusters)
    return {
        "status": "completed Sigma v18B replacement-pair readiness audit",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": hashes,
        "clusters": clusters,
        "matched_pair": list(config["clusters"]),
        "source_construction_authorized": ready,
        "formula_or_spatial_kernel_selection_authorized": False,
        "lensing_target_opened": False,
        "holdout_opened": False,
        "gravity_parameters_fit": 0,
        "decision": (
            "freeze one universal member-stress map construction on this spent pair"
            if ready
            else "do not construct a matched collisionless-stress source"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
