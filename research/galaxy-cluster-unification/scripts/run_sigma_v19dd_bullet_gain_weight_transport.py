#!/usr/bin/env python3
"""Build integrated Bullet products and transport per-ObsID gain covariance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19db_bullet_velocity_combination as v19db

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19dd_bullet_gain_weight_transport.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19dd_bullet_gain_weight_transport"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19dd-bullet-gain-weight-transport/v100")
AUTHORIZED_STATUS = "bullet_integrated_spectrum_and_gain_weight_transport_passed"
C_KM_S = 299792.458


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_parent(item: dict[str, Any]) -> Path:
    path = ROOT / item["path"]
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise RuntimeError(f"V19DD frozen parent changed: {path}")
    return path


def validate_frozen(config: dict[str, Any]) -> dict[str, Path]:
    if config.get("freeze_state") != "frozen_after_v19dc_gain_pass_before_source_redshift_fit":
        raise RuntimeError("V19DD is not frozen before source-response access")
    implementation = config["implementation"]
    if implementation["runner"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise RuntimeError("V19DD config names another runner")
    if implementation["runner_sha256"] != sha256(Path(__file__).resolve()):
        raise RuntimeError("V19DD runner changed after freeze")
    parents = {
        key: validate_parent(item)
        for key, item in config["parents"].items()
        if isinstance(item, dict) and "path" in item
    }
    if load_json(parents["v19db_report"]).get("status") != config["parents"]["v19db_report"]["required_status"]:
        raise RuntimeError("V19DB response parent is not a terminal pass")
    if load_json(parents["v19dc_report"]).get("status") != config["parents"]["v19dc_report"]["required_status"]:
        raise RuntimeError("V19DC gain parent is not a terminal pass")
    auth = config["authorization"]
    if not (
        auth["combine_integrated_bullet_primary_spectrum"]
        and auth["open_source_pha_header_and_arf_response"]
        and auth["open_source_pha_counts_for_exact_combination_audit"]
        and not auth["fit_temperature_abundance_redshift_or_velocity"]
        and not auth["open_obsid554_or_abell2146"]
        and not auth["open_lensing_halo_gravity_or_action"]
    ):
        raise RuntimeError("V19DD authorization boundary is open")
    return parents


def load_plan(config: dict[str, Any], parents: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parent_config = load_json(parents["v19db_config"])
    parent_paths = v19db.validate_frozen(parent_config)
    plan = v19db.build_plan(parent_config, parent_paths)
    if len(plan) != int(config["workload"]["regions"]):
        raise RuntimeError("V19DD regional parent count changed")
    if sum(len(region["cells"]) for region in plan) != int(config["workload"]["response_cells"]):
        raise RuntimeError("V19DD primary cell count changed")
    expected_obsids = [int(value) for value in config["workload"]["obsids"]]
    if [int(value) for value in parent_config["workload"]["obsids"]] != expected_obsids:
        raise RuntimeError("V19DD primary observation list changed")
    group_ids = [int(region["group_id"]) for region in plan]
    if len(group_ids) != len(set(group_ids)):
        raise RuntimeError("V19DD regional parent contains duplicate group IDs")
    for region in plan:
        observed = {int(cell["obsid"]) for cell in region["cells"]}
        if observed != set(expected_obsids):
            raise RuntimeError(f"V19DD region {region['group_id']} lacks a primary observation")
    return parent_config, plan


@cache
def arf_value(path: Path, energy_keV: float) -> float:
    with fits.open(path, memmap=False) as hdus:
        data = hdus["SPECRESP"].data
        lo = np.asarray(data["ENERG_LO"], dtype=float)
        hi = np.asarray(data["ENERG_HI"], dtype=float)
        area = np.asarray(data["SPECRESP"], dtype=float)
    match = np.flatnonzero((lo <= energy_keV) & (hi > energy_keV))
    if len(match) != 1 or not np.isfinite(area[match[0]]) or area[match[0]] <= 0:
        raise RuntimeError(f"V19DD ARF lacks one positive Fe-K bin: {path}")
    return float(area[match[0]])


@cache
def pha_exposure(path: Path) -> float:
    with fits.open(path, memmap=False) as hdus:
        exposure = float(hdus["SPECTRUM"].header["EXPOSURE"])
    if not math.isfinite(exposure) or exposure <= 0:
        raise RuntimeError(f"V19DD invalid source PHA exposure: {path}")
    return exposure


def region_weights(region: dict[str, Any], energy_keV: float, obsids: list[int]) -> list[dict[str, Any]]:
    contributions = {obsid: 0.0 for obsid in obsids}
    cell_counts = {obsid: 0 for obsid in obsids}
    for cell in region["cells"]:
        obsid = int(cell["obsid"])
        if obsid not in contributions:
            raise RuntimeError(f"V19DD region {region['group_id']} contains unexpected ObsID {obsid}")
        contribution = pha_exposure(cell["products"]["source"]) * arf_value(cell["products"]["arf"], energy_keV)
        contributions[obsid] += contribution
        cell_counts[obsid] += 1
    total = sum(contributions.values())
    if not math.isfinite(total) or total <= 0 or any(value <= 0 for value in contributions.values()):
        raise RuntimeError(f"V19DD invalid response contributions for region {region['group_id']}")
    return [
        {
            "group_id": int(region["group_id"]),
            "obsid": obsid,
            "cells": cell_counts[obsid],
            "exposure_area_contribution_cm2_s": contributions[obsid],
            "normalized_weight": contributions[obsid] / total,
        }
        for obsid in obsids
    ]


def effective_gain(
    group_id: int,
    weights: list[dict[str, Any]],
    gain_by_obsid: dict[int, dict[str, Any]],
    observed_fe: float,
) -> dict[str, Any]:
    weight_obsids = [int(row["obsid"]) for row in weights]
    if len(weight_obsids) != len(set(weight_obsids)) or set(weight_obsids) != set(gain_by_obsid):
        raise RuntimeError(f"V19DD region {group_id} gain-weight observation set changed")
    if abs(sum(float(row["normalized_weight"]) for row in weights) - 1.0) > 1e-12:
        raise RuntimeError(f"V19DD region {group_id} gain weights do not sum to one")
    parameters = np.zeros(2, dtype=float)
    covariance = np.zeros((2, 2), dtype=float)
    corrections: list[tuple[float, float]] = []
    for row in weights:
        gain = gain_by_obsid[int(row["obsid"])]["gain"]
        weight = float(row["normalized_weight"])
        parameters += weight * np.array([gain["intercept_keV"], gain["slope"]], dtype=float)
        covariance += weight * weight * np.asarray(gain["covariance_intercept_slope"], dtype=float)
        correction = float(gain["intercept_keV"]) + (float(gain["slope"]) - 1.0) * observed_fe
        corrections.append((weight, correction))
    eigenvalues = np.linalg.eigvalsh(covariance)
    vector = np.array([1.0, observed_fe], dtype=float)
    sigma_energy = math.sqrt(max(0.0, float(vector @ covariance @ vector)))
    effective_correction = float(parameters[0] + (parameters[1] - 1.0) * observed_fe)
    correction_dispersion = math.sqrt(
        max(0.0, sum(weight * (correction - effective_correction) ** 2 for weight, correction in corrections))
    )
    return {
        "group_id": group_id,
        "intercept_keV": float(parameters[0]),
        "slope": float(parameters[1]),
        "covariance_intercept_slope": covariance.tolist(),
        "minimum_covariance_eigenvalue": float(np.min(eigenvalues)),
        "correction_at_observed_fe_keV": effective_correction,
        "weighted_rms_obsid_correction_dispersion_keV": correction_dispersion,
        "weighted_rms_obsid_correction_dispersion_km_s": C_KM_S * correction_dispersion / observed_fe,
        "one_sigma_energy_uncertainty_at_observed_fe_keV": sigma_energy,
        "one_sigma_equivalent_velocity_uncertainty_km_s": C_KM_S * sigma_energy / observed_fe,
        "covariance_finite_symmetric_psd": bool(
            np.isfinite(covariance).all()
            and np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-14)
            and float(np.min(eigenvalues)) >= -1e-14
        ),
    }


def write_weights(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    path = output / "region_obsid_fe_weights.csv"
    fields = ["group_id", "obsid", "cells", "exposure_area_contribution_cm2_s", "normalized_weight"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"path": path.relative_to(ROOT).as_posix(), "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_effective_gain(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    path = output / "effective_gain_by_region.json"
    atomic_json(path, {"regions": rows})
    return {"path": path.relative_to(ROOT).as_posix(), "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}


def integrated_weight_equivalence(
    all_cells: list[dict[str, Any]], scratch: Path, label: str, energy_keV: float, obsids: list[int]
) -> dict[str, Any]:
    direct = {obsid: 0.0 for obsid in obsids}
    for cell in all_cells:
        obsid = int(cell["obsid"])
        direct[obsid] += pha_exposure(cell["products"]["source"]) * arf_value(cell["products"]["arf"], energy_keV)
    rows = []
    for obsid in obsids:
        work = scratch / label / f"obs{obsid}"
        prefix = f"{label}_obs{obsid}"
        pha = work / f"{prefix}_src.pi"
        arf = work / f"{prefix}_src.arf"
        hierarchical = pha_exposure(pha) * arf_value(arf, energy_keV)
        relative = abs(hierarchical - direct[obsid]) / direct[obsid]
        rows.append(
            {
                "obsid": obsid,
                "direct_cell_sum_cm2_s": direct[obsid],
                "hierarchical_cm2_s": hierarchical,
                "relative_difference": relative,
            }
        )
    return {"obsids": rows, "maximum_relative_difference": max(row["relative_difference"] for row in rows)}


def execute(config: dict[str, Any], output: Path, scratch: Path) -> dict[str, Any]:
    parents = validate_frozen(config)
    parent_config, plan = load_plan(config, parents)
    all_cells = [cell for region in plan for cell in region["cells"]]
    integrated_region = {
        "group_id": int(config["workload"]["integrated_group_id"]),
        "root_bin_id": -1,
        "member_bin_ids": sorted({int(cell["bin_id"]) for cell in all_cells}),
        "cells": all_cells,
    }
    integrated = v19db.hierarchical_combine_region(integrated_region, parent_config, scratch, output)
    observed_fe = float(config["weight_definition"]["representative_fe_rest_energy_keV"]) / (
        1.0 + float(config["weight_definition"]["bullet_optical_redshift"])
    )
    obsids = [int(value) for value in config["workload"]["obsids"]]
    with ThreadPoolExecutor(max_workers=int(config["runtime"]["parallel_regions"])) as pool:
        per_region = list(pool.map(lambda region: region_weights(region, observed_fe, obsids), plan))
    weight_rows = [row for region_rows in per_region for row in region_rows]
    gain_report = load_json(parents["v19dc_report"])
    gain_by_obsid = {int(row["obsid"]): row for row in gain_report["obsids"]}
    if set(gain_by_obsid) != set(obsids) or len(gain_report["obsids"]) != len(obsids):
        raise RuntimeError("V19DD gain parent observation set changed")
    effective = [
        effective_gain(int(region["group_id"]), rows, gain_by_obsid, observed_fe)
        for region, rows in zip(plan, per_region, strict=True)
    ]
    weights_artifact = write_weights(weight_rows, output)
    gain_artifact = write_effective_gain(effective, output)
    equivalence = integrated_weight_equivalence(all_cells, scratch, integrated["label"], observed_fe, obsids)
    tolerance = float(
        config["gates"]["direct_cell_sum_matches_observation_hierarchy_fe_weight_relative_difference_at_most"]
    )
    weight_coordinates = {(int(row["group_id"]), int(row["obsid"])) for row in weight_rows}
    expected_coordinates = {(int(region["group_id"]), obsid) for region in plan for obsid in obsids}
    gates = {
        "integrated_uses_all_cells": integrated["cells"] == int(config["workload"]["response_cells"]),
        "integrated_source_counts_exact": integrated["combined_full_pha_source_counts"]
        == int(config["workload"]["integrated_expected_full_pha_source_counts"]),
        "integrated_links_exact": integrated["links_exact"],
        "region_by_obsid_weights_exact": len(weight_rows) == len(expected_coordinates)
        and weight_coordinates == expected_coordinates,
        "every_region_weight_sum_one": all(
            abs(sum(row["normalized_weight"] for row in rows) - 1.0) <= 1e-12 for rows in per_region
        ),
        "direct_hierarchy_weight_equivalence": equivalence["maximum_relative_difference"] <= tolerance,
        "all_effective_gain_covariances_finite_psd": all(row["covariance_finite_symmetric_psd"] for row in effective),
        "all_gain_correction_dispersions_finite": all(
            math.isfinite(row["weighted_rms_obsid_correction_dispersion_keV"])
            and row["weighted_rms_obsid_correction_dispersion_keV"] >= 0
            for row in effective
        ),
    }
    clean_integrated = dict(integrated)
    clean_integrated.pop("scratch_products")
    return {
        "status": AUTHORIZED_STATUS if all(gates.values()) else "bullet_gain_weight_transport_gate_failed",
        "observed_fe_energy_keV": observed_fe,
        "integrated_spectrum": clean_integrated,
        "region_obsid_fe_weights": weights_artifact,
        "effective_gain_by_region": gain_artifact,
        "effective_gain_summary": {
            "minimum_equivalent_velocity_uncertainty_km_s": min(row["one_sigma_equivalent_velocity_uncertainty_km_s"] for row in effective),
            "median_equivalent_velocity_uncertainty_km_s": float(np.median([row["one_sigma_equivalent_velocity_uncertainty_km_s"] for row in effective])),
            "maximum_equivalent_velocity_uncertainty_km_s": max(row["one_sigma_equivalent_velocity_uncertainty_km_s"] for row in effective),
            "minimum_obsid_correction_dispersion_km_s": min(
                row["weighted_rms_obsid_correction_dispersion_km_s"] for row in effective
            ),
            "median_obsid_correction_dispersion_km_s": float(
                np.median([row["weighted_rms_obsid_correction_dispersion_km_s"] for row in effective])
            ),
            "maximum_obsid_correction_dispersion_km_s": max(
                row["weighted_rms_obsid_correction_dispersion_km_s"] for row in effective
            ),
        },
        "integrated_weight_equivalence": equivalence,
        "gates": gates,
        "bullet_source_redshift_fitter_authorized": all(gates.values()),
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    parents = validate_frozen(config)
    _, plan = load_plan(config, parents)
    return {
        "status": "v19dd_payload_blind_gain_weight_plan_passed",
        "regions": len(plan),
        "response_cells": sum(len(region["cells"]) for region in plan),
        "source_pha_or_arf_payload_opened": False,
        "source_redshift_fitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = load_json(config_path)
    try:
        result = preflight(config) if args.preflight_only else execute(config, output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "v19dd_execution_failed_closed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "bullet_source_redshift_fitter_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "source_line_temperature_abundance_redshift_or_velocity_fitted": False,
        "obsid554_or_abell2146_opened": False,
        "lensing_halo_gravity_or_action_opened": False,
    }
    report_path = output / ("preflight_report.json" if args.preflight_only else "report.json")
    atomic_json(report_path, report)
    print(json.dumps({key: report.get(key) for key in ("status", "execution_exception")}, indent=2, sort_keys=True))
    required = "v19dd_payload_blind_gain_weight_plan_passed" if args.preflight_only else AUTHORIZED_STATUS
    if report["status"] != required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
