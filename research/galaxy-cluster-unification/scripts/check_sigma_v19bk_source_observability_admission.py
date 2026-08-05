#!/usr/bin/env python3
"""Audit which registered V19BJ source invariants are actually observable."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bk_source_observability_admission.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_parents(config: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19BK parent hash mismatch: {name}")
        output[name] = actual
    return output


def inspect_collisionless_map(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as handle:
        names = [hdu.name for hdu in handle]
        required = {"LUM_MEAN", "JLOS_MEAN", "PLOS_MEAN", "ANALYSIS_MASK"}
        if not required.issubset(names):
            raise RuntimeError(f"collisionless map lacks required HDUs: {path}")
        luminosity = np.asarray(handle["LUM_MEAN"].data, dtype=float)
        current = np.asarray(handle["JLOS_MEAN"].data, dtype=float)
        second = np.asarray(handle["PLOS_MEAN"].data, dtype=float)
        mask = np.asarray(handle["ANALYSIS_MASK"].data, dtype=bool)
        units = {
            name: str(handle[name].header.get("BUNIT", ""))
            for name in ("LUM_MEAN", "JLOS_MEAN", "PLOS_MEAN")
        }
    finite = np.isfinite(luminosity) & np.isfinite(current) & np.isfinite(second)
    cauchy = luminosity * second - current**2
    scale = np.maximum(luminosity * second, np.finfo(float).tiny)
    normalized_margin = np.divide(
        cauchy,
        scale,
        out=np.zeros_like(cauchy),
        where=scale > np.finfo(float).tiny,
    )
    return {
        "shape_yx": list(luminosity.shape),
        "hdu_names": names,
        "units": units,
        "all_primary_moments_finite": bool(np.all(finite)),
        "luminosity_nonnegative": bool(np.all(luminosity >= 0.0)),
        "second_moment_nonnegative": bool(np.all(second >= 0.0)),
        "minimum_normalized_cauchy_margin": float(np.min(normalized_margin)),
        "analysis_mask_pixels": int(np.count_nonzero(mask)),
        "transverse_current_hdu_present": any(
            "EAST" in name or "NORTH" in name or "TRANS" in name for name in names
        ),
    }


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    implementation = config["implementation"]
    runner = ROOT / implementation["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19BK configuration names another runner")
    if sha256(runner) != implementation["runner_sha256"]:
        raise RuntimeError("V19BK runner changed after freeze")
    parent_hashes = verify_parents(config)
    paths = {name: ROOT / spec["path"] for name, spec in config["parents"].items()}
    bullet_report = load_json(paths["bullet_collisionless_report"])
    abell_report = load_json(paths["abell_collisionless_report"])
    v19bd = load_json(paths["directional_source_report"])
    front_failure = load_json(paths["front_fixture_failure"])
    x4 = load_json(paths["v19x4_preflight_report"])
    maps = {
        "BULLET": inspect_collisionless_map(paths["bullet_collisionless_maps"]),
        "ABELL2146": inspect_collisionless_map(paths["abell_collisionless_maps"]),
    }
    matrix = {row["id"]: row for row in config["observability_matrix"]}
    withheld = {"I1_COMPONENT_OVERLAP", "I2_RELATIVE_CURRENT", "I3_ANISOTROPIC_STRESS", "I6_CAUSAL_RELAXATION_RATE"}
    eligible = {
        row["id"] for row in config["observability_matrix"] if row["eligible_as_new_source"]
    }
    gates = {
        "all_parent_hashes_exact": bool(parent_hashes),
        "both_collisionless_moment_reports_pass": (
            bullet_report["decision"] == "passed"
            and abell_report["decision"] == "passed"
            and all(bullet_report["gate_results"].values())
            and all(abell_report["gate_results"].values())
        ),
        "moment_maps_are_finite_nonnegative_and_cauchy_consistent": all(
            row["all_primary_moments_finite"]
            and row["luminosity_nonnegative"]
            and row["second_moment_nonnegative"]
            and row["minimum_normalized_cauchy_margin"] >= -1.0e-12
            for row in maps.values()
        ),
        "only_line_of_sight_collisionless_moments_exist": all(
            not row["transverse_current_hdu_present"] for row in maps.values()
        ),
        "directional_parent_forbids_cross_filter_amplitudes_and_transverse_imputation": (
            v19bd["decision"] == "passed"
            and not v19bd["cross_filter_luminosity_amplitudes_compared"]
            and not v19bd["missing_luminosity_or_transverse_velocity_imputed"]
        ),
        "I1_I2_I3_I6_are_withheld": all(
            matrix[name]["status"] == "withheld"
            and not matrix[name]["eligible_as_new_source"]
            for name in withheld
        ),
        "only_I4_and_I5_remain_conditionally_eligible": eligible
        == {"I4_THERMODYNAMIC_GRADIENT_STRESS", "I5_BAROCLINICITY"},
        "I5_is_scalar_only": "scalar activation only" in matrix[
            "I5_BAROCLINICITY"
        ]["role"],
        "front_route_stopped_before_cluster_science_reopen": (
            front_failure["status"] == "mandatory_pre_science_fixture_failure"
            and front_failure["cluster_science_array_read"] is False
            and config["authorization"]["run_fourth_front_detector"] is False
        ),
        "v19x4_preflight_passed_without_observed_spectra": (
            x4["status"] == "gas_state_math_preflight_passed_awaiting_v19x3_measurements"
            and x4["observed_regional_spectra_opened"] is False
            and all(x4["gates"].values())
        ),
        "density_control_is_region_level_and_cross_validated": (
            config["density_novelty_control"]["unit_of_scoring"]
            == "one V19M adaptive gas region, not one common-grid pixel"
            and "leave-one-region-out" in config["density_novelty_control"]["cross_validation"]
            and config["density_novelty_control"][
                "required_cross_validated_residual_variance_fraction"
            ]
            >= 0.20
        ),
        "lensing_halo_action_and_gravity_selection_remain_sealed": (
            not config["authorization"]["read_lensing_or_halo_payload"]
            and not config["authorization"]["select_action_or_gravity_parameter"]
            and not config["authorization"]["open_holdout"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "decision": (
            "passed_source_observability_restricted_to_I4_direction_and_I5_scalar"
            if all(gates.values())
            else "failed_closed"
        ),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "parent_hashes": parent_hashes,
        "collisionless_map_evidence": maps,
        "eligible_source_ids_after_observability_audit": sorted(eligible),
        "withheld_source_ids": sorted(withheld),
        "density_novelty_control": config["density_novelty_control"],
        "front_measurement_decision": config["front_measurement_decision"],
        "gates": gates,
        "observed_v19x4_gas_posterior_opened": False,
        "invariant_score_computed": False,
        "lensing_or_halo_payload_opened": False,
        "action_or_gravity_parameter_selected": False,
        "claim_boundary": config["claim_boundary"],
    }


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    report = build_report(config_path)
    config = load_json(config_path)
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["decision"] == "failed_closed":
        raise RuntimeError(f"V19BK failed closed: {report['gates']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
