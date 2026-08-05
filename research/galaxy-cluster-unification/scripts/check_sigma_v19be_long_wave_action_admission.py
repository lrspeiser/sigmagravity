#!/usr/bin/env python3
"""Audit the action-level admission rules for a long-wavelength Sigma mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19be_long_wave_action_admission.json"

AU_M = 149_597_870_700.0
PARSEC_M = 3.0856775814913673e16
SPEED_OF_LIGHT_M_S = 299_792_458.0
JULIAN_YEAR_S = 31_557_600.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_parent_hashes(config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Path]]:
    hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"parent hash mismatch for {name}: {actual} != {spec['sha256']}")
        hashes[name] = actual
        paths[name] = path
    return hashes, paths


def scale_diagnostics(scale: dict[str, float]) -> dict[str, float]:
    length_m = float(scale["illustrative_correlation_length_kpc"]) * 1_000.0 * PARSEC_M
    wavelength_m = 2.0 * math.pi * length_m
    baseline_m = float(scale["solar_baseline_au"]) * AU_M
    galaxy_radius_m = float(scale["galaxy_radius_kpc"]) * 1_000.0 * PARSEC_M
    x = galaxy_radius_m / length_m
    return {
        "literal_wavelength_kpc": wavelength_m / (1_000.0 * PARSEC_M),
        "baseline_phase_change_rad": 2.0 * math.pi * baseline_m / wavelength_m,
        "literal_wave_tidal_scale": (baseline_m / wavelength_m) ** 2,
        "sourced_low_pass_small_baseline_scale": 0.5 * (baseline_m / length_m) ** 2,
        "sourced_low_pass_activation_at_galaxy_radius": 1.0 - (1.0 + x) * math.exp(-x),
        "light_crossing_time_years": wavelength_m
        / SPEED_OF_LIGHT_M_S
        / JULIAN_YEAR_S,
    }


def source_evidence(v19bd_report: dict[str, Any]) -> dict[str, Any]:
    cluster_summaries = v19bd_report["cluster_summaries"]
    paired = v19bd_report["paired_comparison_summary"]
    return {
        "normalized_second_offset_median": {
            cluster: float(
                cluster_summaries[cluster]["normalized_second_offset"]["percentiles"]["50.0"]
            )
            for cluster in ("BULLET", "ABELL2146")
        },
        "normalized_current_separation_median": {
            cluster: float(
                cluster_summaries[cluster]["normalized_current_separation"]["percentiles"][
                    "50.0"
                ]
            )
            for cluster in ("BULLET", "ABELL2146")
        },
        "paired_abell_minus_bullet_second_offset_median": float(
            paired["abell_minus_bullet_normalized_second_offset"]["percentiles"]["50.0"]
        ),
        "paired_abell_minus_bullet_current_separation_median": float(
            paired["abell_minus_bullet_normalized_current_separation"]["percentiles"]["50.0"]
        ),
        "interpretation": (
            "The scale-free second-moment displacement is similar in the two source ensembles, "
            "while the signed-current topology is not; a later direction must be computed from "
            "the measured source tensor rather than assigned as a universal merger template."
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    implementation = config["implementation"]
    runner_path = (ROOT / implementation["runner"]).resolve()
    if runner_path != Path(__file__).resolve():
        raise ValueError("frozen implementation path does not identify this runner")
    runner_hash = sha256(runner_path)
    if runner_hash != implementation["runner_sha256"]:
        raise ValueError("frozen implementation hash mismatch")

    parent_hashes, parent_paths = verify_parent_hashes(config)
    v19bd = json.loads(parent_paths["v19bd_report"].read_text(encoding="utf-8"))
    diagnostics = scale_diagnostics(config["illustrative_scale_check"])
    limits = config["gates"]["numeric_scale_separation"]

    requirements = config["action_admission_requirements"]
    gate_results = {
        "all_parent_hashes_exact": True,
        "v19bd_source_only_parent_passed": (
            v19bd["decision"] == "passed"
            and not v19bd["long_wave_operator_or_parameter_selected"]
            and not v19bd["lensing_halo_gas_response_or_gravity_payload_opened"]
        ),
        "one_physical_metric_required": requirements["one_physical_metric"],
        "diffeomorphism_invariant_action_required": requirements[
            "diffeomorphism_invariant_action"
        ],
        "total_metric_source_conservation_required": requirements[
            "total_metric_source_conservation"
        ],
        "universal_wavelength_no_object_fit_required": requirements[
            "universal_wavelength_no_object_fit"
        ],
        "no_free_halo_equivalent_homogeneous_mode_required": requirements[
            "no_free_halo_equivalent_homogeneous_mode"
        ],
        "quadratic_gr_limit_required": requirements["quadratic_gr_limit"],
        "matter_and_light_unified_required": requirements["matter_and_light_unified"],
        "literal_wave_tidal_scale_small": diagnostics["literal_wave_tidal_scale"]
        <= float(limits["maximum_literal_wave_tidal_scale"]),
        "sourced_low_pass_solar_scale_small": diagnostics[
            "sourced_low_pass_small_baseline_scale"
        ]
        <= float(limits["maximum_sourced_low_pass_small_baseline_scale"]),
        "galaxy_scale_activation_nontrivial": float(
            limits["minimum_galaxy_scale_activation"]
        )
        <= diagnostics["sourced_low_pass_activation_at_galaxy_radius"]
        <= float(limits["maximum_galaxy_scale_activation"]),
        "mode_is_quasistatic_over_human_observations": diagnostics["light_crossing_time_years"]
        >= float(limits["minimum_light_crossing_time_years"]),
        "no_action_operator_or_constant_selected": (
            not config["authorization"]["select_candidate_action"]
            and not config["authorization"]["select_long_wave_operator_or_constant"]
        ),
        "no_gas_lensing_halo_or_holdout_payload_authorized": (
            not config["authorization"]["read_v19w_or_v19x_gas_result"]
            and not config["authorization"]["read_lensing_or_halo_payload"]
            and not config["authorization"]["open_holdout"]
        ),
    }
    gate_results = {name: bool(value) for name, value in gate_results.items()}
    decision = "passed_action_admission_requirements" if all(gate_results.values()) else "failed_closed"

    report = {
        "protocol_version": config["protocol_version"],
        "decision": decision,
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "implementation": {
            "runner": implementation["runner"],
            "runner_sha256": runner_hash,
        },
        "input_hashes": parent_hashes,
        "scale_diagnostics": diagnostics,
        "source_evidence": source_evidence(v19bd),
        "gate_results": gate_results,
        "theory_state": {
            "physical_postulate_recorded": True,
            "covariant_action_selected": False,
            "euler_lagrange_equations_derived": False,
            "weak_field_metric_derived": False,
            "universal_constants_selected": False,
            "gas_source_state_available": False,
            "claim": (
                "V19BE admits only the nonlinear, baryon-sourced, conserved long-wave theory "
                "class for later derivation; it does not assert that any member of that class works."
            ),
        },
        "claim_boundary": config["claim_boundary"],
    }
    output_path = ROOT / config["outputs"]["report"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if decision == "failed_closed":
        raise RuntimeError(f"V19BE failed closed: {gate_results}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
