#!/usr/bin/env python3
"""Verify the frozen V19BL projected source-invariant scoring protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voidscreen.sigma_source_invariants import (
    analytic_press_unexplained_fraction,
    axial_angle_deg,
    axial_interval_summary_deg,
    central_gradient,
    central_hessian,
    projected_source_maps,
)

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bl_source_invariant_scoring_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hashes(config: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19BL parent hash mismatch: {name}")
        hashes[name] = actual
    implementation = config["implementation"]
    checker = ROOT / implementation["checker"]
    module = ROOT / implementation["math_module"]
    if checker.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19BL configuration names another checker")
    if sha256(checker) != implementation["checker_sha256"]:
        raise RuntimeError("V19BL checker changed after freeze")
    if sha256(module) != implementation["math_module_sha256"]:
        raise RuntimeError("V19BL math module changed after freeze")
    return hashes


def manufactured_checks() -> dict[str, Any]:
    axis = np.arange(-5.0, 6.0)
    east, north = np.meshgrid(axis, axis)
    polynomial = 3.0 * east**2 + 2.0 * east * north + 5.0 * north**2
    gradient_east, gradient_north = central_gradient(polynomial, 1.0)
    hessian_ee, hessian_nn, hessian_en = central_hessian(polynomial, 1.0)
    interior = np.s_[1:-1, 1:-1]
    derivative_error = max(
        float(np.max(np.abs(gradient_east[interior] - (6.0 * east + 2.0 * north)[interior]))),
        float(np.max(np.abs(gradient_north[interior] - (2.0 * east + 10.0 * north)[interior]))),
        float(np.max(np.abs(hessian_ee[interior] - 6.0))),
        float(np.max(np.abs(hessian_nn[interior] - 10.0))),
        float(np.max(np.abs(hessian_en[interior] - 2.0))),
    )
    density = np.exp(0.03 * east)
    entropy = np.exp(0.04 * east)
    pressure_parallel = np.exp(0.05 * east)
    pressure_perpendicular = np.exp(0.05 * north)
    surface = np.exp(0.02 * east + 0.01 * north)
    parallel = projected_source_maps(
        density,
        entropy,
        pressure_parallel,
        surface,
        spacing_kpc=1.0,
        resolution_fwhm_kpc=10.0,
    )
    perpendicular = projected_source_maps(
        density,
        entropy,
        pressure_perpendicular,
        surface,
        spacing_kpc=1.0,
        resolution_fwhm_kpc=10.0,
    )
    center = (5, 5)
    rng = np.random.default_rng(1902)
    predictors = rng.normal(size=(160, 5))
    response = rng.normal(size=(160, 2))
    angle = math.radians(31.0)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    press = analytic_press_unexplained_fraction(predictors, response)
    rotated = analytic_press_unexplained_fraction(predictors, response @ rotation.T)
    press_rotation_error = abs(
        float(press["joint_unexplained_fraction"])
        - float(rotated["joint_unexplained_fraction"])
    )
    axial = axial_interval_summary_deg(np.asarray([178.0, 179.0, 0.0, 1.0, 2.0]))
    spin_two = axial_angle_deg(
        np.asarray([1.0, 0.0, -1.0]), np.asarray([0.0, 1.0, 0.0])
    )
    return {
        "maximum_polynomial_derivative_absolute_error": derivative_error,
        "parallel_gradient_baroclinicity": float(parallel["i5_baroclinicity"][center]),
        "perpendicular_gradient_baroclinicity": float(
            perpendicular["i5_baroclinicity"][center]
        ),
        "joint_PRESS_rotation_absolute_error": press_rotation_error,
        "axial_wrap_width_95_deg": axial["width_95_deg"],
        "spin_two_test_angles_deg": spin_two.tolist(),
    }


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    parent_hashes = verify_hashes(config)
    bj = load_json(ROOT / config["parents"]["v19bj_config"]["path"])
    bk = load_json(ROOT / config["parents"]["v19bk_config"]["path"])
    evidence = manufactured_checks()
    gates = config["robustness_and_decision_gates"]
    novelty = config["density_novelty_control"]
    support = config["gradient_support"]
    eligible = {
        row["id"]
        for row in bk["observability_matrix"]
        if row["eligible_as_new_source"]
    }
    checks = {
        "all_parent_and_implementation_hashes_exact": bool(parent_hashes),
        "observability_restricted_to_I4_and_I5": eligible
        == {"I4_THERMODYNAMIC_GRADIENT_STRESS", "I5_BAROCLINICITY"},
        "thresholds_inherit_or_strengthen_v19bj": (
            support["minimum_gradient_detection_sigma"]
            == bj["identifiability_gates"]["minimum_within_cluster_detection_sigma"]
            and gates["minimum_within_cluster_detection_sigma"]
            == bj["identifiability_gates"]["minimum_within_cluster_detection_sigma"]
            and gates["maximum_95_percent_I4_axial_orientation_width_deg"]
            == bj["identifiability_gates"]["maximum_95_percent_axial_orientation_width_deg"]
            and gates["maximum_resolution_change_in_orientation_deg"]
            == bj["identifiability_gates"]["maximum_resolution_change_in_orientation_deg"]
            and gates["maximum_resolution_change_in_activation_fraction"]
            == bj["identifiability_gates"]["maximum_resolution_change_in_normalized_amplitude_fraction"]
            and novelty["minimum_unexplained_variance_fraction"]
            == bj["identifiability_gates"]["minimum_cross_validated_variance_not_predicted_by_total_density_fraction"]
        ),
        "fixed_quadratic_density_control_has_21_coefficients": (
            "exactly 21 coefficients" in novelty["fixed_basis"]
            and support["minimum_supported_regions"] > 21
        ),
        "three_dependence_branches_two_scales_three_radii_frozen": (
            config["registered_inputs"]["temperature_normalization_rank_correlations"]
            == [-0.9, 0.0, 0.9]
            and config["registered_inputs"]["smoothing_fwhm_kpc"] == [50.0, 100.0]
            and config["registered_inputs"]["primary_radius_kpc"] == 350.0
            and config["registered_inputs"]["radius_robustness_kpc"] == [250.0, 500.0]
        ),
        "manufactured_derivatives_exact": evidence[
            "maximum_polynomial_derivative_absolute_error"
        ]
        <= 1.0e-12,
        "baroclinicity_parallel_zero_perpendicular_one": (
            evidence["parallel_gradient_baroclinicity"] <= 1.0e-20
            and abs(evidence["perpendicular_gradient_baroclinicity"] - 1.0)
            <= 1.0e-12
        ),
        "joint_tensor_PRESS_is_rotation_invariant": evidence[
            "joint_PRESS_rotation_absolute_error"
        ]
        <= 1.0e-12,
        "axial_wrap_and_spin_two_are_correct": (
            evidence["axial_wrap_width_95_deg"] < 5.0
            and np.allclose(evidence["spin_two_test_angles_deg"], [0.0, 45.0, 90.0])
        ),
        "all_branches_and_both_clusters_required": (
            gates["all_three_rank_correlation_branches_must_pass"]
            and gates["both_clusters_must_pass"]
        ),
        "lensing_action_gravity_and_holdouts_sealed": (
            not config["authorization"]["read_lensing_or_halo_payload"]
            and not config["authorization"]["select_action_or_gravity_parameter"]
            and not config["authorization"]["change_formula_or_constant"]
            and not config["authorization"]["open_holdout"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "decision": (
            "passed_source_invariant_scoring_math_frozen_awaiting_observed_v19x4"
            if all(checks.values())
            else "failed_closed"
        ),
        "config_sha256": sha256(config_path),
        "checker_sha256": sha256(Path(__file__).resolve()),
        "math_module_sha256": sha256(
            ROOT / config["implementation"]["math_module"]
        ),
        "parent_hashes": parent_hashes,
        "manufactured_evidence": evidence,
        "gates": checks,
        "observed_v19x4_gas_posterior_opened": False,
        "source_invariant_score_computed": False,
        "lensing_or_halo_payload_opened": False,
        "action_or_gravity_parameter_selected": False,
        "claim_boundary": config["claim_boundary"],
    }


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    report = build_report(config_path)
    config = load_json(config_path)
    output = ROOT / config["outputs"]["preflight_report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["decision"] == "failed_closed":
        raise RuntimeError(f"V19BL failed closed: {report['gates']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
