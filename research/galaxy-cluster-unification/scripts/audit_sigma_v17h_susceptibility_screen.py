"""Audit the v17H aether-susceptibility pressure screen before data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def susceptibility(z: np.ndarray | float) -> np.ndarray:
    values = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Z must be finite and non-negative")
    return np.power(1.0 + values, -0.25)


def homogeneous_response(g_over_a_sigma: np.ndarray | float) -> np.ndarray:
    ratio = np.asarray(g_over_a_sigma, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("g/a_sigma must be finite and non-negative")
    return np.reciprocal(np.sqrt(1.0 + np.square(ratio)))


def effective_aether_density(z: np.ndarray | float) -> np.ndarray:
    values = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Z must be finite and non-negative")
    # The rationalized form preserves the small-Z limit without cancellation.
    return values / (np.sqrt(1.0 + values) + 1.0)


def acceleration_hessian_eigenvalues(
    z: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Z must be finite and non-negative")
    tangential = np.reciprocal(np.sqrt(1.0 + values))
    radial = np.power(1.0 + values, -1.5)
    return tangential, radial


def pressure_weighted_uniform_sphere_susceptibility(
    surface_g_over_a_sigma: float,
) -> float:
    """Pressure-weight chi for p proportional to 1-r^2 in a uniform sphere."""

    ratio = float(surface_g_over_a_sigma)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("surface acceleration ratio must be finite and non-negative")

    def weight(x: float) -> float:
        return x**2 * (1.0 - x**2)

    denominator = quad(weight, 0.0, 1.0, epsabs=1e-14, epsrel=1e-13)[0]
    numerator = quad(
        lambda x: weight(x) * float(susceptibility((ratio * x) ** 2)),
        0.0,
        1.0,
        epsabs=1e-14,
        epsrel=1e-11,
        limit=500,
    )[0]
    return float(numerator / denominator)


def minimum_alpha(required_fraction: float, pressure_compactness: float, response: float) -> float:
    values = (float(required_fraction), float(pressure_compactness), float(response))
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("alpha inputs must be finite and positive")
    return math.sqrt(values[0] / (values[1] * values[2]))


def build_report(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    fixed = config["fixed_theory_choices"]
    cluster = config["cluster_selection_envelope"]
    solar = config["solar_control"]
    constants = config["physical_constants"]
    health = config["local_health_scan"]
    gates = config["gates"]
    a_sigma = float(fixed["a_sigma_m_s2"])
    c_u = float(fixed["maxwell_aether_coefficient_c_U"])

    ratios = np.asarray(cluster["representative_g_over_a_sigma"], dtype=float)
    responses = homogeneous_response(ratios)
    alphas = np.asarray(
        [
            minimum_alpha(
                cluster["minimum_extra_Weyl_fraction"],
                cluster["pressure_compactness"],
                response,
            )
            for response in responses
        ]
    )
    cluster_rows = [
        {
            "g_over_a_sigma": float(ratio),
            "homogeneous_source_times_metric_response": float(response),
            "minimum_alpha_for_unit_extra_Weyl": float(alpha),
        }
        for ratio, response, alpha in zip(ratios, responses, alphas, strict=True)
    ]

    gravity_surface = (
        constants["gravitational_constant_m3_kg_s2"]
        * solar["mass_kg"]
        / solar["radius_m"] ** 2
    )
    surface_ratio = gravity_surface / a_sigma
    source_chi = pressure_weighted_uniform_sphere_susceptibility(surface_ratio)
    path_radius_m = solar["maximum_radio_path_radius_au"] * constants["metres_per_au"]
    path_g = (
        constants["gravitational_constant_m3_kg_s2"]
        * solar["mass_kg"]
        / path_radius_m**2
    )
    path_chi = float(susceptibility((path_g / a_sigma) ** 2))

    largest_selection_alpha = float(np.max(alphas))
    solar_gamma = (
        2.0
        * largest_selection_alpha**2
        * solar["conservative_pressure_compactness"]
        * source_chi
        * path_chi
    )
    cassini_safe_alpha = math.sqrt(
        solar["cassini_absolute_gamma_minus_one_max"]
        /
        (
            2.0
            * solar["conservative_pressure_compactness"]
            * source_chi
            * path_chi
        )
    )

    z_positive = np.geomspace(
        health["Z_minimum_positive"],
        health["Z_maximum"],
        int(health["Z_samples"]),
    )
    z_scan = np.concatenate(([0.0], z_positive))
    tangential, radial = acceleration_hessian_eigenvalues(z_scan)
    total_tangential = c_u + tangential
    total_radial = c_u + radial
    high_response = float(homogeneous_response(1.0e5))
    low_response = float(homogeneous_response(0.1))

    gate_results = {
        "high_acceleration_response_pass": high_response
        <= gates["maximum_response_at_g_over_a_sigma_1e5"],
        "low_acceleration_retention_pass": low_response
        >= gates["minimum_response_at_g_over_a_sigma_0p1"],
        "solar_path_proxy_pass": solar_gamma <= gates["maximum_solar_gamma_minus_one"],
        "raw_acceleration_block_positive": float(min(np.min(tangential), np.min(radial)))
        > gates["minimum_raw_acceleration_Hessian_eigenvalue"],
        "maxwell_floored_block_pass": float(
            min(np.min(total_tangential), np.min(total_radial))
        )
        >= gates["minimum_total_acceleration_Hessian_eigenvalue"],
    }
    analytic_selection_pass = all(gate_results.values())

    return {
        "report_version": config["protocol_version"],
        "status": (
            "passed_conditional_action_selection"
            if analytic_selection_pass
            else "failed_action_selection"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "observational_data_opened": False,
        "empirical_fit_performed": False,
        "analytic_identity": {
            "F_A": "sqrt(1+Z)-1",
            "chi": "(1+Z)^(-1/4)",
            "chi_squared": "2 dF_A/dZ=1/sqrt(1+Z)",
            "susceptibility_at_zero": float(susceptibility(0.0)),
            "response_at_g_over_a_sigma_1e5": high_response,
            "response_at_g_over_a_sigma_0p1": low_response,
        },
        "cluster_theory_envelope": cluster_rows,
        "solar_uniform_sphere_control": {
            "surface_gravity_m_s2": gravity_surface,
            "surface_g_over_a_sigma": surface_ratio,
            "pressure_weighted_source_chi": source_chi,
            "maximum_path_radius_au": solar["maximum_radio_path_radius_au"],
            "maximum_path_chi": path_chi,
            "largest_cluster_envelope_alpha_tested": largest_selection_alpha,
            "effective_gamma_minus_one_upper_bound": solar_gamma,
            "fraction_of_Cassini_limit": solar_gamma
            / solar["cassini_absolute_gamma_minus_one_max"],
            "maximum_Cassini_safe_alpha_in_this_control": cassini_safe_alpha,
            "is_full_PPN_calculation": False,
        },
        "reduced_local_health": {
            "minimum_raw_tangential_eigenvalue": float(np.min(tangential)),
            "minimum_raw_radial_eigenvalue": float(np.min(radial)),
            "minimum_total_tangential_eigenvalue": float(np.min(total_tangential)),
            "minimum_total_radial_eigenvalue": float(np.min(total_radial)),
            "raw_condition_number_at_Z_max": float(
                tangential[-1] / radial[-1]
            ),
            "maxwell_floor_c_U": c_u,
            "full_principal_symbol_computed": False,
        },
        "gates": {
            **gate_results,
            "analytic_selection_pass": analytic_selection_pass,
            "full_covariant_variation_pass": False,
            "full_PPN_pass": False,
            "holdout_authorized": False,
        },
        "decision": {
            "outcome": (
                "advance_to_exact_variation_only"
                if analytic_selection_pass
                else "retire_susceptibility_screen"
            ),
            "scope": (
                "The fixed susceptibility removes the v17G amplitude obstruction in a "
                "leading Solar control and keeps a positive reduced aether acceleration "
                "block. It is not a complete constraint, causality, PPN, or data result."
            ),
            "empirical_requirement": (
                "Derive Z from a target-blind three-dimensional baryonic/aether solution "
                "before applying the pressure source to any cluster lensing target."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17h_susceptibility_screened_pressure.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17h_susceptibility_screen",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_report(args.config, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
