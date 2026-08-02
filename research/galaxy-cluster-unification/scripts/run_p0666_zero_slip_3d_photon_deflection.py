#!/usr/bin/env python3
"""Validate 3D baryonic activation and zero-slip photon normalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0660_exact_tensor_activation_audit import sha256

from voidscreen.field_solvers import solve_newtonian
from voidscreen.metric_lensing_3d import (
    KPC_M,
    M_SUN_KG,
    constitutive_tensor_components_3d,
    exact_tensor_activation_3d,
    lift_surface_density_msun_kpc2_to_si_volume,
    normalized_deflection_curl,
    photon_deflection_zero_slip,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0666_zero_slip_3d_photon_deflection.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gaussian_3d(axis, center, sigma, mass, spacing):
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    values = np.exp(
        -0.5
        * (
            ((x - center[0]) / sigma) ** 2
            + ((y - center[1]) / sigma) ** 2
            + ((z - center[2]) / sigma) ** 2
        )
    )
    return values * float(mass) / (float(np.sum(values)) * float(spacing) ** 3)


def point_mass_audit(protocol):
    analytic = protocol["analytic_tests"]
    cells = int(analytic["point_mass_grid_cells"])
    half_width = float(analytic["point_mass_half_width"])
    axis = np.linspace(-half_width, half_width, cells)
    spacing = float(axis[1] - axis[0])
    density = gaussian_3d(
        axis,
        (0.0, 0.0, 0.0),
        float(analytic["point_mass_gaussian_sigma"]),
        1.0,
        spacing,
    )
    gravitational_constant = float(analytic["point_mass_gravitational_constant"])
    light_speed = float(analytic["point_mass_light_speed"])
    field = solve_newtonian(
        density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    deflection = photon_deflection_zero_slip(
        field.acceleration,
        spacing,
        light_speed=light_speed,
    )
    center = cells // 2
    radius = axis[center:]
    measured = deflection.alpha_x_radian[center:, center]
    expected = np.zeros_like(radius)
    expected[1:] = 4.0 * gravitational_constant / (radius[1:] * light_speed**2)
    lower, upper = (float(value) for value in analytic["point_mass_impact_range"])
    active = (radius >= lower) & (radius <= upper)
    relative = np.abs(measured[active] / expected[active] - 1.0)
    rows = pd.DataFrame(
        {
            "impact_parameter": radius[active],
            "measured_deflection_radian": measured[active],
            "expected_GR_deflection_radian": expected[active],
            "relative_error": relative,
        }
    )
    doubled_field = solve_newtonian(
        2.0 * density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    doubled = photon_deflection_zero_slip(
        doubled_field.acceleration,
        spacing,
        light_speed=light_speed,
    )
    mass_scaling_error = float(
        np.sqrt(np.mean((doubled.alpha_x_radian - 2.0 * deflection.alpha_x_radian) ** 2))
        / max(
            float(np.sqrt(np.mean((2.0 * deflection.alpha_x_radian) ** 2))),
            np.finfo(float).tiny,
        )
    )
    curl = normalized_deflection_curl(
        deflection.alpha_x_radian,
        deflection.alpha_y_radian,
        spacing,
    )
    return {
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "mass_scaling_relative_error": mass_scaling_error,
        "normalized_curl_RMS": curl,
        "rows": rows,
        "density": density,
        "axis": axis,
        "spacing": spacing,
    }


def rotation_audit(point):
    axis = point["axis"]
    spacing = point["spacing"]
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    density = np.exp(-0.5 * ((x / 0.8) ** 2 + (y / 1.4) ** 2 + (z / 1.8) ** 2))
    density /= float(np.sum(density) * spacing**3)
    first_field = solve_newtonian(density, spacing, gravitational_constant=1.0)
    second_field = solve_newtonian(np.swapaxes(density, 0, 1), spacing, gravitational_constant=1.0)
    first = photon_deflection_zero_slip(first_field.acceleration, spacing, light_speed=1.0)
    second = photon_deflection_zero_slip(second_field.acceleration, spacing, light_speed=1.0)
    rotated_x = np.swapaxes(second.alpha_y_radian, 0, 1)
    rotated_y = np.swapaxes(second.alpha_x_radian, 0, 1)
    numerator = float(
        np.sqrt(
            np.mean(
                (rotated_x - first.alpha_x_radian) ** 2
                + (rotated_y - first.alpha_y_radian) ** 2
            )
        )
    )
    denominator = float(
        np.sqrt(np.mean(first.alpha_x_radian**2 + first.alpha_y_radian**2))
    )
    return numerator / max(denominator, np.finfo(float).tiny)


def activation_audits(protocol):
    analytic = protocol["analytic_tests"]
    cells = int(analytic["activation_grid_cells"])
    half_width = float(analytic["activation_half_width"])
    axis = np.linspace(-half_width, half_width, cells)
    spacing = float(axis[1] - axis[0])
    star_mass = float(analytic["activation_stellar_mass"])
    gas_mass = float(analytic["activation_gas_mass"])
    star_sigma = float(analytic["activation_stellar_gaussian_sigma"])
    gas_sigma = float(analytic["activation_gas_gaussian_sigma"])
    offset = float(analytic["activation_offset"])
    stars = gaussian_3d(axis, (0.0, 0.0, 0.0), star_sigma, star_mass, spacing)
    radial_gas = gaussian_3d(axis, (0.0, 0.0, 0.0), gas_sigma, gas_mass, spacing)
    offset_gas = gaussian_3d(axis, (offset, 0.0, 0.0), gas_sigma, gas_mass, spacing)
    settings = {
        "gravitational_constant": 1.0,
        "a0": float(analytic["activation_dimensionless_a0"]),
        "coherence_length": float(analytic["activation_dimensionless_coherence_length"]),
        "coherence_power": 2.0,
    }
    radial = exact_tensor_activation_3d(stars, radial_gas, spacing, **settings)
    displaced = exact_tensor_activation_3d(stars, offset_gas, spacing, **settings)

    def weighted(values, density):
        return float(np.sum(values * density) / np.sum(density))

    radial_weighted = weighted(radial.sigma, stars + radial_gas)
    offset_weighted = weighted(displaced.sigma, stars + offset_gas)
    direct = constitutive_tensor_components_3d(
        displaced.sigma,
        displaced.transport_direction,
    )
    reversed_tensor = constitutive_tensor_components_3d(
        displaced.sigma,
        tuple(-component for component in displaced.transport_direction),
    )
    numerator = np.sqrt(
        sum(
            float(np.mean((left - right) ** 2))
            for left, right in zip(direct, reversed_tensor, strict=True)
        )
    )
    denominator = np.sqrt(sum(float(np.mean(component**2)) for component in direct))
    return {
        "radial_sigma_mass_weighted_mean": radial_weighted,
        "offset_sigma_mass_weighted_mean": offset_weighted,
        "sigma_minimum": float(np.min(displaced.sigma)),
        "sigma_maximum": float(np.max(displaced.sigma)),
        "minimum_constitutive_eigenvalue_proxy": float(
            np.min(displaced.minimum_eigenvalue_proxy)
        ),
        "direction_reversal_tensor_relative_error": float(
            numerator / max(denominator, np.finfo(float).tiny)
        ),
    }


def density_lift_audit():
    axis = np.linspace(-4.0, 4.0, 33)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    surface = np.exp(-0.5 * ((xx / 0.7) ** 2 + (yy / 1.2) ** 2)) * 1.0e8
    z = np.linspace(-8.0, 8.0, 65)
    volume, scale_height = lift_surface_density_msun_kpc2_to_si_volume(
        surface,
        z,
        cell_kpc=float(axis[1] - axis[0]),
    )
    reconstructed_surface_si = np.sum(volume, axis=2) * float(z[1] - z[0]) * KPC_M
    expected_surface_si = surface * M_SUN_KG / KPC_M**2
    error = float(
        np.sqrt(np.mean((reconstructed_surface_si - expected_surface_si) ** 2))
        / np.sqrt(np.mean(expected_surface_si**2))
    )
    return error, scale_height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0666_photon_score":
        raise RuntimeError("P0666 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0665 parent no longer passes")

    point = point_mass_audit(protocol)
    rotation_error = rotation_audit(point)
    activation = activation_audits(protocol)
    lift_error, scale_height = density_lift_audit()
    gates = protocol["predeclared_progression_gates"]
    metric = protocol["metric_closure"]
    definitions = protocol["three_dimensional_activation"]
    bounded = bool(
        np.isfinite(activation["sigma_minimum"])
        and np.isfinite(activation["sigma_maximum"])
        and activation["sigma_minimum"] >= 0.0
        and activation["sigma_maximum"] <= 1.0
    )
    gate_results = {
        "P0665_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0665_all_progression_gates_pass"]),
        "point_mass_median": point["median_relative_error"]
        <= gates["point_mass_GR_deflection_median_relative_error_max"],
        "point_mass_p95": point["p95_relative_error"]
        <= gates["point_mass_GR_deflection_p95_relative_error_max"],
        "linear_mass_scaling": point["mass_scaling_relative_error"]
        <= gates["deflection_linear_mass_scaling_relative_error_max"],
        "rotation_covariance": rotation_error
        <= gates["deflection_rotation_covariance_relative_RMS_error_max"],
        "curl_free_deflection": point["normalized_curl_RMS"]
        <= gates["normalized_deflection_curl_RMS_max"],
        "surface_to_volume_mass": lift_error
        <= gates["surface_to_volume_component_mass_relative_error_max"],
        "radial_activation_null": activation["radial_sigma_mass_weighted_mean"]
        <= gates["radial_cocentered_3D_sigma_mass_weighted_mean_max"],
        "offset_activation_present": activation["offset_sigma_mass_weighted_mean"]
        >= gates["offset_3D_sigma_mass_weighted_mean_min"],
        "bounded_sigma": bounded is bool(gates["sigma_finite_and_in_closed_unit_interval"]),
        "positive_eigenvalue": bool(activation["minimum_constitutive_eigenvalue_proxy"] > 0.0)
        is bool(gates["minimum_constitutive_eigenvalue_strictly_positive"]),
        "direction_reversal": activation["direction_reversal_tensor_relative_error"]
        <= gates["direction_reversal_tensor_relative_error_max"],
        "zero_slip": float(metric["gravitational_slip"])
        == float(gates["gravitational_slip_exactly_zero"]),
        "no_fitted_photon_amplitude": bool(metric["fitted_photon_amplitude"])
        is bool(gates["fitted_photon_amplitude"]),
        "no_new_constants": int(definitions["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(definitions["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "spent_lensing_untouched": not bool(gates["spent_lensing_outcomes_opened"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics = {
        "point_mass_GR_deflection_median_relative_error": point["median_relative_error"],
        "point_mass_GR_deflection_p95_relative_error": point["p95_relative_error"],
        "deflection_linear_mass_scaling_relative_error": point["mass_scaling_relative_error"],
        "deflection_rotation_covariance_relative_RMS_error": rotation_error,
        "normalized_deflection_curl_RMS": point["normalized_curl_RMS"],
        "surface_to_volume_component_mass_relative_error": lift_error,
        "derived_projected_RMS_scale_height_kpc": scale_height,
        **activation,
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0666-ZERO-SLIP-3D-PHOTON-DEFLECTION-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_RXJ2129_map_build": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "metric_source_sha256": sha256(ROOT / "src/voidscreen/metric_lensing_3d.py"),
        "metrics": metrics,
        "gate_results": gate_results,
        "fitted_photon_amplitude_parameters": 0,
        "spent_RXJ2129_lensing_outcomes_opened": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    point["rows"].to_csv(output / "point_mass_deflection.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    rows = point["rows"]
    axes[0].plot(rows.impact_parameter, rows.expected_GR_deflection_radian, label="4GM/bc^2")
    axes[0].plot(rows.impact_parameter, rows.measured_deflection_radian, "o", label="3D solve")
    axes[0].set_xlabel("impact parameter")
    axes[0].set_ylabel("deflection (radian)")
    axes[0].legend()
    axes[1].bar(
        ["radial", "offset"],
        [
            activation["radial_sigma_mass_weighted_mean"],
            activation["offset_sigma_mass_weighted_mean"],
        ],
        color=["#3274a1", "#d95f02"],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("3D mass-weighted sigma")
    axes[1].set_title("3D component geometry")
    figure.suptitle("P0666 zero-slip photon deflection")
    figure.tight_layout()
    figure.savefig(output / "p0666_zero_slip_photon_deflection.png", dpi=180)
    plt.close(figure)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0666 zero-slip 3D photon deflection

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Point-mass GR median/p95 deflection errors: **{point['median_relative_error']:.3%} / {point['p95_relative_error']:.3%}**.
- Rotation/curl errors: **{rotation_error:.3e} / {point['normalized_curl_RMS']:.3e}**.
- Radial/offset 3D mass-weighted sigma: **{activation['radial_sigma_mass_weighted_mean']:.3e} / {activation['offset_sigma_mass_weighted_mean']:.3e}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Spent and sealed lensing outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
