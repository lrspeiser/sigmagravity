#!/usr/bin/env python3
"""Run the frozen no-observation P0695 radial-path mathematics audit."""

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

from run_adaptive_route_raw_rxj2129 import json_safe
from run_p0660_exact_tensor_activation_audit import sha256

from voidscreen.field_solvers import (
    boundary_mask,
    cell_coordinates,
    simple_mond_acceleration,
    solve_newtonian,
    solve_poisson_dirichlet,
)
from voidscreen.radial_path_potential import (
    hybrid_path_routing_potential,
    normalized_acceleration_curl,
    radial_path_potential_from_newtonian,
)
from voidscreen.source_routing_qumond import (
    projected_baryonic_spectral_anisotropy,
    solve_source_conserving_baryonic_routing,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0695_radial_path_potential_math_audit.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_vector_rms(left, right, mask) -> float:
    numerator = float(
        np.sqrt(
            np.mean(
                sum((np.asarray(a)[mask] - np.asarray(b)[mask]) ** 2 for a, b in zip(left, right, strict=True))
            )
        )
    )
    denominator = float(
        np.sqrt(np.mean(sum(np.asarray(item)[mask] ** 2 for item in right)))
    )
    return numerator / max(denominator, np.finfo(float).tiny)


def relative_grid_rms(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    numerator = float(np.sqrt(np.mean((left[mask] - right[mask]) ** 2)))
    denominator = float(np.sqrt(np.mean(right[mask] ** 2)))
    return numerator / max(denominator, np.finfo(float).tiny)


def normalized_density(values: np.ndarray, spacing: float) -> np.ndarray:
    density = np.asarray(values, dtype=float)
    return density / float(np.sum(density) * spacing**3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    stage = "P0695B" if str(protocol.get("protocol_version", "")).startswith("P0695B-") else "P0695"
    accepted_status = {
        "P0695": "frozen_before_any_P0695_synthetic_path_or_hybrid_metric",
        "P0695B": "frozen_before_any_P0695B_synthetic_path_or_hybrid_metric",
    }[stage]
    if protocol.get("status") != accepted_status:
        raise RuntimeError(f"{stage} protocol is not frozen")
    failure_path = ROOT / protocol["failure_parent"]
    failure = read_json(failure_path)
    equation = protocol["equation"]
    expected = protocol["predeclared_integrity_gates"]
    if stage == "P0695B":
        integrity = {
            "P0695_status": failure.get("status") == expected["P0695_status"],
            "P0695_not_advanced": bool(
                failure.get("candidate_advanced_to_spent_joint_screen")
            )
            is bool(expected["P0695_candidate_advanced_to_spent_joint_screen"]),
            "P0695_failed_gates_reproduced": list(failure.get("failed_gates", []))
            == list(expected["P0695_failed_gates"]),
            "only_interpolation_changed": expected["only_declared_change"]
            == "interpolation_order: 1 -> 3",
            "no_observational_outcome_read": not bool(
                expected["galaxy_or_cluster_outcomes_read"]
            ),
            "no_new_constants": int(equation["new_universal_constants"])
            == int(expected["new_universal_constants"]),
            "no_fitted_physics": int(equation["fitted_physical_parameters"])
            == int(expected["fitted_physical_parameters"]),
            "sealed_targets_untouched": not bool(expected["sealed_target_outcomes_opened"]),
        }
    else:
        integrity = {
            "P0694_status": failure.get("status") == expected["P0694_status"],
            "P0694_endpoint_pair_retired": bool(
                failure.get("shared_linear_endpoint_pair_retired")
            )
            is bool(expected["P0694_shared_linear_endpoint_pair_retired"]),
            "no_observational_outcome_read": not bool(
                expected["galaxy_or_cluster_outcomes_read"]
            ),
            "no_new_constants": int(equation["new_universal_constants"])
            == int(expected["new_universal_constants"]),
            "no_fitted_physics": int(equation["fitted_physical_parameters"])
            == int(expected["fitted_physical_parameters"]),
            "sealed_targets_untouched": not bool(expected["sealed_target_outcomes_opened"]),
        }
    if not all(integrity.values()):
        raise RuntimeError(f"{stage} integrity failure before metrics: {integrity}")

    numerics = protocol["numerics"]
    cells = int(numerics["grid_cells"])
    spacing = float(numerics["spacing"])
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    radius = np.sqrt(x * x + y * y + z * z)
    sphere_scale = float(numerics["sphere_scale"])
    sphere_density = normalized_density(
        np.power(1.0 + (radius / sphere_scale) ** 2, -2.5),
        spacing,
    )
    ellipsoid_scales = np.asarray(numerics["ellipsoid_scales"], dtype=float)
    ellipsoid_density = normalized_density(
        np.power(
            1.0
            + (x / ellipsoid_scales[0]) ** 2
            + (y / ellipsoid_scales[1]) ** 2
            + (z / ellipsoid_scales[2]) ** 2,
            -2.5,
        ),
        spacing,
    )
    gravitational_constant = float(numerics["gravitational_constant"])
    a0 = float(numerics["a0"])
    primary_order = int(numerics["primary_gauss_legendre_order"])
    convergence_order = int(numerics["convergence_gauss_legendre_order"])
    interpolation_order = int(numerics["interpolation_order"])

    print(f"{stage}: spherical path and quadrature audit", flush=True)
    sphere_newtonian = solve_newtonian(
        sphere_density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    sphere_path = radial_path_potential_from_newtonian(
        sphere_density,
        sphere_newtonian.potential,
        sphere_newtonian.acceleration,
        spacing,
        a0=a0,
        quadrature_order=primary_order,
        interpolation_order=interpolation_order,
    )
    sphere_path_convergence = radial_path_potential_from_newtonian(
        sphere_density,
        sphere_newtonian.potential,
        sphere_newtonian.acceleration,
        spacing,
        a0=a0,
        quadrature_order=convergence_order,
        interpolation_order=interpolation_order,
    )
    safe_radius = np.maximum(radius, np.finfo(float).tiny)
    radial_unit = (x / safe_radius, y / safe_radius, z / safe_radius)
    newtonian_magnitude = np.sqrt(
        sum(component * component for component in sphere_newtonian.acceleration)
    )
    algebraic_magnitude = simple_mond_acceleration(newtonian_magnitude, a0)
    boost = np.divide(
        algebraic_magnitude,
        newtonian_magnitude,
        out=np.zeros_like(newtonian_magnitude),
        where=newtonian_magnitude > 0.0,
    )
    target_acceleration = tuple(
        boost * component for component in sphere_newtonian.acceleration
    )
    path_radial = -sum(
        component * direction
        for component, direction in zip(sphere_path.acceleration, radial_unit, strict=True)
    )
    target_radial = -sum(
        component * direction
        for component, direction in zip(target_acceleration, radial_unit, strict=True)
    )
    lower_cells, upper_cells = (
        float(value) for value in numerics["comparison_radius_grid_cells"]
    )
    comparison = (
        (radius >= lower_cells * spacing)
        & (radius <= upper_cells * spacing)
        & (target_radial > 0.0)
    )
    relative_error = (path_radial[comparison] - target_radial[comparison]) / target_radial[
        comparison
    ]
    radial_relative_rms = float(np.sqrt(np.mean(relative_error**2)))
    radial_median_absolute = float(np.median(np.abs(relative_error)))
    path_magnitude_squared = sum(component * component for component in sphere_path.acceleration)
    tangential_squared = np.maximum(path_magnitude_squared - path_radial**2, 0.0)
    tangential_to_radial = float(
        np.sqrt(np.mean(tangential_squared[comparison]))
        / max(float(np.sqrt(np.mean(path_radial[comparison] ** 2))), np.finfo(float).tiny)
    )
    shell_index = np.rint(radius / spacing).astype(int)
    radial_rows = []
    angular_scatter = []
    for shell in range(int(np.ceil(lower_cells)), int(np.floor(upper_cells)) + 1):
        shell_mask = comparison & shell_index.__eq__(shell)
        if np.count_nonzero(shell_mask) < 12:
            continue
        path_values = path_radial[shell_mask]
        target_values = target_radial[shell_mask]
        scatter = float(np.std(path_values) / max(abs(float(np.mean(path_values))), np.finfo(float).tiny))
        angular_scatter.append(scatter)
        radial_rows.append(
            {
                "radius_grid_cells": shell,
                "radius": shell * spacing,
                "cells": int(np.count_nonzero(shell_mask)),
                "path_radial_mean": float(np.mean(path_values)),
                "target_radial_mean": float(np.mean(target_values)),
                "mean_relative_error": float(np.mean(path_values / target_values - 1.0)),
                "median_absolute_relative_error": float(
                    np.median(np.abs(path_values / target_values - 1.0))
                ),
                "angular_scatter_fraction": scatter,
            }
        )
    maximum_angular_scatter = max(angular_scatter)
    sphere_curl = normalized_acceleration_curl(sphere_path.acceleration, spacing)
    quadrature_relative = relative_vector_rms(
        sphere_path.acceleration,
        sphere_path_convergence.acceleration,
        comparison,
    )

    print(f"{stage}: ellipsoid rotation covariance and hybrid identity", flush=True)
    ellipsoid_newtonian = solve_newtonian(
        ellipsoid_density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    ellipsoid_path = radial_path_potential_from_newtonian(
        ellipsoid_density,
        ellipsoid_newtonian.potential,
        ellipsoid_newtonian.acceleration,
        spacing,
        a0=a0,
        quadrature_order=primary_order,
        interpolation_order=interpolation_order,
    )
    rotated_density = np.swapaxes(ellipsoid_density, 0, 1)
    rotated_newtonian = solve_newtonian(
        rotated_density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    rotated_path = radial_path_potential_from_newtonian(
        rotated_density,
        rotated_newtonian.potential,
        rotated_newtonian.acceleration,
        spacing,
        a0=a0,
        quadrature_order=primary_order,
        interpolation_order=interpolation_order,
    )
    interior = np.zeros((cells,) * 3, dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True
    expected_rotated_potential = np.swapaxes(ellipsoid_path.potential, 0, 1)
    expected_rotated_acceleration = (
        np.swapaxes(ellipsoid_path.acceleration[1], 0, 1),
        np.swapaxes(ellipsoid_path.acceleration[0], 0, 1),
        np.swapaxes(ellipsoid_path.acceleration[2], 0, 1),
    )
    rotation_potential = relative_grid_rms(
        rotated_path.potential,
        expected_rotated_potential,
        interior,
    )
    rotation_acceleration = relative_vector_rms(
        rotated_path.acceleration,
        expected_rotated_acceleration,
        interior,
    )

    routing = solve_source_conserving_baryonic_routing(
        ellipsoid_density,
        spacing,
        gravitational_constant=gravitational_constant,
        a0=a0,
        transition_depth=float(numerics["transition_depth"]),
        transition_power=float(numerics["transition_power"]),
        extra_spatial_channels=float(numerics["extra_spatial_channels"]),
        path_power=float(numerics["path_power"]),
        light_speed=float(numerics["light_speed"]),
    )
    local_potential = solve_poisson_dirichlet(
        routing.local_generator_source,
        spacing,
        routing.boundary_potential,
    )
    projected_surface = np.sum(ellipsoid_density, axis=2) * spacing
    fraction, covariance, eigenvalues = projected_baryonic_spectral_anisotropy(
        projected_surface,
        spacing,
    )
    hybrid = hybrid_path_routing_potential(
        ellipsoid_path,
        local_potential,
        routing.field.potential,
        spacing,
        fraction,
    )
    expected_hybrid = ellipsoid_path.potential + fraction * (
        routing.field.potential - local_potential
    )
    hybrid_identity = relative_grid_rms(hybrid.potential, expected_hybrid, interior)
    edge = boundary_mask(ellipsoid_density.shape)
    correction_scale = max(
        float(np.max(np.abs(routing.field.potential[edge]))),
        np.finfo(float).tiny,
    )
    correction_boundary = float(
        np.max(np.abs((routing.field.potential - local_potential)[edge])) / correction_scale
    )
    ellipsoid_curl = normalized_acceleration_curl(ellipsoid_path.acceleration, spacing)
    rotated_curl = normalized_acceleration_curl(rotated_path.acceleration, spacing)
    hybrid_curl = normalized_acceleration_curl(hybrid.acceleration, spacing)
    maximum_curl = max(sphere_curl, ellipsoid_curl, rotated_curl, hybrid_curl)
    finite = bool(
        all(
            np.all(np.isfinite(array))
            for array in (
                sphere_path.potential,
                *sphere_path.acceleration,
                sphere_path.equation_source,
                sphere_path_convergence.potential,
                *sphere_path_convergence.acceleration,
                ellipsoid_path.potential,
                *ellipsoid_path.acceleration,
                rotated_path.potential,
                *rotated_path.acceleration,
                hybrid.potential,
                *hybrid.acceleration,
                hybrid.equation_source,
            )
        )
    )

    gates = protocol["predeclared_math_gates"]
    gate_results = {
        **integrity,
        "finite": finite is bool(gates["all_potentials_accelerations_sources_finite"]),
        "sphere_radial_RMS": radial_relative_rms
        <= float(gates["sphere_radial_acceleration_relative_RMS_max"]),
        "sphere_radial_median": radial_median_absolute
        <= float(gates["sphere_radial_acceleration_median_absolute_relative_error_max"]),
        "sphere_tangential": tangential_to_radial
        <= float(gates["sphere_tangential_to_radial_RMS_max"]),
        "sphere_angular_scatter": maximum_angular_scatter
        <= float(gates["sphere_radial_angular_scatter_fraction_max"]),
        "curl": maximum_curl <= float(gates["normalized_acceleration_curl_RMS_max"]),
        "quadrature_convergence": quadrature_relative
        <= float(gates["quadrature_24_to_48_acceleration_relative_RMS_max"]),
        "rotation_potential": rotation_potential
        <= float(gates["rotation_covariance_potential_relative_RMS_max"]),
        "rotation_acceleration": rotation_acceleration
        <= float(gates["rotation_covariance_acceleration_relative_RMS_max"]),
        "hybrid_identity": hybrid_identity
        <= float(gates["hybrid_potential_identity_relative_RMS_max"]),
        "correction_boundary": correction_boundary
        <= float(gates["routing_correction_boundary_relative_mismatch_max"]),
        "fraction_lower": fraction >= float(gates["projected_fraction_min"]),
        "fraction_upper": fraction <= float(gates["projected_fraction_max"]),
        "accounting_no_constants": int(equation["new_universal_constants"])
        == int(gates["new_universal_constants"]),
        "accounting_no_fitted_physics": int(equation["fitted_physical_parameters"])
        == int(gates["fitted_physical_parameters"]),
        "no_observational_score": bool(gates["no_observational_score"]),
        "sealed_outcomes_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    radial_table = pd.DataFrame(radial_rows)
    radial_table.to_csv(output / protocol["outputs"]["radial_table"], index=False)
    field_path = output / protocol["outputs"]["synthetic_fields"]
    np.savez_compressed(
        field_path,
        axis=np.arange(cells, dtype=float) * spacing - (cells - 1.0) * spacing / 2.0,
        sphere_density=sphere_density,
        sphere_path_potential=sphere_path.potential,
        sphere_path_acceleration_x=sphere_path.acceleration[0],
        sphere_path_acceleration_y=sphere_path.acceleration[1],
        sphere_path_acceleration_z=sphere_path.acceleration[2],
        ellipsoid_density=ellipsoid_density,
        ellipsoid_path_potential=ellipsoid_path.potential,
        ellipsoid_hybrid_potential=hybrid.potential,
        ellipsoid_routing_correction=hybrid.correction_potential,
        ellipsoid_projected_fraction=fraction,
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(radial_table.radius, radial_table.target_radial_mean, marker="o", label="algebraic target")
    axes[0, 0].plot(radial_table.radius, radial_table.path_radial_mean, marker="s", label="path potential")
    axes[0, 0].set(title="Spherical radial limit", xlabel="radius", ylabel="inward acceleration")
    axes[0, 0].legend()
    axes[0, 1].plot(radial_table.radius, radial_table.median_absolute_relative_error, marker="o", label="median |relative error|")
    axes[0, 1].plot(radial_table.radius, radial_table.angular_scatter_fraction, marker="s", label="angular scatter")
    axes[0, 1].set(title="Spherical discretization", xlabel="radius", ylabel="fraction")
    axes[0, 1].legend()
    middle = cells // 2
    extent = [x.min(), x.max(), y.min(), y.max()]
    image = axes[1, 0].imshow(
        ellipsoid_path.potential[:, :, middle].T,
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axes[1, 0].set(title="Ellipsoid path potential", xlabel="x", ylabel="y")
    figure.colorbar(image, ax=axes[1, 0], shrink=0.75)
    correction_image = axes[1, 1].imshow(
        hybrid.correction_potential[:, :, middle].T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
    )
    axes[1, 1].set(title=f"Routed-local correction, e={fraction:.3f}", xlabel="x", ylabel="y")
    figure.colorbar(correction_image, ax=axes[1, 1], shrink=0.75)
    for axis_plot in axes.ravel():
        axis_plot.grid(alpha=0.15)
    figure.suptitle(f"{stage} radial path potential mathematical audit")
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": f"{stage}-RADIAL-PATH-POTENTIAL-MATH-AUDIT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_math_gates_pass": all_pass,
        "candidate_advanced_to_spent_joint_screen": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/radial_path_potential.py"),
        "routing_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "field_sha256": sha256(field_path),
        "parent_sha256": sha256(failure_path),
        "integrity_gates": integrity,
        "sphere": {
            "radial_acceleration_relative_RMS": radial_relative_rms,
            "radial_acceleration_median_absolute_relative_error": radial_median_absolute,
            "tangential_to_radial_RMS": tangential_to_radial,
            "maximum_radial_angular_scatter_fraction": maximum_angular_scatter,
            "normalized_acceleration_curl_RMS": sphere_curl,
            "quadrature_24_to_48_acceleration_relative_RMS": quadrature_relative,
        },
        "rotation_covariance": {
            "potential_relative_RMS": rotation_potential,
            "acceleration_relative_RMS": rotation_acceleration,
        },
        "hybrid": {
            "projected_routing_fraction": fraction,
            "projected_covariance": covariance.tolist(),
            "projected_eigenvalues": eigenvalues.tolist(),
            "potential_identity_relative_RMS": hybrid_identity,
            "routing_correction_boundary_relative_mismatch": correction_boundary,
            "path_normalized_acceleration_curl_RMS": ellipsoid_curl,
            "rotated_path_normalized_acceleration_curl_RMS": rotated_curl,
            "joint_normalized_acceleration_curl_RMS": hybrid_curl,
        },
        "maximum_normalized_acceleration_curl_RMS": maximum_curl,
        "all_finite": finite,
        "gate_results": gate_results,
        "failed_gates": [name for name, passed in gate_results.items() if not passed],
        "observational_scores_computed": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )
    summary = f"""# {stage} radial path potential mathematical audit

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Spherical radial relative RMS / median error: **{radial_relative_rms:.4g} / {radial_median_absolute:.4g}**.
- Spherical tangential/radial RMS / maximum angular scatter: **{tangential_to_radial:.4g} / {maximum_angular_scatter:.4g}**.
- 24-to-48 quadrature acceleration difference: **{quadrature_relative:.4g}**.
- Rotation potential / acceleration relative RMS: **{rotation_potential:.4g} / {rotation_acceleration:.4g}**.
- Maximum normalized curl / correction-boundary mismatch: **{maximum_curl:.4g} / {correction_boundary:.4g}**.
- Failed gates: **{', '.join(report['failed_gates']) if report['failed_gates'] else 'none'}**.
- Observational scores computed: **no**.
- Advanced to spent joint screen: **{'yes' if all_pass else 'no'}**.
- Sealed P0633/P0640 outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
