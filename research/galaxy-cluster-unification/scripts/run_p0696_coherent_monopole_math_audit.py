#!/usr/bin/env python3
"""Run the frozen no-observation P0696 coherent-monopole audit."""

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

from voidscreen.coherent_monopole import (
    coherent_monopole_potential,
    hybrid_coherent_routing_potential,
)
from voidscreen.field_solvers import (
    boundary_mask,
    cell_coordinates,
    solve_newtonian,
    solve_poisson_dirichlet,
)
from voidscreen.radial_path_potential import normalized_acceleration_curl
from voidscreen.source_routing_qumond import (
    projected_baryonic_spectral_anisotropy,
    solve_source_conserving_baryonic_routing,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0696_coherent_monopole_math_audit.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_vector_rms(left, right, mask) -> float:
    numerator = float(
        np.sqrt(
            np.mean(
                sum(
                    (np.asarray(a)[mask] - np.asarray(b)[mask]) ** 2
                    for a, b in zip(left, right, strict=True)
                )
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
    if protocol.get("status") != "frozen_before_any_P0696_synthetic_coherent_or_hybrid_metric":
        raise RuntimeError("P0696 protocol is not frozen")
    failure_path = ROOT / protocol["failure_parent"]
    failure = read_json(failure_path)
    equation = protocol["equation"]
    expected = protocol["predeclared_integrity_gates"]
    integrity = {
        "P0695B_status": failure.get("status") == expected["P0695B_status"],
        "P0695B_not_advanced": bool(
            failure.get("candidate_advanced_to_spent_joint_screen")
        )
        is bool(expected["P0695B_candidate_advanced_to_spent_joint_screen"]),
        "P0695B_failed_gates_reproduced": list(failure.get("failed_gates", []))
        == list(expected["P0695B_failed_gates"]),
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
        raise RuntimeError(f"P0696 integrity failure before metrics: {integrity}")

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

    print("P0696: spherical coherent-monopole and high-acceleration audit", flush=True)
    sphere_newtonian = solve_newtonian(
        sphere_density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    sphere_coherent = coherent_monopole_potential(
        sphere_density,
        sphere_newtonian.potential,
        sphere_newtonian.acceleration,
        spacing,
        a0=a0,
    )
    sphere_high_acceleration = coherent_monopole_potential(
        sphere_density,
        sphere_newtonian.potential,
        sphere_newtonian.acceleration,
        spacing,
        a0=float(numerics["high_acceleration_a0"]),
    )
    center = sphere_coherent.center_of_mass
    displacement = (x - center[0], y - center[1], z - center[2])
    centered_radius = np.sqrt(sum(component * component for component in displacement))
    safe_radius = np.maximum(centered_radius, np.finfo(float).tiny)
    radial_unit = tuple(component / safe_radius for component in displacement)
    shell_index = np.rint(centered_radius / spacing).astype(int)
    coherent_radial = -sum(
        component * direction
        for component, direction in zip(
            sphere_coherent.acceleration,
            radial_unit,
            strict=True,
        )
    )
    correction_radial = -sum(
        component * direction
        for component, direction in zip(
            sphere_coherent.correction_acceleration,
            radial_unit,
            strict=True,
        )
    )
    correction_magnitude_squared = sum(
        component * component for component in sphere_coherent.correction_acceleration
    )
    correction_tangential_squared = np.maximum(
        correction_magnitude_squared - correction_radial**2,
        0.0,
    )
    lower_cells, upper_cells = (
        float(value) for value in numerics["comparison_radius_grid_cells"]
    )
    comparison = (
        (centered_radius >= lower_cells * spacing)
        & (centered_radius <= upper_cells * spacing)
    )
    radial_rows = []
    for shell in range(int(np.ceil(lower_cells)), int(np.floor(upper_cells)) + 1):
        shell_mask = comparison & shell_index.__eq__(shell)
        if np.count_nonzero(shell_mask) < 12:
            continue
        coherent_values = coherent_radial[shell_mask]
        correction_values = correction_radial[shell_mask]
        target = float(sphere_coherent.coherent_completed_acceleration[shell])
        radial_rows.append(
            {
                "radius_grid_cells": shell,
                "radius": shell * spacing,
                "cells": int(np.count_nonzero(shell_mask)),
                "coherent_radial_mean": float(np.mean(coherent_values)),
                "target_shell_acceleration": target,
                "mean_relative_error": float(np.mean(coherent_values) / target - 1.0),
                "correction_radial_mean": float(np.mean(correction_values)),
                "correction_angular_scatter_fraction": float(
                    np.std(correction_values)
                    / max(abs(float(np.mean(correction_values))), np.finfo(float).tiny)
                ),
            }
        )
    radial_table = pd.DataFrame(radial_rows)
    shell_relative_errors = (
        radial_table.coherent_radial_mean / radial_table.target_shell_acceleration - 1.0
    ).to_numpy()
    radial_relative_rms = float(np.sqrt(np.mean(shell_relative_errors**2)))
    radial_median_absolute = float(np.median(np.abs(shell_relative_errors)))
    tangential_to_radial = float(
        np.sqrt(np.mean(correction_tangential_squared[comparison]))
        / max(
            float(np.sqrt(np.mean(correction_radial[comparison] ** 2))),
            np.finfo(float).tiny,
        )
    )
    maximum_angular_scatter = float(
        radial_table.correction_angular_scatter_fraction.max()
    )
    high_acceleration_ratio = relative_vector_rms(
        sphere_high_acceleration.correction_acceleration,
        sphere_newtonian.acceleration,
        comparison,
    )
    sphere_identity = relative_grid_rms(
        sphere_coherent.potential,
        sphere_newtonian.potential + sphere_coherent.correction_potential,
        comparison,
    )

    print("P0696: ellipsoid rotation covariance and hybrid identity", flush=True)
    ellipsoid_newtonian = solve_newtonian(
        ellipsoid_density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    ellipsoid_coherent = coherent_monopole_potential(
        ellipsoid_density,
        ellipsoid_newtonian.potential,
        ellipsoid_newtonian.acceleration,
        spacing,
        a0=a0,
    )
    rotated_density = np.swapaxes(ellipsoid_density, 0, 1)
    rotated_newtonian = solve_newtonian(
        rotated_density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    rotated_coherent = coherent_monopole_potential(
        rotated_density,
        rotated_newtonian.potential,
        rotated_newtonian.acceleration,
        spacing,
        a0=a0,
    )
    interior = np.zeros((cells,) * 3, dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True
    expected_rotated_potential = np.swapaxes(ellipsoid_coherent.potential, 0, 1)
    expected_rotated_acceleration = (
        np.swapaxes(ellipsoid_coherent.acceleration[1], 0, 1),
        np.swapaxes(ellipsoid_coherent.acceleration[0], 0, 1),
        np.swapaxes(ellipsoid_coherent.acceleration[2], 0, 1),
    )
    rotation_potential = relative_grid_rms(
        rotated_coherent.potential,
        expected_rotated_potential,
        interior,
    )
    rotation_acceleration = relative_vector_rms(
        rotated_coherent.acceleration,
        expected_rotated_acceleration,
        interior,
    )
    ellipsoid_identity = relative_grid_rms(
        ellipsoid_coherent.potential,
        ellipsoid_newtonian.potential + ellipsoid_coherent.correction_potential,
        interior,
    )
    maximum_coherent_identity = max(sphere_identity, ellipsoid_identity)

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
    hybrid = hybrid_coherent_routing_potential(
        ellipsoid_coherent,
        local_potential,
        routing.field.potential,
        spacing,
        fraction,
    )
    expected_hybrid = ellipsoid_coherent.potential + fraction * (
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
    curl_scores = {
        "sphere_coherent": normalized_acceleration_curl(
            sphere_coherent.acceleration,
            spacing,
        ),
        "sphere_correction": normalized_acceleration_curl(
            sphere_coherent.correction_acceleration,
            spacing,
        ),
        "ellipsoid_coherent": normalized_acceleration_curl(
            ellipsoid_coherent.acceleration,
            spacing,
        ),
        "rotated_coherent": normalized_acceleration_curl(
            rotated_coherent.acceleration,
            spacing,
        ),
        "hybrid": normalized_acceleration_curl(hybrid.acceleration, spacing),
    }
    maximum_curl = max(curl_scores.values())
    finite = bool(
        all(
            np.all(np.isfinite(array))
            for array in (
                sphere_coherent.potential,
                *sphere_coherent.acceleration,
                sphere_coherent.equation_source,
                sphere_coherent.correction_potential,
                *sphere_coherent.correction_acceleration,
                sphere_high_acceleration.potential,
                *sphere_high_acceleration.acceleration,
                ellipsoid_coherent.potential,
                *ellipsoid_coherent.acceleration,
                rotated_coherent.potential,
                *rotated_coherent.acceleration,
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
        <= float(gates["sphere_shell_mean_radial_acceleration_relative_RMS_max"]),
        "sphere_radial_median": radial_median_absolute
        <= float(
            gates[
                "sphere_shell_mean_radial_acceleration_median_absolute_relative_error_max"
            ]
        ),
        "correction_tangential": tangential_to_radial
        <= float(gates["added_monopole_correction_tangential_to_radial_RMS_max"]),
        "correction_angular_scatter": maximum_angular_scatter
        <= float(
            gates["added_monopole_correction_maximum_angular_scatter_fraction_max"]
        ),
        "curl": maximum_curl <= float(gates["normalized_acceleration_curl_RMS_max"]),
        "rotation_potential": rotation_potential
        <= float(gates["rotation_covariance_potential_relative_RMS_max"]),
        "rotation_acceleration": rotation_acceleration
        <= float(gates["rotation_covariance_acceleration_relative_RMS_max"]),
        "coherent_identity": maximum_coherent_identity
        <= float(gates["coherent_potential_identity_relative_RMS_max"]),
        "high_acceleration_limit": high_acceleration_ratio
        <= float(gates["high_acceleration_correction_to_newtonian_RMS_max"]),
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
    radial_table.to_csv(output / protocol["outputs"]["radial_table"], index=False)
    field_path = output / protocol["outputs"]["synthetic_fields"]
    np.savez_compressed(
        field_path,
        axis=np.arange(cells, dtype=float) * spacing - (cells - 1.0) * spacing / 2.0,
        sphere_density=sphere_density,
        sphere_newtonian_potential=sphere_newtonian.potential,
        sphere_coherent_potential=sphere_coherent.potential,
        sphere_correction_potential=sphere_coherent.correction_potential,
        sphere_coherent_acceleration_x=sphere_coherent.acceleration[0],
        sphere_coherent_acceleration_y=sphere_coherent.acceleration[1],
        sphere_coherent_acceleration_z=sphere_coherent.acceleration[2],
        ellipsoid_density=ellipsoid_density,
        ellipsoid_coherent_potential=ellipsoid_coherent.potential,
        ellipsoid_hybrid_potential=hybrid.potential,
        ellipsoid_routing_correction=hybrid.routing_correction_potential,
        ellipsoid_projected_fraction=fraction,
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(
        radial_table.radius,
        radial_table.target_shell_acceleration,
        marker="o",
        label="shell completion target",
    )
    axes[0, 0].plot(
        radial_table.radius,
        radial_table.coherent_radial_mean,
        marker="s",
        label="field shell mean",
    )
    axes[0, 0].set(
        title="Spherical coherent limit",
        xlabel="radius",
        ylabel="inward acceleration",
    )
    axes[0, 0].legend()
    axes[0, 1].plot(
        radial_table.radius,
        np.abs(radial_table.mean_relative_error),
        marker="o",
        label="|shell relative error|",
    )
    axes[0, 1].plot(
        radial_table.radius,
        radial_table.correction_angular_scatter_fraction,
        marker="s",
        label="correction angular scatter",
    )
    axes[0, 1].set(title="Native-shell discretization", xlabel="radius", ylabel="fraction")
    axes[0, 1].legend()
    middle = cells // 2
    extent = [x.min(), x.max(), y.min(), y.max()]
    coherent_image = axes[1, 0].imshow(
        ellipsoid_coherent.correction_potential[:, :, middle].T,
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axes[1, 0].set(title="Ellipsoid coherent correction", xlabel="x", ylabel="y")
    figure.colorbar(coherent_image, ax=axes[1, 0], shrink=0.75)
    routing_image = axes[1, 1].imshow(
        hybrid.routing_correction_potential[:, :, middle].T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
    )
    axes[1, 1].set(
        title=f"Routed-local correction, e={fraction:.3f}",
        xlabel="x",
        ylabel="y",
    )
    figure.colorbar(routing_image, ax=axes[1, 1], shrink=0.75)
    for axis_plot in axes.ravel():
        axis_plot.grid(alpha=0.15)
    figure.suptitle("P0696 coherent-monopole mathematical audit")
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0696-COHERENT-MONOPOLE-MATH-AUDIT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_math_gates_pass": all_pass,
        "candidate_advanced_to_spent_joint_screen": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/coherent_monopole.py"),
        "routing_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "field_sha256": sha256(field_path),
        "parent_sha256": sha256(failure_path),
        "integrity_gates": integrity,
        "sphere": {
            "shell_mean_radial_acceleration_relative_RMS": radial_relative_rms,
            "shell_mean_radial_acceleration_median_absolute_relative_error": radial_median_absolute,
            "correction_tangential_to_radial_RMS": tangential_to_radial,
            "correction_maximum_angular_scatter_fraction": maximum_angular_scatter,
            "high_acceleration_correction_to_newtonian_RMS": high_acceleration_ratio,
        },
        "rotation_covariance": {
            "potential_relative_RMS": rotation_potential,
            "acceleration_relative_RMS": rotation_acceleration,
        },
        "hybrid": {
            "projected_routing_fraction": fraction,
            "projected_covariance": covariance.tolist(),
            "projected_eigenvalues": eigenvalues.tolist(),
            "coherent_potential_identity_relative_RMS": maximum_coherent_identity,
            "potential_identity_relative_RMS": hybrid_identity,
            "routing_correction_boundary_relative_mismatch": correction_boundary,
        },
        "normalized_acceleration_curl_RMS": curl_scores,
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
    summary = f"""# P0696 coherent-monopole mathematical audit

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Spherical shell-mean radial relative RMS / median error: **{radial_relative_rms:.4g} / {radial_median_absolute:.4g}**.
- Correction tangential/radial RMS / maximum angular scatter: **{tangential_to_radial:.4g} / {maximum_angular_scatter:.4g}**.
- High-acceleration correction/Newtonian RMS: **{high_acceleration_ratio:.4g}**.
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
