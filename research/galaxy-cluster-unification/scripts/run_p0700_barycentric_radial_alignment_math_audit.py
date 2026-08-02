#!/usr/bin/env python3
"""Run the frozen no-observation P0700 barycentric-alignment audit."""

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
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import json_safe
from run_p0660_exact_tensor_activation_audit import sha256

from voidscreen.barycentric_radial_alignment import (
    barycentric_radial_alignment,
    vector_radial_alignment,
)
from voidscreen.coherent_monopole import coherent_monopole_potential
from voidscreen.field_solvers import (
    boundary_mask,
    cell_coordinates,
    solve_newtonian,
    solve_poisson_dirichlet,
)
from voidscreen.local_vector_coherence import (
    base_boundary_relative_mismatch,
    coherence_gated_source_potential,
    hybrid_coherence_routing_potential,
)
from voidscreen.radial_path_potential import normalized_acceleration_curl
from voidscreen.source_routing_qumond import (
    projected_baryonic_spectral_anisotropy,
    solve_source_conserving_baryonic_routing,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0700_barycentric_radial_alignment_math_audit.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_grid_rms(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    numerator = float(np.sqrt(np.mean((left[mask] - right[mask]) ** 2)))
    denominator = float(np.sqrt(np.mean(right[mask] ** 2)))
    return numerator / max(denominator, np.finfo(float).tiny)


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


def normalized_density(values: np.ndarray, spacing: float) -> np.ndarray:
    density = np.asarray(values, dtype=float)
    return density / float(np.sum(density) * spacing**3)


def solve_synthetic(
    density: np.ndarray,
    spacing: float,
    *,
    gravitational_constant: float,
    a0: float,
    light_speed: float,
    numerics: dict,
):
    routing = solve_source_conserving_baryonic_routing(
        density,
        spacing,
        gravitational_constant=gravitational_constant,
        a0=a0,
        transition_depth=float(numerics["transition_depth"]),
        transition_power=float(numerics["transition_power"]),
        extra_spatial_channels=float(numerics["extra_spatial_channels"]),
        path_power=float(numerics["path_power"]),
        light_speed=light_speed,
    )
    local_potential = solve_poisson_dirichlet(
        routing.local_generator_source,
        spacing,
        routing.boundary_potential,
    )
    coherent = coherent_monopole_potential(
        density,
        routing.newtonian.potential,
        routing.newtonian.acceleration,
        spacing,
        a0=a0,
    )
    controller = barycentric_radial_alignment(
        density,
        routing.newtonian.acceleration,
        spacing,
    )
    base = coherence_gated_source_potential(
        coherent,
        routing.local_generator_source,
        controller.alignment,
        spacing,
    )
    surface = np.sum(density, axis=2) * spacing
    fraction, covariance, eigenvalues = projected_baryonic_spectral_anisotropy(
        surface,
        spacing,
    )
    hybrid = hybrid_coherence_routing_potential(
        base,
        local_potential,
        routing.field.potential,
        spacing,
        fraction,
    )
    return {
        "routing": routing,
        "local_potential": local_potential,
        "coherent": coherent,
        "controller": controller,
        "base": base,
        "hybrid": hybrid,
        "fraction": fraction,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != (
        "frozen_before_any_P0700_synthetic_alignment_or_gated_field_metric"
    ):
        raise RuntimeError("P0700 protocol is not frozen")
    failure_path = ROOT / protocol["failure_parent"]
    failure = read_json(failure_path)
    equation = protocol["equation"]
    expected = protocol["predeclared_integrity_gates"]
    integrity = {
        "P0699_status": failure.get("status") == expected["P0699_status"],
        "P0699_not_advanced": bool(
            failure.get("candidate_advanced_to_robustness_and_solar")
        )
        is bool(expected["P0699_candidate_advanced_to_robustness_and_solar"]),
        "P0699_failed_gates_reproduced": list(failure.get("failed_gates", []))
        == list(expected["P0699_failed_gates"]),
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
        raise RuntimeError(f"P0700 integrity failure before metrics: {integrity}")

    numerics = protocol["numerics"]
    cells = int(numerics["primary_grid_cells"])
    spacing = float(numerics["primary_spacing"])
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    radius = np.sqrt(x * x + y * y + z * z)
    single_scale = float(numerics["single_gaussian_scale"])
    single_density = normalized_density(
        np.exp(-0.5 * radius**2 / single_scale**2),
        spacing,
    )
    dual_scale = float(numerics["two_center_gaussian_scale"])
    left_offset, right_offset = (
        float(value) for value in numerics["two_center_offsets"]
    )
    dual_density = normalized_density(
        np.exp(-0.5 * ((x - left_offset) ** 2 + y**2 + z**2) / dual_scale**2)
        + np.exp(-0.5 * ((x - right_offset) ** 2 + y**2 + z**2) / dual_scale**2),
        spacing,
    )
    gravitational_constant = float(numerics["gravitational_constant"])
    a0 = float(numerics["a0"])
    light_speed = float(numerics["light_speed"])

    print("P0700: single-center and two-center alignment fields", flush=True)
    single = solve_synthetic(
        single_density,
        spacing,
        gravitational_constant=gravitational_constant,
        a0=a0,
        light_speed=light_speed,
        numerics=numerics,
    )
    dual = solve_synthetic(
        dual_density,
        spacing,
        gravitational_constant=gravitational_constant,
        a0=a0,
        light_speed=light_speed,
        numerics=numerics,
    )
    single_lower, single_upper = (
        float(value) for value in numerics["single_center_outer_annulus_grid_cells"]
    )
    single_annulus = (
        (radius >= single_lower * spacing) & (radius <= single_upper * spacing)
    )
    single_outer_median = float(
        np.median(single["controller"].alignment[single_annulus])
    )
    structure_lower, structure_upper = (
        float(value) for value in numerics["two_center_structure_radius_grid_cells"]
    )
    structure_mask = (
        (radius >= structure_lower * spacing) & (radius <= structure_upper * spacing)
    )
    dual_structure_fraction_below = float(
        np.mean(dual["controller"].alignment[structure_mask] < 0.9)
    )
    middle = cells // 2
    center_alignment = float(
        single["controller"].alignment[middle, middle, middle]
    )

    safe_radius = np.maximum(radius, np.finfo(float).tiny)
    radial_unit = (x / safe_radius, y / safe_radius, z / safe_radius)
    inward_control, _, _ = vector_radial_alignment(
        (x, y, z),
        tuple(-component for component in radial_unit),
    )
    outward_control, _, _ = vector_radial_alignment((x, y, z), radial_unit)
    tangential_control, _, _ = vector_radial_alignment(
        (x, y, z),
        (-radial_unit[1], radial_unit[0], np.zeros_like(x)),
    )
    control_active = radius > 0.0
    inward_control_median = float(np.median(inward_control[control_active]))
    outward_control_maximum = float(np.max(outward_control))
    tangential_control_maximum = float(np.max(tangential_control))

    radial_rows = []
    shell_index = np.rint(radius / spacing).astype(int)
    for shell in range(1, int(np.max(shell_index)) + 1):
        shell_mask = shell_index == shell
        if np.count_nonzero(shell_mask) < 12:
            continue
        radial_rows.append(
            {
                "radius_grid_cells": shell,
                "radius": shell * spacing,
                "cells": int(np.count_nonzero(shell_mask)),
                "single_alignment_mean": float(
                    np.mean(single["controller"].alignment[shell_mask])
                ),
                "single_alignment_median": float(
                    np.median(single["controller"].alignment[shell_mask])
                ),
                "dual_alignment_mean": float(
                    np.mean(dual["controller"].alignment[shell_mask])
                ),
                "dual_alignment_median": float(
                    np.median(dual["controller"].alignment[shell_mask])
                ),
            }
        )
    radial_table = pd.DataFrame(radial_rows)

    print("P0700: rotations, translation, resolution, and endpoint limits", flush=True)
    rotated_density = np.swapaxes(dual_density, 0, 1)
    rotated = solve_synthetic(
        rotated_density,
        spacing,
        gravitational_constant=gravitational_constant,
        a0=a0,
        light_speed=light_speed,
        numerics=numerics,
    )
    interior = np.zeros((cells,) * 3, dtype=bool)
    interior[3:-3, 3:-3, 3:-3] = True
    expected_rotated_alignment = np.swapaxes(
        dual["controller"].alignment,
        0,
        1,
    )
    rotation_alignment = relative_grid_rms(
        rotated["controller"].alignment,
        expected_rotated_alignment,
        interior,
    )
    expected_rotated_potential = np.swapaxes(dual["hybrid"].potential, 0, 1)
    expected_rotated_acceleration = (
        np.swapaxes(dual["hybrid"].acceleration[1], 0, 1),
        np.swapaxes(dual["hybrid"].acceleration[0], 0, 1),
        np.swapaxes(dual["hybrid"].acceleration[2], 0, 1),
    )
    rotation_potential = relative_grid_rms(
        rotated["hybrid"].potential,
        expected_rotated_potential,
        interior,
    )
    rotation_acceleration = relative_vector_rms(
        rotated["hybrid"].acceleration,
        expected_rotated_acceleration,
        interior,
    )

    translation_cells = int(numerics["translation_cells"])
    translated_density = normalized_density(
        np.exp(
            -0.5
            * ((x - translation_cells * spacing) ** 2 + y**2 + z**2)
            / single_scale**2
        ),
        spacing,
    )
    translated_newtonian = solve_newtonian(
        translated_density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    translated_alignment = barycentric_radial_alignment(
        translated_density,
        translated_newtonian.acceleration,
        spacing,
    )
    expected_translated = np.roll(
        single["controller"].alignment,
        translation_cells,
        axis=0,
    )
    translation_mask = np.zeros((cells,) * 3, dtype=bool)
    translation_mask[5:-5, 5:-5, 5:-5] = True
    translation_rms = float(
        np.sqrt(
            np.mean(
                (
                    translated_alignment.alignment[translation_mask]
                    - expected_translated[translation_mask]
                )
                ** 2
            )
        )
    )

    resolution_cells = int(numerics["resolution_grid_cells"])
    resolution_spacing = float(numerics["resolution_spacing"])
    xh, yh, zh = cell_coordinates((resolution_cells,) * 3, resolution_spacing)
    radius_high = np.sqrt(xh * xh + yh * yh + zh * zh)
    high_density = normalized_density(
        np.exp(-0.5 * radius_high**2 / single_scale**2),
        resolution_spacing,
    )
    resolution_newtonian = solve_newtonian(
        high_density,
        resolution_spacing,
        gravitational_constant=gravitational_constant,
    )
    resolution_alignment = barycentric_radial_alignment(
        high_density,
        resolution_newtonian.acceleration,
        resolution_spacing,
    )
    high_axis = (
        np.arange(resolution_cells, dtype=float) - (resolution_cells - 1.0) / 2.0
    ) * resolution_spacing
    interpolator = RegularGridInterpolator(
        (high_axis, high_axis, high_axis),
        resolution_alignment.alignment,
        bounds_error=True,
    )
    high_on_primary = interpolator(np.column_stack((x.ravel(), y.ravel(), z.ravel()))).reshape(
        x.shape
    )
    resolution_lower, resolution_upper = (
        float(value) for value in numerics["resolution_comparison_radius"]
    )
    resolution_mask = (radius >= resolution_lower) & (radius <= resolution_upper)
    resolution_rms = float(
        np.sqrt(
            np.mean(
                (
                    single["controller"].alignment[resolution_mask]
                    - high_on_primary[resolution_mask]
                )
                ** 2
            )
        )
    )

    endpoint = coherence_gated_source_potential(
        single["coherent"],
        single["routing"].local_generator_source,
        np.ones_like(single_density),
        spacing,
    )
    endpoint_identity = relative_grid_rms(
        endpoint.potential,
        single["coherent"].potential,
        interior,
    )
    high_newtonian = solve_newtonian(
        single_density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    high_coherent = coherent_monopole_potential(
        single_density,
        high_newtonian.potential,
        high_newtonian.acceleration,
        spacing,
        a0=float(numerics["high_acceleration_a0"]),
    )
    high_base = coherence_gated_source_potential(
        high_coherent,
        high_newtonian.equation_source,
        single["controller"].alignment,
        spacing,
    )
    high_acceleration_limit = relative_vector_rms(
        high_base.acceleration,
        high_newtonian.acceleration,
        single_annulus,
    )

    solutions = [single, dual, rotated]
    all_alignment = np.concatenate(
        [item["controller"].alignment.ravel() for item in solutions]
        + [translated_alignment.alignment.ravel(), resolution_alignment.alignment.ravel()]
    )
    alignment_min = float(np.min(all_alignment))
    alignment_max = float(np.max(all_alignment))
    source_identity_scores = []
    hybrid_identity_scores = []
    base_boundary_scores = []
    routing_boundary_scores = []
    residual_scores = []
    curl_scores = []
    finite_arrays = []
    for item in solutions:
        expected_source = (
            item["controller"].alignment * item["coherent"].equation_source
            + (1.0 - item["controller"].alignment)
            * item["routing"].local_generator_source
        )
        source_identity_scores.append(
            relative_grid_rms(item["base"].equation_source, expected_source, interior)
        )
        expected_hybrid = item["base"].potential + item["fraction"] * (
            item["routing"].field.potential - item["local_potential"]
        )
        hybrid_identity_scores.append(
            relative_grid_rms(item["hybrid"].potential, expected_hybrid, interior)
        )
        base_boundary_scores.append(
            base_boundary_relative_mismatch(item["base"], item["coherent"])
        )
        edge = boundary_mask(item["base"].potential.shape)
        routing_scale = max(
            float(np.max(np.abs(item["routing"].boundary_potential[edge]))),
            np.finfo(float).tiny,
        )
        routing_boundary_scores.append(
            float(
                np.max(
                    np.abs(
                        (item["routing"].field.potential - item["local_potential"])[edge]
                    )
                )
                / routing_scale
            )
        )
        residual_scores.append(float(item["base"].normalized_residual_rms))
        curl_scores.extend(
            [
                normalized_acceleration_curl(item["base"].acceleration, spacing),
                normalized_acceleration_curl(item["hybrid"].acceleration, spacing),
            ]
        )
        finite_arrays.extend(
            [
                item["controller"].alignment,
                item["controller"].inward_radial_acceleration,
                item["controller"].acceleration_magnitude,
                item["base"].potential,
                *item["base"].acceleration,
                item["base"].equation_source,
                item["hybrid"].potential,
                *item["hybrid"].acceleration,
            ]
        )
    high_expected_source = (
        single["controller"].alignment * high_coherent.equation_source
        + (1.0 - single["controller"].alignment)
        * high_newtonian.equation_source
    )
    source_identity_scores.append(
        relative_grid_rms(high_base.equation_source, high_expected_source, interior)
    )
    base_boundary_scores.append(
        base_boundary_relative_mismatch(high_base, high_coherent)
    )
    residual_scores.append(float(high_base.normalized_residual_rms))
    curl_scores.append(normalized_acceleration_curl(high_base.acceleration, spacing))
    finite_arrays.extend(
        [
            high_coherent.potential,
            *high_coherent.acceleration,
            high_base.potential,
            *high_base.acceleration,
            high_base.equation_source,
        ]
    )
    maximum_source_identity = max(source_identity_scores)
    maximum_hybrid_identity = max(hybrid_identity_scores)
    maximum_base_boundary = max(base_boundary_scores)
    maximum_routing_boundary = max(routing_boundary_scores)
    maximum_residual = max(residual_scores)
    maximum_curl = max(curl_scores)
    all_finite = bool(all(np.all(np.isfinite(array)) for array in finite_arrays))
    fractions = [float(item["fraction"]) for item in solutions]

    gates = protocol["predeclared_math_gates"]
    gate_results = {
        **integrity,
        "finite": all_finite
        is bool(gates["all_alignments_sources_potentials_accelerations_finite"]),
        "alignment_lower": alignment_min >= float(gates["alignment_min"]),
        "alignment_upper": alignment_max <= float(gates["alignment_max"]),
        "explicit_inward": inward_control_median
        >= float(gates["explicit_inward_radial_alignment_median_min"]),
        "explicit_outward": outward_control_maximum
        <= float(gates["explicit_outward_radial_alignment_max"]),
        "explicit_tangential": tangential_control_maximum
        <= float(gates["explicit_tangential_alignment_max"]),
        "center_rule": center_alignment <= float(gates["center_alignment_max"]),
        "single_center_outer": single_outer_median
        >= float(gates["single_center_outer_annulus_median_alignment_min"]),
        "two_center_structure": dual_structure_fraction_below
        >= float(gates["two_center_structure_fraction_below_0_9_min"]),
        "rotation_alignment": rotation_alignment
        <= float(gates["rotation_covariance_alignment_relative_RMS_max"]),
        "rotation_potential": rotation_potential
        <= float(gates["rotation_covariance_potential_relative_RMS_max"]),
        "rotation_acceleration": rotation_acceleration
        <= float(gates["rotation_covariance_acceleration_relative_RMS_max"]),
        "translation_alignment": translation_rms
        <= float(gates["translation_covariance_alignment_RMS_max"]),
        "resolution_alignment": resolution_rms
        <= float(gates["resolution_alignment_RMS_max"]),
        "source_identity": maximum_source_identity
        <= float(gates["gated_source_identity_relative_RMS_max"]),
        "coherent_endpoint": endpoint_identity
        <= float(gates["coherent_endpoint_potential_relative_RMS_max"]),
        "field_residual": maximum_residual
        <= float(gates["field_normalized_residual_RMS_max"]),
        "curl": maximum_curl <= float(gates["normalized_acceleration_curl_RMS_max"]),
        "base_boundary": maximum_base_boundary
        <= float(gates["base_coherent_boundary_relative_mismatch_max"]),
        "hybrid_identity": maximum_hybrid_identity
        <= float(gates["hybrid_potential_identity_relative_RMS_max"]),
        "routing_boundary": maximum_routing_boundary
        <= float(gates["routing_correction_boundary_relative_mismatch_max"]),
        "high_acceleration_limit": high_acceleration_limit
        <= float(
            gates["high_acceleration_base_to_newtonian_acceleration_relative_RMS_max"]
        ),
        "fraction_lower": min(fractions) >= float(gates["projected_fraction_min"]),
        "fraction_upper": max(fractions) <= float(gates["projected_fraction_max"]),
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
        single_density=single_density,
        single_alignment=single["controller"].alignment,
        single_base_potential=single["base"].potential,
        single_hybrid_potential=single["hybrid"].potential,
        dual_density=dual_density,
        dual_alignment=dual["controller"].alignment,
        dual_base_potential=dual["base"].potential,
        dual_hybrid_potential=dual["hybrid"].potential,
        high_resolution_single_alignment=resolution_alignment.alignment,
        high_resolution_axis=high_axis,
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    extent = [x.min(), x.max(), y.min(), y.max()]
    image_single = axes[0, 0].imshow(
        single["controller"].alignment[:, :, middle].T,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[0, 0].set(title="Single-center radial alignment", xlabel="x", ylabel="y")
    figure.colorbar(image_single, ax=axes[0, 0], shrink=0.75)
    image_dual = axes[0, 1].imshow(
        dual["controller"].alignment[:, :, middle].T,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[0, 1].set(title="Two-center radial alignment", xlabel="x", ylabel="y")
    figure.colorbar(image_dual, ax=axes[0, 1], shrink=0.75)
    axes[1, 0].plot(
        radial_table.radius,
        radial_table.single_alignment_median,
        marker="o",
        label="single center",
    )
    axes[1, 0].plot(
        radial_table.radius,
        radial_table.dual_alignment_median,
        marker="s",
        label="two centers",
    )
    axes[1, 0].set(title="Shell-median alignment", xlabel="radius", ylabel="alignment")
    axes[1, 0].legend()
    potential_image = axes[1, 1].imshow(
        dual["hybrid"].potential[:, :, middle].T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
    )
    axes[1, 1].set(title="Two-center hybrid potential", xlabel="x", ylabel="y")
    figure.colorbar(potential_image, ax=axes[1, 1], shrink=0.75)
    for axis_plot in axes.ravel():
        axis_plot.grid(alpha=0.15)
    figure.suptitle("P0700 barycentric radial-alignment mathematical audit")
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0700-BARYCENTRIC-RADIAL-ALIGNMENT-MATH-AUDIT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_math_gates_pass": all_pass,
        "candidate_advanced_to_spent_joint_screen": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(
            ROOT / "src/voidscreen/barycentric_radial_alignment.py"
        ),
        "gating_source_sha256": sha256(ROOT / "src/voidscreen/local_vector_coherence.py"),
        "coherent_source_sha256": sha256(ROOT / "src/voidscreen/coherent_monopole.py"),
        "routing_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "field_sha256": sha256(field_path),
        "parent_sha256": sha256(failure_path),
        "integrity_gates": integrity,
        "alignment": {
            "minimum": alignment_min,
            "maximum": alignment_max,
            "explicit_inward_radial_median": inward_control_median,
            "explicit_outward_radial_maximum": outward_control_maximum,
            "explicit_tangential_maximum": tangential_control_maximum,
            "center_value": center_alignment,
            "single_center_outer_annulus_median": single_outer_median,
            "two_center_structure_fraction_below_0_9": dual_structure_fraction_below,
        },
        "covariance": {
            "rotation_alignment_relative_RMS": rotation_alignment,
            "rotation_potential_relative_RMS": rotation_potential,
            "rotation_acceleration_relative_RMS": rotation_acceleration,
            "translation_alignment_RMS": translation_rms,
            "resolution_alignment_RMS": resolution_rms,
        },
        "field": {
            "maximum_gated_source_identity_relative_RMS": maximum_source_identity,
            "coherent_endpoint_potential_relative_RMS": endpoint_identity,
            "maximum_normalized_residual_RMS": maximum_residual,
            "maximum_normalized_acceleration_curl_RMS": maximum_curl,
            "maximum_base_coherent_boundary_relative_mismatch": maximum_base_boundary,
            "maximum_hybrid_potential_identity_relative_RMS": maximum_hybrid_identity,
            "maximum_routing_correction_boundary_relative_mismatch": maximum_routing_boundary,
            "high_acceleration_base_to_newtonian_acceleration_relative_RMS": high_acceleration_limit,
            "projected_fractions": fractions,
        },
        "all_finite": all_finite,
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
    summary = f"""# P0700 barycentric radial-alignment mathematical audit

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Alignment range / explicit inward-outward-tangential controls: **[{alignment_min:.4g}, {alignment_max:.4g}] / {inward_control_median:.4g} / {outward_control_maximum:.4g} / {tangential_control_maximum:.4g}**.
- Single outer median / two-center fraction below 0.9: **{single_outer_median:.4g} / {dual_structure_fraction_below:.4g}**.
- Rotation alignment / potential / acceleration RMS: **{rotation_alignment:.4g} / {rotation_potential:.4g} / {rotation_acceleration:.4g}**.
- Translation / resolution alignment RMS: **{translation_rms:.4g} / {resolution_rms:.4g}**.
- Endpoint / residual / curl / strong-field error: **{endpoint_identity:.4g} / {maximum_residual:.4g} / {maximum_curl:.4g} / {high_acceleration_limit:.4g}**.
- Failed gates: **{', '.join(report['failed_gates']) if report['failed_gates'] else 'none'}**.
- Observational scores computed: **no**.
- Advanced to spent joint screen: **{'yes' if all_pass else 'no'}**.
- Sealed P0633/P0640 outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
