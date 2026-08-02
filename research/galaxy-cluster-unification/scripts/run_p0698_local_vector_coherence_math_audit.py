#!/usr/bin/env python3
"""Run the frozen no-observation P0698 local-vector-coherence audit."""

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

from voidscreen.coherent_monopole import coherent_monopole_potential
from voidscreen.field_solvers import (
    boundary_mask,
    cell_coordinates,
    solve_newtonian,
    solve_poisson_dirichlet,
)
from voidscreen.local_vector_coherence import (
    baryonic_vector_coherence,
    base_boundary_relative_mismatch,
    coherence_gated_source_potential,
    hybrid_coherence_routing_potential,
)
from voidscreen.radial_path_potential import normalized_acceleration_curl
from voidscreen.source_routing_qumond import (
    projected_baryonic_spectral_anisotropy,
    solve_source_conserving_baryonic_routing,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0698_local_vector_coherence_math_audit.json"


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
    vector_coherence = baryonic_vector_coherence(
        density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    base = coherence_gated_source_potential(
        coherent,
        routing.local_generator_source,
        vector_coherence.coherence,
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
        "vector_coherence": vector_coherence,
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
        "frozen_before_any_P0698_synthetic_coherence_or_gated_field_metric"
    ):
        raise RuntimeError("P0698 protocol is not frozen")
    failure_path = ROOT / protocol["failure_parent"]
    failure = read_json(failure_path)
    equation = protocol["equation"]
    expected = protocol["predeclared_integrity_gates"]
    integrity = {
        "P0697_status": failure.get("status") == expected["P0697_status"],
        "P0697_not_advanced": bool(
            failure.get("candidate_advanced_to_robustness_and_solar")
        )
        is bool(expected["P0697_candidate_advanced_to_robustness_and_solar"]),
        "P0697_failed_gates_reproduced": list(failure.get("failed_gates", []))
        == list(expected["P0697_failed_gates"]),
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
        raise RuntimeError(f"P0698 integrity failure before metrics: {integrity}")

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

    print("P0698: single-center and two-center coherence fields", flush=True)
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
    far_lower, far_upper = (
        float(value) for value in numerics["two_center_far_annulus_grid_cells"]
    )
    far_annulus = (radius >= far_lower * spacing) & (radius <= far_upper * spacing)
    single_outer_median = float(
        np.median(single["vector_coherence"].coherence[single_annulus])
    )
    dual_far_median = float(
        np.median(dual["vector_coherence"].coherence[far_annulus])
    )
    middle = cells // 2
    dual_midpoint = float(dual["vector_coherence"].coherence[middle, middle, middle])
    far_midpoint_contrast = dual_far_median - dual_midpoint

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
                "single_coherence_mean": float(
                    np.mean(single["vector_coherence"].coherence[shell_mask])
                ),
                "single_coherence_median": float(
                    np.median(single["vector_coherence"].coherence[shell_mask])
                ),
                "dual_coherence_mean": float(
                    np.mean(dual["vector_coherence"].coherence[shell_mask])
                ),
                "dual_coherence_median": float(
                    np.median(dual["vector_coherence"].coherence[shell_mask])
                ),
            }
        )
    radial_table = pd.DataFrame(radial_rows)

    print("P0698: rotations, translation, resolution, and endpoint limits", flush=True)
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
    expected_rotated_coherence = np.swapaxes(
        dual["vector_coherence"].coherence,
        0,
        1,
    )
    rotation_coherence = relative_grid_rms(
        rotated["vector_coherence"].coherence,
        expected_rotated_coherence,
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
    translated_coherence = baryonic_vector_coherence(
        translated_density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    expected_translated = np.roll(
        single["vector_coherence"].coherence,
        translation_cells,
        axis=0,
    )
    translation_mask = np.zeros((cells,) * 3, dtype=bool)
    translation_mask[5:-5, 5:-5, 5:-5] = True
    translation_rms = float(
        np.sqrt(
            np.mean(
                (
                    translated_coherence.coherence[translation_mask]
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
    high_coherence = baryonic_vector_coherence(
        high_density,
        resolution_spacing,
        gravitational_constant=gravitational_constant,
    )
    high_axis = (
        np.arange(resolution_cells, dtype=float) - (resolution_cells - 1.0) / 2.0
    ) * resolution_spacing
    interpolator = RegularGridInterpolator(
        (high_axis, high_axis, high_axis),
        high_coherence.coherence,
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
                    single["vector_coherence"].coherence[resolution_mask]
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
        single["vector_coherence"].coherence,
        spacing,
    )
    high_acceleration_limit = relative_vector_rms(
        high_base.acceleration,
        high_newtonian.acceleration,
        single_annulus,
    )

    solutions = [single, dual, rotated]
    raw_triangle_excess = max(
        item["vector_coherence"].maximum_triangle_inequality_excess
        for item in solutions
    )
    all_coherence = np.concatenate(
        [item["vector_coherence"].coherence.ravel() for item in solutions]
        + [translated_coherence.coherence.ravel(), high_coherence.coherence.ravel()]
    )
    coherence_min = float(np.min(all_coherence))
    coherence_max = float(np.max(all_coherence))
    source_identity_scores = []
    hybrid_identity_scores = []
    base_boundary_scores = []
    routing_boundary_scores = []
    residual_scores = []
    curl_scores = []
    finite_arrays = []
    for item in solutions:
        expected_source = (
            item["vector_coherence"].coherence * item["coherent"].equation_source
            + (1.0 - item["vector_coherence"].coherence)
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
                item["vector_coherence"].coherence,
                item["vector_coherence"].raw_coherence,
                item["vector_coherence"].unsummed_acceleration_strength,
                *item["vector_coherence"].direct_acceleration,
                item["base"].potential,
                *item["base"].acceleration,
                item["base"].equation_source,
                item["hybrid"].potential,
                *item["hybrid"].acceleration,
            ]
        )
    high_expected_source = (
        single["vector_coherence"].coherence * high_coherent.equation_source
        + (1.0 - single["vector_coherence"].coherence)
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
        is bool(gates["all_coherence_sources_potentials_accelerations_finite"]),
        "triangle_inequality": raw_triangle_excess
        <= float(gates["raw_triangle_inequality_excess_max"]),
        "coherence_lower": coherence_min >= float(gates["coherence_min"]),
        "coherence_upper": coherence_max <= float(gates["coherence_max"]),
        "single_center_outer": single_outer_median
        >= float(gates["single_center_outer_annulus_median_coherence_min"]),
        "two_center_midpoint": dual_midpoint
        <= float(gates["two_center_midpoint_coherence_max"]),
        "two_center_far": dual_far_median
        >= float(gates["two_center_far_annulus_median_coherence_min"]),
        "two_center_contrast": far_midpoint_contrast
        >= float(gates["two_center_far_minus_midpoint_coherence_min"]),
        "rotation_coherence": rotation_coherence
        <= float(gates["rotation_covariance_coherence_relative_RMS_max"]),
        "rotation_potential": rotation_potential
        <= float(gates["rotation_covariance_potential_relative_RMS_max"]),
        "rotation_acceleration": rotation_acceleration
        <= float(gates["rotation_covariance_acceleration_relative_RMS_max"]),
        "translation_coherence": translation_rms
        <= float(gates["translation_covariance_coherence_RMS_max"]),
        "resolution_coherence": resolution_rms
        <= float(gates["resolution_coherence_RMS_max"]),
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
        single_coherence=single["vector_coherence"].coherence,
        single_base_potential=single["base"].potential,
        single_hybrid_potential=single["hybrid"].potential,
        dual_density=dual_density,
        dual_coherence=dual["vector_coherence"].coherence,
        dual_base_potential=dual["base"].potential,
        dual_hybrid_potential=dual["hybrid"].potential,
        high_resolution_single_coherence=high_coherence.coherence,
        high_resolution_axis=high_axis,
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    extent = [x.min(), x.max(), y.min(), y.max()]
    image_single = axes[0, 0].imshow(
        single["vector_coherence"].coherence[:, :, middle].T,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[0, 0].set(title="Single-center vector coherence", xlabel="x", ylabel="y")
    figure.colorbar(image_single, ax=axes[0, 0], shrink=0.75)
    image_dual = axes[0, 1].imshow(
        dual["vector_coherence"].coherence[:, :, middle].T,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[0, 1].set(title="Two-center vector coherence", xlabel="x", ylabel="y")
    figure.colorbar(image_dual, ax=axes[0, 1], shrink=0.75)
    axes[1, 0].plot(
        radial_table.radius,
        radial_table.single_coherence_median,
        marker="o",
        label="single center",
    )
    axes[1, 0].plot(
        radial_table.radius,
        radial_table.dual_coherence_median,
        marker="s",
        label="two centers",
    )
    axes[1, 0].set(title="Shell-median coherence", xlabel="radius", ylabel="coherence")
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
    figure.suptitle("P0698 local vector coherence mathematical audit")
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0698-LOCAL-VECTOR-COHERENCE-MATH-AUDIT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_math_gates_pass": all_pass,
        "candidate_advanced_to_spent_joint_screen": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/local_vector_coherence.py"),
        "coherent_source_sha256": sha256(ROOT / "src/voidscreen/coherent_monopole.py"),
        "routing_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "field_sha256": sha256(field_path),
        "parent_sha256": sha256(failure_path),
        "integrity_gates": integrity,
        "coherence": {
            "minimum": coherence_min,
            "maximum": coherence_max,
            "raw_triangle_inequality_excess_maximum": raw_triangle_excess,
            "single_center_outer_annulus_median": single_outer_median,
            "two_center_midpoint": dual_midpoint,
            "two_center_far_annulus_median": dual_far_median,
            "two_center_far_minus_midpoint": far_midpoint_contrast,
        },
        "covariance": {
            "rotation_coherence_relative_RMS": rotation_coherence,
            "rotation_potential_relative_RMS": rotation_potential,
            "rotation_acceleration_relative_RMS": rotation_acceleration,
            "translation_coherence_RMS": translation_rms,
            "resolution_coherence_RMS": resolution_rms,
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
    summary = f"""# P0698 local vector coherence mathematical audit

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Coherence range / raw triangle excess: **[{coherence_min:.4g}, {coherence_max:.4g}] / {raw_triangle_excess:.4g}**.
- Single outer / two-center midpoint / two-center far coherence: **{single_outer_median:.4g} / {dual_midpoint:.4g} / {dual_far_median:.4g}**.
- Rotation coherence / potential / acceleration RMS: **{rotation_coherence:.4g} / {rotation_potential:.4g} / {rotation_acceleration:.4g}**.
- Translation / resolution coherence RMS: **{translation_rms:.4g} / {resolution_rms:.4g}**.
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
