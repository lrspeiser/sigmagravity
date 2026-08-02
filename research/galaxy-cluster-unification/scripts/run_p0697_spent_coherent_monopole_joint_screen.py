#!/usr/bin/env python3
"""Run the frozen P0697 spent DDO154/RXJ2129 coherent joint screen."""

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
from run_p0635_ddo154_map_commissioning import radial_circular_speed, score_curve
from run_p0635_map_geometry_sensitivity import build_density
from run_p0660_exact_tensor_activation_audit import sha256
from run_p0672_spent_rxj2129_absolute_raw_topology import (
    AbsoluteGridLens,
    PhysicalDeflectionGrid,
    exact_fit,
    global_topology,
    load_images,
    near_bound_count,
    split_images,
    topology_summary,
)

from voidscreen.coherent_monopole import (
    coherent_monopole_potential,
    hybrid_coherent_routing_potential,
)
from voidscreen.data import load_curves
from voidscreen.field_solvers import (
    boundary_mask,
    laplacian,
    normalized_residual_rms,
    solve_poisson_dirichlet,
)
from voidscreen.metric_lensing_3d import (
    KPC_M,
    normalized_deflection_curl,
    photon_deflection_zero_slip,
)
from voidscreen.radial_path_potential import normalized_acceleration_curl
from voidscreen.source_routing_qumond import (
    projected_baryonic_spectral_anisotropy,
    solve_source_conserving_baryonic_routing,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0697_spent_coherent_monopole_joint_screen.json"
SPARC = ROOT / "data" / "raw" / "sparc"
MODEL = "coherent_monopole_projected_routing_P0697"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def vector_rms(x_values: np.ndarray, y_values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x_values[mask] ** 2 + y_values[mask] ** 2)))


def relative_grid_rms(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    numerator = float(np.sqrt(np.mean((left[mask] - right[mask]) ** 2)))
    denominator = float(np.sqrt(np.mean(right[mask] ** 2)))
    return numerator / max(denominator, np.finfo(float).tiny)


def solve_joint_field(
    density: np.ndarray,
    surface_density: np.ndarray,
    spacing: float,
    *,
    gravitational_constant: float,
    a0: float,
    light_speed: float,
    equation: dict,
):
    fraction, covariance, eigenvalues = projected_baryonic_spectral_anisotropy(
        surface_density,
        spacing,
    )
    routing = solve_source_conserving_baryonic_routing(
        density,
        spacing,
        gravitational_constant=gravitational_constant,
        a0=a0,
        transition_depth=float(equation["chi_t"]),
        transition_power=float(equation["transition_power_n"]),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        path_power=float(equation["path_power_q"]),
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
    joint = hybrid_coherent_routing_potential(
        coherent,
        local_potential,
        routing.field.potential,
        spacing,
        fraction,
    )
    interior = np.zeros(density.shape, dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True
    coherent_identity = relative_grid_rms(
        coherent.potential,
        routing.newtonian.potential + coherent.correction_potential,
        interior,
    )
    expected_joint = coherent.potential + fraction * (
        routing.field.potential - local_potential
    )
    joint_identity = relative_grid_rms(joint.potential, expected_joint, interior)
    local_residual = normalized_residual_rms(
        laplacian(local_potential, spacing) - routing.local_generator_source,
        routing.local_generator_source,
    )
    edge = boundary_mask(density.shape)
    boundary_scale = max(
        float(np.max(np.abs(routing.boundary_potential[edge]))),
        np.finfo(float).tiny,
    )
    routing_boundary = float(
        np.max(np.abs((routing.field.potential - local_potential)[edge]))
        / boundary_scale
    )
    finite = bool(
        all(
            np.all(np.isfinite(array))
            for array in (
                routing.newtonian.potential,
                *routing.newtonian.acceleration,
                local_potential,
                routing.field.potential,
                *routing.field.acceleration,
                coherent.potential,
                *coherent.acceleration,
                coherent.equation_source,
                joint.potential,
                *joint.acceleration,
                joint.equation_source,
            )
        )
    )
    audit = {
        "newtonian_normalized_residual_RMS": routing.newtonian.normalized_residual_rms,
        "local_routing_component_normalized_residual_RMS": local_residual,
        "routed_component_normalized_residual_RMS": routing.field.normalized_residual_rms,
        "coherent_potential_identity_relative_RMS": coherent_identity,
        "hybrid_potential_identity_relative_RMS": joint_identity,
        "routing_correction_boundary_relative_mismatch": routing_boundary,
        "normalized_acceleration_curl_RMS": normalized_acceleration_curl(
            joint.acceleration,
            spacing,
        ),
        "finite": finite,
    }
    return joint, coherent, routing, local_potential, fraction, covariance, eigenvalues, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != (
        "frozen_before_any_P0697_DDO154_rotation_or_RXJ2129_field_fit_or_topology_score"
    ):
        raise RuntimeError("P0697 protocol is not frozen")
    equation = protocol["equation"]
    expected = protocol["predeclared_integrity_gates"]
    parent_paths = {
        "P0696": ROOT / protocol["math_parent"],
        "P0635": ROOT / protocol["galaxy_parent"],
        "P0670": ROOT / protocol["cluster_map_parent"],
        "P0693": ROOT / protocol["raw_method_parent"],
    }
    parents = {key: read_json(path) for key, path in parent_paths.items()}
    integrity = {
        "P0696_status": parents["P0696"].get("status") == expected["P0696_status"],
        "P0696_math_pass": bool(parents["P0696"].get("all_math_gates_pass"))
        is bool(expected["P0696_all_math_gates_pass"]),
        "P0696_advanced": bool(
            parents["P0696"].get("candidate_advanced_to_spent_joint_screen")
        )
        is bool(expected["P0696_candidate_advanced_to_spent_joint_screen"]),
        "P0635_status": parents["P0635"].get("status") == expected["P0635_status"],
        "P0635_no_velocity_product": bool(
            parents["P0635"]["data_boundary"][
                "little_things_velocity_products_downloaded"
            ]
        )
        is bool(expected["P0635_velocity_products_downloaded"]),
        "P0670_parent": bool(parents["P0670"].get("all_progression_gates_pass"))
        is bool(expected["P0670_all_progression_gates_pass"]),
        "P0693_status": parents["P0693"].get("status") == expected["P0693_status"],
        "spent_systems_registered": bool(
            "spent_DDO154" in parents["P0693"] and "spent_RXJ2129" in parents["P0693"]
        )
        is bool(expected["P0693_DDO154_and_RXJ2129_spent"]),
        "no_new_constants": int(equation["new_universal_constants"])
        == int(expected["new_universal_constants"]),
        "no_per_object_gravity": int(equation["per_object_gravity_parameters"])
        == int(expected["per_object_gravity_parameters"]),
        "no_fitted_routing": int(equation["fitted_routing_parameters"])
        == int(expected["fitted_routing_parameters"]),
        "no_fitted_photon": int(equation["fitted_photon_parameters"])
        == int(expected["fitted_photon_parameters"]),
        "nuisance_parameter_count": int(
            equation["fitted_observational_nuisance_parameters"]
        )
        == int(expected["fitted_observational_nuisance_parameters"]),
        "sealed_targets_untouched": not bool(expected["sealed_target_outcomes_opened"]),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"P0697 integrity failure before scores: {integrity}")

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)

    print("P0697: solving spent DDO154 coherent joint field", flush=True)
    galaxy_map_path = ROOT / protocol["galaxy_map_input"]
    with np.load(galaxy_map_path) as data:
        galaxy_axis = data["axis_kpc"].astype(float)
        gas_surface = data["gas_surface_density_solar_kpc2"].astype(float)
        stellar_surface = data["stellar_surface_density_solar_kpc2"].astype(float)
        total_surface = data["total_surface_density_solar_kpc2"].astype(float)
    galaxy_spacing = float(galaxy_axis[1] - galaxy_axis[0])
    galaxy_density = build_density(
        gas_surface,
        stellar_surface,
        galaxy_axis,
        float(protocol["spent_galaxy"]["gas_scale_height_kpc"]),
        float(protocol["spent_galaxy"]["stellar_scale_height_kpc"]),
    )
    (
        galaxy_joint,
        galaxy_coherent,
        galaxy_routing,
        galaxy_local,
        galaxy_fraction,
        galaxy_covariance,
        galaxy_eigenvalues,
        galaxy_audit,
    ) = solve_joint_field(
        galaxy_density,
        total_surface,
        galaxy_spacing,
        gravitational_constant=float(
            equation["gravitational_constant_galaxy_kpc_km2_s2_per_solar_mass"]
        ),
        a0=float(equation["a0_galaxy_km2_s2_per_kpc"]),
        light_speed=float(equation["light_speed_galaxy_km_s"]),
        equation=equation,
    )
    galaxy_curve = radial_circular_speed(galaxy_joint, galaxy_axis)
    galaxy_curve.insert(0, "routing_fraction", galaxy_fraction)
    galaxy_curve.insert(0, "variant", "coherent_monopole_projected_routing")
    sparc_curve = next(
        curve for curve in load_curves(SPARC) if curve.metadata.name == "DDO154"
    )
    galaxy_score = score_curve(
        galaxy_curve.radius_kpc.to_numpy(),
        galaxy_curve.circular_speed_km_s.to_numpy(),
        sparc_curve.radius_kpc,
        sparc_curve.velocity_observed_kms,
        sparc_curve.velocity_error_kms,
    )
    algebraic_score = parents["P0635"]["spent_DDO154_rotation_scores"][
        "algebraic_simple_mond"
    ]
    qumond_score = parents["P0635"]["spent_DDO154_rotation_scores"]["QUMOND_3d_map"]
    galaxy_comparisons = {
        "candidate_RMSE_to_algebraic_MOND_ratio": galaxy_score["RMSE_km_s"]
        / float(algebraic_score["RMSE_km_s"]),
        "candidate_weighted_RMSE_to_algebraic_MOND_ratio": galaxy_score[
            "weighted_RMSE_km_s"
        ]
        / float(algebraic_score["weighted_RMSE_km_s"]),
        "candidate_RMSE_to_3D_QUMOND_ratio": galaxy_score["RMSE_km_s"]
        / float(qumond_score["RMSE_km_s"]),
    }
    galaxy_curve.to_csv(output / protocol["outputs"]["galaxy_curves"], index=False)
    pd.DataFrame(
        [{**galaxy_score, **galaxy_comparisons, "routing_fraction": galaxy_fraction}]
    ).to_csv(output / protocol["outputs"]["galaxy_scores"], index=False)
    galaxy_field_path = output / protocol["outputs"]["galaxy_field"]
    np.savez_compressed(
        galaxy_field_path,
        axis_kpc=galaxy_axis,
        routing_fraction=galaxy_fraction,
        covariance_kpc2=galaxy_covariance,
        eigenvalues_kpc2=galaxy_eigenvalues,
        newtonian_potential_km2_s2=galaxy_routing.newtonian.potential,
        coherent_potential_km2_s2=galaxy_coherent.potential,
        routing_correction_potential_km2_s2=galaxy_routing.field.potential
        - galaxy_local,
        joint_potential_km2_s2=galaxy_joint.potential,
        joint_acceleration_x_km2_s2_kpc=galaxy_joint.acceleration[0],
        joint_acceleration_y_km2_s2_kpc=galaxy_joint.acceleration[1],
        joint_acceleration_z_km2_s2_kpc=galaxy_joint.acceleration[2],
    )
    print(
        f"DDO154: e2D={galaxy_fraction:.6g}, RMSE={galaxy_score['RMSE_km_s']:.4g} km/s, "
        f"MOND ratio={galaxy_comparisons['candidate_RMSE_to_algebraic_MOND_ratio']:.4g}",
        flush=True,
    )

    print("P0697: solving spent RX J2129 coherent joint field", flush=True)
    cluster_map_path = ROOT / protocol["cluster_map_input"]
    with np.load(cluster_map_path) as data:
        cluster_axis = data["axis_kpc"].astype(float)
        cluster_surface = data["stellar_surface_density_msun_kpc2"].astype(float) + data[
            "gas_surface_density_msun_kpc2"
        ].astype(float)
        cluster_density = data["stellar_volume_density_kg_m3"].astype(float) + data[
            "gas_volume_density_kg_m3"
        ].astype(float)
        map_a0 = float(data["a0_m_s2"])
    if not np.isclose(map_a0, float(equation["a0_si_m_s2"]), rtol=0.0, atol=0.0):
        raise RuntimeError("P0697 cluster a0 no longer matches the registered map")
    cluster_spacing_m = float(cluster_axis[1] - cluster_axis[0]) * KPC_M
    (
        cluster_joint,
        cluster_coherent,
        cluster_routing,
        cluster_local,
        cluster_fraction,
        cluster_covariance,
        cluster_eigenvalues,
        cluster_audit,
    ) = solve_joint_field(
        cluster_density,
        cluster_surface,
        cluster_spacing_m,
        gravitational_constant=float(equation["gravitational_constant_si"]),
        a0=float(equation["a0_si_m_s2"]),
        light_speed=float(equation["light_speed_si_m_s"]),
        equation=equation,
    )
    deflection = photon_deflection_zero_slip(
        cluster_joint.acceleration,
        cluster_spacing_m,
        distance_ratio=1.0,
    )
    magnitude = np.hypot(deflection.alpha_x_arcsec, deflection.alpha_y_arcsec)
    x_kpc, y_kpc = np.meshgrid(cluster_axis, cluster_axis, indexing="ij")
    annulus = (np.hypot(x_kpc, y_kpc) >= 15.8) & (np.hypot(x_kpc, y_kpc) <= 76.5)
    cluster_field = {
        **cluster_audit,
        "strong_lens_median_physical_deflection_arcsec": float(
            np.median(magnitude[annulus])
        ),
        "strong_lens_RMS_physical_deflection_arcsec": vector_rms(
            deflection.alpha_x_arcsec,
            deflection.alpha_y_arcsec,
            annulus,
        ),
        "normalized_deflection_curl_RMS": normalized_deflection_curl(
            deflection.alpha_x_arcsec,
            deflection.alpha_y_arcsec,
            float(cluster_axis[1] - cluster_axis[0]),
        ),
    }
    cluster_field["finite"] = bool(
        cluster_field["finite"] and np.all(np.isfinite(magnitude))
    )
    cluster_field_path = output / protocol["outputs"]["cluster_field"]
    np.savez_compressed(
        cluster_field_path,
        axis_kpc=cluster_axis,
        routing_fraction=cluster_fraction,
        covariance_kpc2=cluster_covariance / KPC_M**2,
        eigenvalues_kpc2=cluster_eigenvalues / KPC_M**2,
        coherent_potential_m2_s2=cluster_coherent.potential,
        routing_correction_potential_m2_s2=cluster_routing.field.potential
        - cluster_local,
        joint_potential_m2_s2=cluster_joint.potential,
        alpha_x_physical_arcsec=deflection.alpha_x_arcsec,
        alpha_y_physical_arcsec=deflection.alpha_y_arcsec,
    )

    raw = read_json(ROOT / protocol["raw_protocol"])
    images = load_images(raw)
    training, heldout = split_images(images, raw)
    integrity["raw_coverage"] = (
        len(training) == int(expected["training_images"])
        and len(heldout) == int(expected["spent_heldout_images"])
        and int(images.source_family.nunique()) == int(expected["source_families"])
    )
    if not integrity["raw_coverage"]:
        raise RuntimeError("P0697 raw-image coverage changed")
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    grid = PhysicalDeflectionGrid(
        cluster_axis / scale,
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
    )
    lens = AbsoluteGridLens(raw, {MODEL: grid})
    fit = exact_fit(lens, MODEL, training, heldout, protocol, seed_offset=0)
    training_rms = float(fit["training_score"]["exact_radial_RMS_arcsec"])
    heldout_rms = float(fit["heldout_score"]["exact_radial_RMS_arcsec"])
    cluster_fit = {
        "model": MODEL,
        "training_RMS_arcsec": training_rms,
        "training_roots_converged": int(fit["training_score"]["converged_roots"]),
        "heldout_RMS_arcsec": heldout_rms,
        "heldout_roots_converged": int(fit["heldout_score"]["converged_roots"]),
        "optimizer_cost": float(fit["optimizer_cost"]),
        "nuisance_parameters_near_bound": near_bound_count(fit["parameters"]),
    }
    roots, assignments, families, critical_maps = global_topology(
        lens,
        MODEL,
        fit,
        images,
        protocol["global_topology"],
    )
    topology = topology_summary(families)
    compact_report = read_json(ROOT / protocol["compact_halo_comparator_report"])
    compact_halo = float(
        compact_report["model_scores"]["GR_plus_cluster_halo"]["heldout"][
            "exact_radial_RMS_arcsec"
        ]
    )
    halo_ratio = heldout_rms / compact_halo if np.isfinite(heldout_rms) else float("inf")
    nuisance_rows = [
        {
            "model": MODEL,
            "parameter": label,
            "value": float(value),
            "lower": float(lower),
            "upper": float(upper),
        }
        for label, value, lower, upper in zip(
            AbsoluteGridLens.labels,
            fit["parameters"],
            AbsoluteGridLens.lower,
            AbsoluteGridLens.upper,
            strict=True,
        )
    ]
    families.to_csv(output / protocol["outputs"]["cluster_family_topology"], index=False)
    pd.DataFrame(nuisance_rows).to_csv(
        output / protocol["outputs"]["cluster_nuisance"],
        index=False,
    )
    roots.to_csv(output / "rxj2129_coherent_global_roots.csv", index=False)
    assignments.to_csv(output / "rxj2129_coherent_global_assignments.csv", index=False)

    audits = [galaxy_audit, cluster_audit]
    maximum_component_residual = max(
        float(audit[key])
        for audit in audits
        for key in (
            "newtonian_normalized_residual_RMS",
            "local_routing_component_normalized_residual_RMS",
            "routed_component_normalized_residual_RMS",
        )
    )
    maximum_coherent_identity = max(
        float(audit["coherent_potential_identity_relative_RMS"]) for audit in audits
    )
    maximum_joint_identity = max(
        float(audit["hybrid_potential_identity_relative_RMS"]) for audit in audits
    )
    maximum_boundary = max(
        float(audit["routing_correction_boundary_relative_mismatch"]) for audit in audits
    )
    maximum_acceleration_curl = max(
        float(audit["normalized_acceleration_curl_RMS"]) for audit in audits
    )
    all_finite = bool(
        all(bool(audit["finite"]) for audit in audits) and cluster_field["finite"]
    )
    gates = protocol["predeclared_advancement_gates"]
    gate_results = {
        **integrity,
        "fraction_lower": min(galaxy_fraction, cluster_fraction)
        >= float(gates["spectral_fraction_min"]),
        "fraction_upper": max(galaxy_fraction, cluster_fraction)
        <= float(gates["spectral_fraction_max"]),
        "finite": all_finite
        is bool(gates["all_potentials_accelerations_sources_and_deflections_finite"]),
        "component_residuals": maximum_component_residual
        <= float(gates["newtonian_and_routing_component_normalized_residual_RMS_max"]),
        "coherent_identity": maximum_coherent_identity
        <= float(gates["coherent_potential_identity_relative_RMS_max"]),
        "hybrid_identity": maximum_joint_identity
        <= float(gates["hybrid_potential_identity_relative_RMS_max"]),
        "routing_boundary": maximum_boundary
        <= float(gates["routing_correction_boundary_relative_mismatch_max"]),
        "acceleration_curl": maximum_acceleration_curl
        <= float(gates["normalized_acceleration_curl_RMS_max"]),
        "galaxy_points": int(galaxy_score["points"])
        == int(gates["galaxy_rotation_points"]),
        "galaxy_RMSE_vs_MOND": galaxy_comparisons[
            "candidate_RMSE_to_algebraic_MOND_ratio"
        ]
        <= float(gates["galaxy_candidate_RMSE_to_algebraic_MOND_ratio_max"]),
        "galaxy_weighted_RMSE_vs_MOND": galaxy_comparisons[
            "candidate_weighted_RMSE_to_algebraic_MOND_ratio"
        ]
        <= float(gates["galaxy_candidate_weighted_RMSE_to_algebraic_MOND_ratio_max"]),
        "galaxy_RMSE_vs_3D_QUMOND": galaxy_comparisons[
            "candidate_RMSE_to_3D_QUMOND_ratio"
        ]
        <= float(gates["galaxy_candidate_RMSE_to_3D_QUMOND_ratio_max"]),
        "galaxy_bias": abs(float(galaxy_score["mean_bias_km_s"]))
        <= float(gates["galaxy_absolute_mean_bias_km_s_max"]),
        "cluster_field_amplitude_lower": cluster_field[
            "strong_lens_median_physical_deflection_arcsec"
        ]
        >= float(gates["cluster_strong_lens_median_physical_deflection_arcsec_min"]),
        "cluster_field_amplitude_upper": cluster_field[
            "strong_lens_median_physical_deflection_arcsec"
        ]
        <= float(gates["cluster_strong_lens_median_physical_deflection_arcsec_max"]),
        "cluster_field_curl": cluster_field["normalized_deflection_curl_RMS"]
        <= float(gates["cluster_normalized_deflection_curl_RMS_max"]),
        "training_roots": cluster_fit["training_roots_converged"]
        == int(gates["training_roots_converged"]),
        "heldout_roots": cluster_fit["heldout_roots_converged"]
        == int(gates["heldout_roots_converged"]),
        "training_RMS": training_rms <= float(gates["training_RMS_arcsec_max"]),
        "heldout_RMS": heldout_rms <= float(gates["heldout_RMS_arcsec_max"]),
        "compact_halo_comparison": halo_ratio
        <= float(gates["candidate_to_compact_halo_heldout_RMS_ratio_max"]),
        "no_missing_multiplicity": topology["missing_multiplicity_families"]
        <= int(gates["missing_multiplicity_families_max"]),
        "observable_surplus": topology["potentially_observable_surplus_families"]
        <= int(gates["potentially_observable_surplus_families_max"]),
        "acceptable_multiplicity": topology["exact_or_demagnified_only_families"]
        >= int(gates["exact_or_demagnified_only_families_min"]),
        "parity_diversity": topology["parity_diverse_families"]
        >= int(gates["parity_diverse_families_min"]),
        "critical_curves": topology["critical_curve_present_families"]
        >= int(gates["critical_curve_present_families_min"]),
        "nuisance_bounds": cluster_fit["nuisance_parameters_near_bound"]
        <= int(gates["nuisance_parameters_near_bound_max"]),
        "accounting_no_new_constants": int(equation["new_universal_constants"])
        == int(gates["new_universal_constants"]),
        "accounting_no_per_object_gravity": int(equation["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "accounting_no_fitted_routing": int(equation["fitted_routing_parameters"])
        == int(gates["fitted_routing_parameters"]),
        "accounting_no_fitted_photon": int(equation["fitted_photon_parameters"])
        == int(gates["fitted_photon_parameters"]),
        "accounting_nuisance_count": int(
            equation["fitted_observational_nuisance_parameters"]
        )
        == int(gates["fitted_observational_nuisance_parameters"]),
        "sealed_outcomes_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))

    comparator_curves = pd.read_csv(
        ROOT / "results/p0635_ddo154_map_commissioning/field_rotation_curves.csv"
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].errorbar(
        sparc_curve.radius_kpc,
        sparc_curve.velocity_observed_kms,
        yerr=sparc_curve.velocity_error_kms,
        fmt="o",
        color="black",
        label="spent DDO154 observations",
    )
    axes[0, 0].plot(
        galaxy_curve.radius_kpc,
        galaxy_curve.circular_speed_km_s,
        label="coherent joint",
    )
    for law, style in (("QUMOND_3d_map", "--"), ("algebraic_simple_mond", ":")):
        frame = comparator_curves.loc[comparator_curves.law.eq(law)]
        axes[0, 0].plot(
            frame.radius_kpc,
            frame.circular_speed_km_s,
            linestyle=style,
            label=law.replace("_", " "),
        )
    axes[0, 0].set(
        title="Spent real-map galaxy",
        xlabel="radius (kpc)",
        ylabel="circular speed (km/s)",
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].bar(
        ["DDO154", "RX J2129"],
        [galaxy_fraction, cluster_fraction],
        color=["C1", "C3"],
    )
    axes[0, 1].set(title="Baryon-derived routing fraction", ylabel="e2D", ylim=(0, 1))
    axes[1, 0].bar(
        ["training", "heldout"],
        [cluster_fit["training_roots_converged"], cluster_fit["heldout_roots_converged"]],
    )
    axes[1, 0].scatter(["training", "heldout"], [15, 7], color="black", label="required")
    axes[1, 0].set(title="RX J2129 exact roots", ylabel="recovered")
    axes[1, 0].legend()
    topology_labels = ["missing", "exact", "observable surplus", "parity diverse", "critical"]
    topology_values = [
        topology["missing_multiplicity_families"],
        topology["exact_multiplicity_families"],
        topology["potentially_observable_surplus_families"],
        topology["parity_diverse_families"],
        topology["critical_curve_present_families"],
    ]
    axes[1, 1].bar(topology_labels, topology_values)
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].set(title="RX J2129 global topology", ylabel="families")
    for axis_plot in axes.ravel():
        axis_plot.grid(alpha=0.2)
    figure.suptitle("P0697 spent coherent-monopole joint screen")
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0697-SPENT-COHERENT-MONOPOLE-JOINT-SCREEN-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "evidence_class": protocol["evidence_class"],
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_robustness_and_solar": all_pass,
        "candidate_advanced_to_sealed_outcomes": False,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/coherent_monopole.py"),
        "routing_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "parent_sha256": {key: sha256(path) for key, path in parent_paths.items()},
        "galaxy_map_sha256": sha256(galaxy_map_path),
        "cluster_map_sha256": sha256(cluster_map_path),
        "galaxy_field_sha256": sha256(galaxy_field_path),
        "cluster_field_sha256": sha256(cluster_field_path),
        "integrity_gates": integrity,
        "spent_DDO154": {
            "routing_fraction": galaxy_fraction,
            "covariance_kpc2": galaxy_covariance.tolist(),
            "eigenvalues_kpc2": galaxy_eigenvalues.tolist(),
            "field_audit": galaxy_audit,
            "candidate_score": galaxy_score,
            "algebraic_MOND_comparator": algebraic_score,
            "three_dimensional_QUMOND_comparator": qumond_score,
            "comparisons": galaxy_comparisons,
        },
        "spent_RXJ2129": {
            "routing_fraction": cluster_fraction,
            "covariance_kpc2": (cluster_covariance / KPC_M**2).tolist(),
            "eigenvalues_kpc2": (cluster_eigenvalues / KPC_M**2).tolist(),
            "field": cluster_field,
            "fit": cluster_fit,
            "compact_halo_heldout_RMS_arcsec": compact_halo,
            "candidate_to_compact_halo_heldout_RMS_ratio": halo_ratio,
            "topology": topology,
        },
        "maximum_component_normalized_residual_RMS": maximum_component_residual,
        "maximum_coherent_potential_identity_relative_RMS": maximum_coherent_identity,
        "maximum_hybrid_potential_identity_relative_RMS": maximum_joint_identity,
        "maximum_routing_correction_boundary_relative_mismatch": maximum_boundary,
        "maximum_normalized_acceleration_curl_RMS": maximum_acceleration_curl,
        "gate_results": gate_results,
        "failed_gates": [name for name, passed in gate_results.items() if not passed],
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )
    failed = report["failed_gates"]
    summary = f"""# P0697 spent coherent-monopole joint screen

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- DDO154 routing fraction / RMSE / weighted RMSE: **{galaxy_fraction:.6g} / {galaxy_score['RMSE_km_s']:.4g} / {galaxy_score['weighted_RMSE_km_s']:.4g} km/s**.
- DDO154 ordinary / weighted algebraic-MOND ratios: **{galaxy_comparisons['candidate_RMSE_to_algebraic_MOND_ratio']:.4g} / {galaxy_comparisons['candidate_weighted_RMSE_to_algebraic_MOND_ratio']:.4g}**.
- RX J2129 routing fraction / median deflection: **{cluster_fraction:.6g} / {cluster_field['strong_lens_median_physical_deflection_arcsec']:.4g} arcsec**.
- RX J2129 training / heldout roots: **{cluster_fit['training_roots_converged']}/15 / {cluster_fit['heldout_roots_converged']}/7**.
- RX J2129 training / heldout RMS / compact-halo ratio: **{training_rms:.4g} / {heldout_rms:.4g} arcsec / {halo_ratio:.4g}**.
- Missing / observable-surplus / parity-diverse / critical families: **{topology['missing_multiplicity_families']} / {topology['potentially_observable_surplus_families']} / {topology['parity_diverse_families']} / {topology['critical_curve_present_families']}**.
- Failed gates: **{', '.join(failed) if failed else 'none'}**.
- Advanced to robustness and Solar-System gates: **{'yes' if all_pass else 'no'}**.
- Sealed P0633/P0640 outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
