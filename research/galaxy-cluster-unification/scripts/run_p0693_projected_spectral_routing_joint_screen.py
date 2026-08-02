#!/usr/bin/env python3
"""Run the frozen P0693 real-galaxy and raw-cluster joint screen."""

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
from run_p0635_map_geometry_sensitivity import axisymmetrize, build_density
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

from voidscreen.data import load_curves
from voidscreen.field_solvers import boundary_mask
from voidscreen.metric_lensing_3d import (
    KPC_M,
    normalized_deflection_curl,
    photon_deflection_zero_slip,
)
from voidscreen.source_routing_qumond import (
    projected_baryonic_spectral_anisotropy,
    solve_linear_routing_mixture,
    solve_source_conserving_baryonic_routing,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0693_projected_spectral_routing_joint_screen.json"
SPARC = ROOT / "data" / "raw" / "sparc"
MODEL = "projected_spectral_routing_P0693"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def vector_rms(x_values: np.ndarray, y_values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x_values[mask] ** 2 + y_values[mask] ** 2)))


def mixture_audit(solution, expected_source: np.ndarray) -> dict:
    identity_scale = max(
        float(np.sqrt(np.mean(np.square(expected_source)))),
        np.finfo(float).tiny,
    )
    identity = float(
        np.sqrt(np.mean(np.square(solution.mixed_source - expected_source)))
        / identity_scale
    )
    edge = boundary_mask(solution.mixed_source.shape)
    boundary_scale = max(
        float(np.max(np.abs(solution.routing.boundary_potential[edge]))),
        np.finfo(float).tiny,
    )
    boundary = float(
        np.max(
            np.abs(
                solution.field.potential[edge]
                - solution.routing.boundary_potential[edge]
            )
        )
        / boundary_scale
    )
    finite = bool(
        np.all(np.isfinite(solution.mixed_source))
        and np.all(np.isfinite(solution.field.potential))
        and all(np.all(np.isfinite(item)) for item in solution.field.acceleration)
    )
    return {
        "mixed_source_identity_relative_RMS": identity,
        "field_normalized_residual_RMS": solution.field.normalized_residual_rms,
        "boundary_maximum_relative_mismatch": boundary,
        "finite": finite,
    }


def solve_spectral_mixture(
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
    solution = solve_linear_routing_mixture(routing, spacing, fraction)
    expected_source = (
        (1.0 - fraction) * routing.local_generator_source
        + fraction * routing.routed_source
    )
    return solution, fraction, covariance, eigenvalues, mixture_audit(solution, expected_source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != (
        "frozen_before_any_P0693_candidate_galaxy_field_rotation_or_cluster_photon_topology_score"
    ):
        raise RuntimeError("P0693 protocol is not frozen")
    equation = protocol["equation"]
    expected = protocol["predeclared_integrity_gates"]

    parent_paths = {
        "P0692": ROOT / protocol["generator_parent"],
        "P0635": ROOT / protocol["galaxy_parent"],
        "P0635_geometry": ROOT / protocol["galaxy_geometry_parent"],
        "P0639": ROOT / protocol["sealed_baryon_parent"],
        "P0670": ROOT / protocol["cluster_map_parent"],
    }
    parents = {key: read_json(path) for key, path in parent_paths.items()}
    integrity = {
        "P0692_status": parents["P0692"].get("status") == expected["P0692_status"],
        "P0692_not_advanced": bool(parents["P0692"].get("candidate_advanced"))
        is bool(expected["P0692_candidate_advanced"]),
        "P0635_status": parents["P0635"].get("status") == expected["P0635_status"],
        "P0635_no_velocity_product": bool(
            parents["P0635"]["data_boundary"]["little_things_velocity_products_downloaded"]
        )
        is bool(expected["P0635_velocity_products_downloaded"]),
        "P0639_status": parents["P0639"].get("status") == expected["P0639_status"],
        "P0639_outcomes_sealed": bool(
            parents["P0639"].get("sealed_target_observables_opened")
        )
        is bool(expected["P0639_sealed_target_observables_opened"]),
        "P0670_parent": bool(parents["P0670"].get("all_progression_gates_pass"))
        is bool(expected["P0670_all_progression_gates_pass"]),
        "no_new_constants": int(equation["new_universal_constants"])
        == int(expected["new_universal_constants"]),
        "no_per_object_gravity": int(equation["per_object_gravity_parameters"])
        == int(expected["per_object_gravity_parameters"]),
        "no_fitted_routing": int(equation["fitted_routing_parameters"])
        == int(expected["fitted_routing_parameters"]),
        "no_fitted_photon": int(equation["fitted_photon_parameters"])
        == int(expected["fitted_photon_parameters"]),
        "sealed_targets_untouched": not bool(expected["sealed_target_outcomes_opened"]),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"P0693 integrity failure before scores: {integrity}")

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)

    print("P0693: solving spent DDO154 real-map fields", flush=True)
    galaxy_map_path = ROOT / protocol["galaxy_map_input"]
    with np.load(galaxy_map_path) as data:
        galaxy_axis = data["axis_kpc"].astype(float)
        gas_surface = data["gas_surface_density_solar_kpc2"].astype(float)
        stellar_surface = data["stellar_surface_density_solar_kpc2"].astype(float)
        total_surface = data["total_surface_density_solar_kpc2"].astype(float)
    galaxy_spacing = float(galaxy_axis[1] - galaxy_axis[0])
    gas_axisymmetric = axisymmetrize(gas_surface, galaxy_axis)
    stars_axisymmetric = axisymmetrize(stellar_surface, galaxy_axis)
    total_axisymmetric = gas_axisymmetric + stars_axisymmetric
    baseline_density = build_density(
        gas_surface,
        stellar_surface,
        galaxy_axis,
        float(protocol["spent_galaxy"]["nominal_gas_scale_height_kpc"]),
        float(protocol["spent_galaxy"]["nominal_stellar_scale_height_kpc"]),
    )
    axisymmetric_density = build_density(
        gas_axisymmetric,
        stars_axisymmetric,
        galaxy_axis,
        float(protocol["spent_galaxy"]["nominal_gas_scale_height_kpc"]),
        float(protocol["spent_galaxy"]["nominal_stellar_scale_height_kpc"]),
    )
    galaxy_component_geometry = {}
    for label, surface in (
        ("gas", gas_surface),
        ("stars", stellar_surface),
        ("total", total_surface),
    ):
        fraction, covariance, eigenvalues = projected_baryonic_spectral_anisotropy(
            surface,
            galaxy_spacing,
        )
        galaxy_component_geometry[label] = {
            "fraction": fraction,
            "covariance_kpc2": covariance.tolist(),
            "eigenvalues_kpc2": eigenvalues.tolist(),
        }
    galaxy_solution, galaxy_fraction, galaxy_covariance, galaxy_eigenvalues, galaxy_audit = (
        solve_spectral_mixture(
            baseline_density,
            total_surface,
            galaxy_spacing,
            gravitational_constant=float(
                equation["gravitational_constant_galaxy_kpc_km2_s2_per_solar_mass"]
            ),
            a0=float(equation["a0_galaxy_km2_s2_per_kpc"]),
            light_speed=float(equation["light_speed_galaxy_km_s"]),
            equation=equation,
        )
    )
    (
        galaxy_axisymmetric_solution,
        galaxy_axisymmetric_fraction,
        galaxy_axisymmetric_covariance,
        galaxy_axisymmetric_eigenvalues,
        galaxy_axisymmetric_audit,
    ) = solve_spectral_mixture(
        axisymmetric_density,
        total_axisymmetric,
        galaxy_spacing,
        gravitational_constant=float(
            equation["gravitational_constant_galaxy_kpc_km2_s2_per_solar_mass"]
        ),
        a0=float(equation["a0_galaxy_km2_s2_per_kpc"]),
        light_speed=float(equation["light_speed_galaxy_km_s"]),
        equation=equation,
    )
    galaxy_curves = []
    galaxy_scores = []
    sparc_curve = next(curve for curve in load_curves(SPARC) if curve.metadata.name == "DDO154")
    for variant, solution, fraction in (
        ("lumpy_nominal_total", galaxy_solution, galaxy_fraction),
        (
            "axisymmetric_nominal_total",
            galaxy_axisymmetric_solution,
            galaxy_axisymmetric_fraction,
        ),
    ):
        curve = radial_circular_speed(solution.field, galaxy_axis)
        curve.insert(0, "routing_fraction", fraction)
        curve.insert(0, "variant", variant)
        galaxy_curves.append(curve)
        score = score_curve(
            curve["radius_kpc"].to_numpy(),
            curve["circular_speed_km_s"].to_numpy(),
            sparc_curve.radius_kpc,
            sparc_curve.velocity_observed_kms,
            sparc_curve.velocity_error_kms,
        )
        galaxy_scores.append({"variant": variant, "routing_fraction": fraction, **score})
    galaxy_curve_frame = pd.concat(galaxy_curves, ignore_index=True)
    galaxy_score_frame = pd.DataFrame(galaxy_scores)
    baseline_score = galaxy_scores[0]
    algebraic_score = parents["P0635"]["spent_DDO154_rotation_scores"][
        "algebraic_simple_mond"
    ]
    qumond_score = parents["P0635"]["spent_DDO154_rotation_scores"]["QUMOND_3d_map"]
    galaxy_comparisons = {
        "candidate_RMSE_to_algebraic_MOND_ratio": baseline_score["RMSE_km_s"]
        / float(algebraic_score["RMSE_km_s"]),
        "candidate_weighted_RMSE_to_algebraic_MOND_ratio": baseline_score[
            "weighted_RMSE_km_s"
        ]
        / float(algebraic_score["weighted_RMSE_km_s"]),
        "candidate_RMSE_to_3D_QUMOND_ratio": baseline_score["RMSE_km_s"]
        / float(qumond_score["RMSE_km_s"]),
    }
    np.savez_compressed(
        output / protocol["outputs"]["galaxy_fields"],
        axis_kpc=galaxy_axis,
        lumpy_routing_fraction=galaxy_fraction,
        lumpy_potential_km2_s2=galaxy_solution.field.potential,
        lumpy_acceleration_x_km2_s2_kpc=galaxy_solution.field.acceleration[0],
        lumpy_acceleration_y_km2_s2_kpc=galaxy_solution.field.acceleration[1],
        lumpy_acceleration_z_km2_s2_kpc=galaxy_solution.field.acceleration[2],
        axisymmetric_routing_fraction=galaxy_axisymmetric_fraction,
        axisymmetric_potential_km2_s2=galaxy_axisymmetric_solution.field.potential,
    )
    galaxy_curve_frame.to_csv(output / protocol["outputs"]["galaxy_curves"], index=False)
    galaxy_score_frame.to_csv(output / protocol["outputs"]["galaxy_scores"], index=False)
    print(
        f"DDO154: e2D={galaxy_fraction:.6g}, RMSE={baseline_score['RMSE_km_s']:.4g} km/s, "
        f"MOND ratio={galaxy_comparisons['candidate_RMSE_to_algebraic_MOND_ratio']:.4g}",
        flush=True,
    )

    print("P0693: auditing sealed baryonic geometry without outcomes", flush=True)
    sealed_rows = []
    sealed_map_directory = ROOT / protocol["sealed_baryon_map_directory"]
    for map_path in sorted(sealed_map_directory.glob("*.npz")):
        with np.load(map_path) as data:
            axis = data["axis_kpc"].astype(float)
            surfaces = {
                "gas": data["gas"].astype(float),
                "stars": data["stars"].astype(float),
                "total": data["total"].astype(float),
            }
        spacing = float(axis[1] - axis[0])
        for component, surface in surfaces.items():
            fraction, covariance, eigenvalues = projected_baryonic_spectral_anisotropy(
                surface,
                spacing,
            )
            sealed_rows.append(
                {
                    "galaxy": map_path.stem,
                    "component": component,
                    "routing_fraction": fraction,
                    "lambda_min_kpc2": float(eigenvalues[0]),
                    "lambda_max_kpc2": float(eigenvalues[-1]),
                    "covariance_xx_kpc2": float(covariance[0, 0]),
                    "covariance_xy_kpc2": float(covariance[0, 1]),
                    "covariance_yy_kpc2": float(covariance[1, 1]),
                    "map_sha256": sha256(map_path),
                    "target_outcome_opened": False,
                }
            )
    sealed_geometry = pd.DataFrame(sealed_rows)
    sealed_geometry.to_csv(output / protocol["outputs"]["sealed_geometry"], index=False)

    print("P0693: solving RX J2129 projected-spectral field", flush=True)
    cluster_map_path = ROOT / protocol["cluster_map_input"]
    with np.load(cluster_map_path) as data:
        cluster_axis = data["axis_kpc"].astype(float)
        stellar_surface_cluster = data["stellar_surface_density_msun_kpc2"].astype(float)
        gas_surface_cluster = data["gas_surface_density_msun_kpc2"].astype(float)
        cluster_density = data["stellar_volume_density_kg_m3"].astype(float) + data[
            "gas_volume_density_kg_m3"
        ].astype(float)
        map_a0 = float(data["a0_m_s2"])
    cluster_surface = stellar_surface_cluster + gas_surface_cluster
    cluster_spacing_m = float(cluster_axis[1] - cluster_axis[0]) * KPC_M
    if not np.isclose(map_a0, float(equation["a0_si_m_s2"]), rtol=0.0, atol=0.0):
        raise RuntimeError("P0693 cluster a0 no longer matches the registered map")
    (
        cluster_solution,
        cluster_fraction,
        cluster_covariance,
        cluster_eigenvalues,
        cluster_audit,
    ) = solve_spectral_mixture(
        cluster_density,
        cluster_surface,
        cluster_spacing_m,
        gravitational_constant=float(equation["gravitational_constant_si"]),
        a0=float(equation["a0_si_m_s2"]),
        light_speed=float(equation["light_speed_si_m_s"]),
        equation=equation,
    )
    deflection = photon_deflection_zero_slip(
        cluster_solution.field.acceleration,
        cluster_spacing_m,
        distance_ratio=1.0,
    )
    magnitude = np.hypot(deflection.alpha_x_arcsec, deflection.alpha_y_arcsec)
    x_kpc, y_kpc = np.meshgrid(cluster_axis, cluster_axis, indexing="ij")
    annulus = (np.hypot(x_kpc, y_kpc) >= 15.8) & (np.hypot(x_kpc, y_kpc) <= 76.5)
    cluster_field = {
        **cluster_audit,
        "strong_lens_median_physical_deflection_arcsec": float(np.median(magnitude[annulus])),
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
        covariance_kpc2=cluster_covariance / (KPC_M**2),
        eigenvalues_kpc2=cluster_eigenvalues / (KPC_M**2),
        potential_m2_s2=cluster_solution.field.potential,
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
        raise RuntimeError("P0693 raw-image coverage changed")
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
        compact_report["model_scores"]["GR_plus_cluster_halo"]["heldout"]
        ["exact_radial_RMS_arcsec"]
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
    roots.to_csv(output / "rxj2129_spectral_global_roots.csv", index=False)
    assignments.to_csv(output / "rxj2129_spectral_global_assignments.csv", index=False)

    all_fractions = [
        value["fraction"] for value in galaxy_component_geometry.values()
    ] + [galaxy_axisymmetric_fraction, cluster_fraction] + sealed_geometry[
        "routing_fraction"
    ].astype(float).tolist()
    all_audits = [galaxy_audit, galaxy_axisymmetric_audit, cluster_audit]
    max_identity = max(float(item["mixed_source_identity_relative_RMS"]) for item in all_audits)
    max_residual = max(float(item["field_normalized_residual_RMS"]) for item in all_audits)
    max_boundary = max(float(item["boundary_maximum_relative_mismatch"]) for item in all_audits)
    all_finite = bool(all(bool(item["finite"]) for item in all_audits) and cluster_field["finite"])
    gates = protocol["predeclared_advancement_gates"]
    gate_results = {
        **integrity,
        "spectral_fractions_finite": bool(np.all(np.isfinite(all_fractions)))
        is bool(gates["all_spectral_fractions_finite"]),
        "spectral_fraction_lower": float(np.min(all_fractions))
        >= float(gates["spectral_fraction_min"]),
        "spectral_fraction_upper": float(np.max(all_fractions))
        <= float(gates["spectral_fraction_max"]),
        "axisymmetric_galaxy_fraction": galaxy_axisymmetric_fraction
        <= float(gates["axisymmetric_galaxy_fraction_max"]),
        "sealed_geometry_coverage": int(
            sealed_geometry.loc[sealed_geometry.component.eq("total"), "galaxy"].nunique()
        )
        == int(gates["sealed_baryon_geometry_targets"]),
        "mixed_source_identity": max_identity
        <= float(gates["mixed_source_linear_identity_relative_RMS_max"]),
        "field_residual": max_residual <= float(gates["field_normalized_residual_RMS_max"]),
        "boundary": max_boundary <= float(gates["boundary_maximum_relative_mismatch_max"]),
        "finite": all_finite
        is bool(gates["all_sources_potentials_accelerations_and_deflections_finite"]),
        "galaxy_points": int(baseline_score["points"]) == int(gates["galaxy_rotation_points"]),
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
        "galaxy_bias": abs(float(baseline_score["mean_bias_km_s"]))
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
        "sealed_outcomes_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))

    comparator_curves = pd.read_csv(
        ROOT / "results" / "p0635_ddo154_map_commissioning" / "field_rotation_curves.csv"
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
    for variant, frame in galaxy_curve_frame.groupby("variant", sort=False):
        axes[0, 0].plot(
            frame.radius_kpc,
            frame.circular_speed_km_s,
            label=f"spectral: {variant}",
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
    total_sealed = sealed_geometry.loc[sealed_geometry.component.eq("total")]
    axes[0, 1].hist(
        total_sealed.routing_fraction,
        bins=np.linspace(0.0, 1.0, 11),
        alpha=0.65,
        label="sealed baryon inputs (no outcomes)",
    )
    axes[0, 1].axvline(galaxy_fraction, color="C1", label=f"DDO154 {galaxy_fraction:.3f}")
    axes[0, 1].axvline(cluster_fraction, color="C3", label=f"RX J2129 {cluster_fraction:.3f}")
    axes[0, 1].set(
        title="Calculated baryonic controller",
        xlabel="projected spectral fraction",
        ylabel="systems",
    )
    axes[0, 1].legend(fontsize=8)
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
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle("P0693 parameter-free projected spectral routing joint screen")
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0693-PROJECTED-SPECTRAL-ROUTING-JOINT-SCREEN-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_robustness": all_pass,
        "candidate_advanced_to_sealed_outcomes": False,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "parent_sha256": {key: sha256(path) for key, path in parent_paths.items()},
        "galaxy_map_sha256": sha256(galaxy_map_path),
        "cluster_map_sha256": sha256(cluster_map_path),
        "galaxy_field_sha256": sha256(output / protocol["outputs"]["galaxy_fields"]),
        "cluster_field_sha256": sha256(cluster_field_path),
        "integrity_gates": integrity,
        "spectral_fraction_range_all_permitted_baryon_maps": [
            float(np.min(all_fractions)),
            float(np.max(all_fractions)),
        ],
        "spent_DDO154": {
            "component_geometry": galaxy_component_geometry,
            "candidate_fraction": galaxy_fraction,
            "candidate_covariance_kpc2": galaxy_covariance.tolist(),
            "candidate_eigenvalues_kpc2": galaxy_eigenvalues.tolist(),
            "axisymmetric_fraction": galaxy_axisymmetric_fraction,
            "axisymmetric_covariance_kpc2": galaxy_axisymmetric_covariance.tolist(),
            "axisymmetric_eigenvalues_kpc2": galaxy_axisymmetric_eigenvalues.tolist(),
            "candidate_audit": galaxy_audit,
            "axisymmetric_audit": galaxy_axisymmetric_audit,
            "candidate_score": baseline_score,
            "axisymmetric_score": galaxy_scores[1],
            "algebraic_MOND_comparator": algebraic_score,
            "three_dimensional_QUMOND_comparator": qumond_score,
            "comparisons": galaxy_comparisons,
        },
        "sealed_baryon_geometry_only": {
            "targets": int(total_sealed.galaxy.nunique()),
            "total_fraction_min": float(total_sealed.routing_fraction.min()),
            "total_fraction_median": float(total_sealed.routing_fraction.median()),
            "total_fraction_max": float(total_sealed.routing_fraction.max()),
            "kinematics_opened": False,
            "candidate_scores_computed": False,
        },
        "spent_RXJ2129": {
            "candidate_fraction": cluster_fraction,
            "candidate_covariance_kpc2": (cluster_covariance / (KPC_M**2)).tolist(),
            "candidate_eigenvalues_kpc2": (cluster_eigenvalues / (KPC_M**2)).tolist(),
            "field": cluster_field,
            "fit": cluster_fit,
            "compact_halo_heldout_RMS_arcsec": compact_halo,
            "candidate_to_compact_halo_heldout_RMS_ratio": halo_ratio,
            "topology": topology,
        },
        "maximum_joint_mixed_source_identity_relative_RMS": max_identity,
        "maximum_joint_field_normalized_residual_RMS": max_residual,
        "maximum_joint_boundary_relative_mismatch": max_boundary,
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
    summary = f"""# P0693 projected spectral routing joint screen

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- DDO154 calculated fraction / RMSE: **{galaxy_fraction:.6g} / {baseline_score['RMSE_km_s']:.4g} km/s**.
- DDO154 candidate / algebraic-MOND RMSE ratio: **{galaxy_comparisons['candidate_RMSE_to_algebraic_MOND_ratio']:.4g}**.
- RX J2129 calculated fraction / median deflection: **{cluster_fraction:.6g} / {cluster_field['strong_lens_median_physical_deflection_arcsec']:.4g} arcsec**.
- RX J2129 training / heldout roots: **{cluster_fit['training_roots_converged']}/15 / {cluster_fit['heldout_roots_converged']}/7**.
- RX J2129 heldout RMS / compact-halo ratio: **{heldout_rms:.4g} arcsec / {halo_ratio:.4g}**.
- Missing / observable-surplus / parity-diverse / critical families: **{topology['missing_multiplicity_families']} / {topology['potentially_observable_surplus_families']} / {topology['parity_diverse_families']} / {topology['critical_curve_present_families']}**.
- Failed gates: **{', '.join(failed) if failed else 'none'}**.
- Advanced to robustness: **{'yes' if all_pass else 'no'}**.
- Sealed P0633/P0640 outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
