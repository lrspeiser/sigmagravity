#!/usr/bin/env python3
"""Run the frozen P0690 routed-source radial and raw-topology screen."""

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
from run_p0683_potential_channel_qumond_reconnaissance import (
    KPC_M,
    cluster_scores,
    equal_system_rmse,
    predict_cluster_set,
    prepare_clusters,
    prepare_galaxies,
)
from run_p0684_path_diluted_potential_channel_qumond import solar_predictions

from voidscreen.field_solvers import acceleration_from_potential
from voidscreen.metric_lensing_3d import (
    normalized_deflection_curl,
    photon_deflection_zero_slip,
)
from voidscreen.potential_channel_qumond import (
    path_diluted_potential_channel_acceleration,
)
from voidscreen.source_routing_spherical import source_conserving_spherical_response

DEFAULT_CONFIG = ROOT / "configs" / "p0690_source_routing_empirical_screen.json"
MODEL = "source_routing_P0690"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def galaxy_shallow_limit(
    galaxies: pd.DataFrame,
    protocol: dict,
) -> tuple[pd.DataFrame, float]:
    equation = protocol["equation"]
    common = {
        "a0_m_s2": float(equation["a0_m_s2"]),
        "transition_depth": float(equation["chi_t"]),
        "transition_power": float(equation["transition_power_n"]),
        "path_power": float(equation["path_power_q"]),
    }
    local = path_diluted_potential_channel_acceleration(
        galaxies.g_bar_m_s2.to_numpy(float),
        galaxies.potential_depth.to_numpy(float),
        galaxies.potential_path_ratio.to_numpy(float),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        **common,
    )
    base = path_diluted_potential_channel_acceleration(
        galaxies.g_bar_m_s2.to_numpy(float),
        galaxies.potential_depth.to_numpy(float),
        galaxies.potential_path_ratio.to_numpy(float),
        extra_spatial_channels=0.0,
        **common,
    )
    fractional = np.abs(
        local["predicted_acceleration_m_s2"] / base["predicted_acceleration_m_s2"] - 1.0
    )
    output = galaxies[
        [
            "galaxy",
            "radius_adjusted_kpc",
            "g_bar_m_s2",
            "potential_depth",
            "potential_path_ratio",
            "velocity_observed_adjusted_kms",
        ]
    ].copy()
    output["local_generator_fractional_acceleration"] = fractional
    output["predicted_acceleration_m_s2"] = base["predicted_acceleration_m_s2"]
    output["velocity_predicted_kms"] = np.sqrt(
        output.predicted_acceleration_m_s2.to_numpy(float)
        * output.radius_adjusted_kpc.to_numpy(float)
        * KPC_M
        / 1e6
    )
    output["velocity_residual_kms"] = (
        output.velocity_predicted_kms - output.velocity_observed_adjusted_kms
    )
    return output, float(np.max(fractional))


def routed_cluster_predictions(
    clusters: list[dict],
    protocol: dict,
) -> tuple[pd.DataFrame, dict[str, float]]:
    equation = protocol["equation"]
    responses = {}
    conservation = {}
    for item in clusters:
        response = source_conserving_spherical_response(
            item["radius_grid"] * KPC_M,
            item["gbar_grid"],
            item["potential_depth_grid"],
            item["potential_path_ratio_grid"],
            a0_m_s2=float(equation["a0_m_s2"]),
            transition_depth=float(equation["chi_t"]),
            transition_power=float(equation["transition_power_n"]),
            extra_spatial_channels=float(equation["extra_spatial_channels"]),
            path_power=float(equation["path_power_q"]),
        )
        responses[item["label"]] = response.routed_acceleration_m_s2
        conservation[item["label"]] = response.net_added_flux_fraction
    prediction = predict_cluster_set(
        clusters,
        protocol,
        lambda item: responses[item["label"]],
    )
    return prediction, conservation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0690_radial_photon_or_raw_lens_score":
        raise RuntimeError("P0690 protocol is not frozen")
    parent_path = ROOT / protocol["parent_result"]
    parent = read_json(parent_path)
    expected = protocol["predeclared_integrity_gates"]
    if bool(parent.get("all_progression_gates_pass")) is not bool(
        expected["P0689_all_progression_gates_pass"]
    ):
        raise RuntimeError("P0689 parent state changed")

    galaxy_path = ROOT / protocol["spent_galaxy_test"]["input"]
    galaxies = prepare_galaxies(galaxy_path)
    galaxy_prediction, galaxy_generator_max = galaxy_shallow_limit(galaxies, protocol)
    galaxy_rmse = equal_system_rmse(
        galaxy_prediction,
        "galaxy",
        "velocity_residual_kms",
    )
    comparator_path = ROOT / protocol["radial_comparator_report"]
    radial_comparator = read_json(comparator_path)
    fixed_rar_galaxy = float(radial_comparator["fixed_RAR_comparators"]["galaxy_equal_RMSE_km_s"])
    fixed_rar_cluster = float(
        radial_comparator["fixed_RAR_comparators"]["cluster_all5_log_RMS_dex"]
    )
    galaxy_ratio = galaxy_rmse / fixed_rar_galaxy

    print("evaluating six spent spherical cluster transfers", flush=True)
    clusters = prepare_clusters(protocol)
    cluster_prediction, cluster_conservation = routed_cluster_predictions(clusters, protocol)
    cluster_metric = cluster_scores(cluster_prediction)
    gap_closed = 1.0 - cluster_metric["all5_log_RMS_dex"] / fixed_rar_cluster
    equation = protocol["equation"]
    solar = solar_predictions(
        a0=float(equation["a0_m_s2"]),
        extra_channels=float(equation["extra_spatial_channels"]),
        transition_power=float(equation["transition_power_n"]),
        path_power=float(equation["path_power_q"]),
        transition_depth=float(equation["chi_t"]),
    )

    raw_settings = protocol["raw_lensing"]
    raw = read_json(ROOT / raw_settings["raw_protocol"])
    images = load_images(raw)
    training, heldout = split_images(images, raw)
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    field_path = ROOT / protocol["field_input"]
    with np.load(field_path) as data:
        axis_kpc = data["axis_kpc"].astype(float)
        routed_potential = data["routed_potential_m2_s2"].astype(float)
    spacing_m = float(axis_kpc[1] - axis_kpc[0]) * KPC_M
    acceleration = acceleration_from_potential(routed_potential, spacing_m)
    deflection = photon_deflection_zero_slip(acceleration, spacing_m, distance_ratio=1.0)
    deflection_magnitude = np.hypot(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
    )
    x_kpc, y_kpc = np.meshgrid(axis_kpc, axis_kpc, indexing="ij")
    annulus = (np.hypot(x_kpc, y_kpc) >= 15.8) & (np.hypot(x_kpc, y_kpc) <= 76.5)
    field_median = float(np.median(deflection_magnitude[annulus]))
    field_curl = normalized_deflection_curl(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
        float(axis_kpc[1] - axis_kpc[0]),
    )
    field_finite = bool(
        np.all(np.isfinite(routed_potential))
        and all(np.all(np.isfinite(item)) for item in acceleration)
        and np.all(np.isfinite(deflection_magnitude))
    )

    grid = PhysicalDeflectionGrid(
        axis_kpc / scale,
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
    )
    lens = AbsoluteGridLens(raw, {MODEL: grid})
    fit_protocol = {"nuisance_fit": raw_settings}
    fit = exact_fit(lens, MODEL, training, heldout, fit_protocol, seed_offset=0)
    prediction = pd.concat(
        [fit["training_prediction"], fit["heldout_prediction"]],
        ignore_index=True,
    )
    fit_score = {
        "model": MODEL,
        "training_RMS_arcsec": fit["training_score"]["exact_radial_RMS_arcsec"],
        "training_roots_converged": fit["training_score"]["converged_roots"],
        "heldout_RMS_arcsec": fit["heldout_score"]["exact_radial_RMS_arcsec"],
        "heldout_roots_converged": fit["heldout_score"]["converged_roots"],
        "optimizer_cost": fit["optimizer_cost"],
        "nuisance_parameters_near_bound": near_bound_count(fit["parameters"]),
    }
    parameter_rows = [
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
    roots, assignments, families, critical_maps = global_topology(
        lens,
        MODEL,
        fit,
        images,
        protocol["global_topology"],
    )
    topology = topology_summary(families)
    compact_report = read_json(ROOT / raw_settings["compact_halo_comparator_report"])
    compact_halo = float(
        compact_report["model_scores"]["GR_plus_cluster_halo"]["heldout"]["exact_radial_RMS_arcsec"]
    )
    heldout_rms = float(fit_score["heldout_RMS_arcsec"])
    halo_ratio = heldout_rms / compact_halo if np.isfinite(heldout_rms) else float("inf")

    gates = protocol["predeclared_advancement_gates"]
    solar_by_name = solar.set_index("location").fractional_force_change
    gate_results = {
        "P0689_parent": bool(parent["all_progression_gates_pass"])
        is bool(expected["P0689_all_progression_gates_pass"]),
        "galaxy_coverage": int(galaxies.galaxy.nunique()) == int(expected["galaxy_systems"])
        and len(galaxies) == int(expected["galaxy_points"]),
        "cluster_coverage": len(clusters) == int(expected["cluster_systems"])
        and int(sum(item["primary"] for item in clusters))
        == int(expected["primary_cluster_systems"])
        and int(sum(item["reliable"] for item in clusters))
        == int(expected["reliable_cluster_systems"]),
        "raw_coverage": len(training) == int(expected["training_images"])
        and len(heldout) == int(expected["spent_heldout_images"])
        and int(images.source_family.nunique()) == int(expected["source_families"]),
        "galaxy_shallow_bound": galaxy_generator_max
        <= float(gates["galaxy_local_generator_fractional_acceleration_max"]),
        "galaxy_RAR_competitive": galaxy_ratio
        <= float(gates["galaxy_RMSE_ratio_to_fixed_RAR_max"]),
        "cluster_all5": cluster_metric["all5_log_RMS_dex"]
        <= float(gates["all5_cluster_log_RMS_dex_max"]),
        "cluster_reliable3": cluster_metric["reliable3_log_RMS_dex"]
        <= float(gates["reliable3_cluster_log_RMS_dex_max"]),
        "cluster_gap_closed": gap_closed
        >= float(gates["minimum_all5_gap_fraction_closed_from_fixed_RAR"]),
        "solar_limb": float(solar_by_name.loc["solar_limb"])
        <= float(gates["solar_limb_fractional_force_change_max"]),
        "solar_Earth": float(solar_by_name.loc["Earth"])
        <= float(gates["Earth_fractional_force_change_max"]),
        "solar_Saturn": float(solar_by_name.loc["Saturn"])
        <= float(gates["Saturn_fractional_force_change_max"]),
        "field_finite": field_finite is bool(gates["field_all_finite"]),
        "field_deflection_present": field_median
        >= float(gates["field_strong_lens_median_physical_deflection_arcsec_min"]),
        "field_curl": field_curl <= float(gates["field_normalized_deflection_curl_RMS_max"]),
        "training_roots": int(fit_score["training_roots_converged"])
        == int(gates["training_roots_converged"]),
        "heldout_roots": int(fit_score["heldout_roots_converged"])
        == int(gates["heldout_roots_converged"]),
        "training_RMS": float(fit_score["training_RMS_arcsec"])
        <= float(gates["training_RMS_arcsec_max"]),
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
        "nuisance_bounds": int(fit_score["nuisance_parameters_near_bound"])
        <= int(gates["nuisance_parameters_near_bound_max"]),
        "no_new_constants": int(equation["new_universal_constants"])
        == int(gates["new_universal_constants"]),
        "no_fitted_gravity": int(equation["gravity_parameters_fit_per_object"])
        == int(gates["gravity_parameters_fit_per_object"]),
        "no_fitted_photon_amplitude": int(equation["photon_amplitudes_fit_per_object"])
        == int(gates["photon_amplitudes_fit_per_object"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    galaxy_prediction.to_csv(output / "spent_galaxy_predictions.csv", index=False)
    cluster_prediction.to_csv(output / "spent_cluster_predictions.csv", index=False)
    solar.to_csv(output / "solar_proxies.csv", index=False)
    pd.DataFrame([fit_score]).to_csv(output / "fit_scores.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "nuisance_parameters.csv", index=False)
    prediction.to_csv(output / "exact_predictions.csv", index=False)
    roots.to_csv(output / "global_roots.csv", index=False)
    assignments.to_csv(output / "global_assignments.csv", index=False)
    families.to_csv(output / "family_topology.csv", index=False)
    deflection_path = output / protocol["outputs"]["field"]
    np.savez_compressed(
        deflection_path,
        axis_kpc=axis_kpc,
        alpha_x_physical_arcsec=deflection.alpha_x_arcsec,
        alpha_y_physical_arcsec=deflection.alpha_y_arcsec,
    )
    report = {
        "report_version": "P0690-SOURCE-ROUTING-EMPIRICAL-SCREEN-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_real_2D_and_robustness": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "spherical_operator_sha256": sha256(ROOT / "src/voidscreen/source_routing_spherical.py"),
        "field_sha256": sha256(field_path),
        "deflection_sha256": sha256(deflection_path),
        "coverage": {
            "galaxy_systems": int(galaxies.galaxy.nunique()),
            "galaxy_points": len(galaxies),
            "cluster_systems": len(clusters),
            "training_images": len(training),
            "spent_heldout_images": len(heldout),
            "source_families": int(images.source_family.nunique()),
            "gravity_parameters": int(equation["gravity_parameters_fit_per_object"]),
            "photon_amplitudes": int(equation["photon_amplitudes_fit_per_object"]),
        },
        "galaxy": {
            "local_generator_fractional_acceleration_max": galaxy_generator_max,
            "equal_RMSE_km_s": galaxy_rmse,
            "fixed_RAR_equal_RMSE_km_s": fixed_rar_galaxy,
            "RMSE_ratio_to_fixed_RAR": galaxy_ratio,
        },
        "clusters": {
            **cluster_metric,
            "fixed_RAR_all5_log_RMS_dex": fixed_rar_cluster,
            "all5_gap_fraction_closed": gap_closed,
            "maximum_net_added_flux_fraction": max(cluster_conservation.values()),
        },
        "field": {
            "strong_lens_median_physical_deflection_arcsec": field_median,
            "normalized_deflection_curl_RMS": field_curl,
        },
        "fit_score": fit_score,
        "comparisons": {
            "compact_halo_heldout_RMS_arcsec": compact_halo,
            "candidate_to_compact_halo_heldout_RMS_ratio": halo_ratio,
        },
        "topology": topology,
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for label, group in cluster_prediction.groupby("label", sort=True):
        line = axes[0].plot(
            group.radius_midpoint_kpc,
            group.predicted_radial_deflection_arcsec,
            label=label,
        )[0]
        axes[0].plot(
            group.radius_midpoint_kpc,
            group.target_total_radial_deflection_arcsec,
            linestyle="--",
            color=line.get_color(),
            alpha=0.7,
        )
    axes[0].set(xscale="log", title="Cluster transfer", xlabel="radius (kpc)", ylabel="arcsec")
    axes[0].legend(fontsize=7)
    extent = [axis_kpc[0], axis_kpc[-1], axis_kpc[0], axis_kpc[-1]]
    image = axes[1].imshow(deflection_magnitude.T, origin="lower", extent=extent, cmap="viridis")
    axes[1].set(title="Routed physical deflection", xlabel="x (kpc)", ylabel="y (kpc)")
    figure.colorbar(image, ax=axes[1], shrink=0.75, label="arcsec")
    axes[2].bar(
        ["training", "heldout"],
        [fit_score["training_roots_converged"], fit_score["heldout_roots_converged"]],
    )
    axes[2].scatter(
        ["training", "heldout"],
        [len(training), len(heldout)],
        color="black",
        label="required",
    )
    axes[2].set(title="Exact roots", ylabel="recovered")
    axes[2].legend()
    determinant = critical_maps[1]
    half = float(protocol["global_topology"]["critical_grid_half_width_arcsec"])
    sign_image = axes[3].imshow(
        np.sign(determinant).T,
        origin="lower",
        extent=[-half, half, -half, half],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[3].set(title="Family 1 Jacobian sign", xlabel="x (arcsec)", ylabel="y (arcsec)")
    figure.colorbar(sign_image, ax=axes[3], shrink=0.75)
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0690 source-routing empirical screen

- Status: **{"PASS" if all_pass else "FAIL"}**.
- Galaxy shallow bound / RMSE ratio to fixed RAR: **{galaxy_generator_max:.3g} / {galaxy_ratio:.4g}**.
- Cluster all-five / reliable-three log RMS: **{cluster_metric["all5_log_RMS_dex"]:.4g} / {cluster_metric["reliable3_log_RMS_dex"]:.4g} dex**.
- Fixed-RAR cluster gap closed: **{100 * gap_closed:.3g}%**.
- Routed strong-lens median physical deflection / curl: **{field_median:.4g} arcsec / {field_curl:.3g}**.
- Training / heldout exact roots: **{fit_score["training_roots_converged"]}/15 / {fit_score["heldout_roots_converged"]}/7**.
- Missing / acceptable / parity-diverse / critical families: **{topology["missing_multiplicity_families"]} / {topology["exact_or_demagnified_only_families"]} / {topology["parity_diverse_families"]} / {topology["critical_curve_present_families"]}**.
- Failed frozen gates: **{", ".join(failed) if failed else "none"}**.
- Sealed P0633/P0640 targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
