#!/usr/bin/env python3
"""Run the frozen P0684 path-diluted potential-channel QUMOND test."""

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

from run_p0683_potential_channel_qumond_reconnaissance import (  # noqa: E402
    AU_M,
    C_M_S,
    G_SI,
    KPC_M,
    M_SUN_KG,
    R_SUN_M,
    cluster_scores,
    equal_system_rmse,
    json_safe,
    predict_cluster_set,
    prepare_clusters,
    prepare_galaxies,
    read_json,
    sha256,
)
from voidscreen.potential_channel_qumond import (  # noqa: E402
    path_diluted_potential_channel_acceleration,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0684_path_diluted_potential_channel_qumond.json"


def galaxy_predictions(
    galaxies: pd.DataFrame,
    *,
    a0: float,
    extra_channels: float,
    transition_power: float,
    path_power: float,
    transition_depth: float,
) -> pd.DataFrame:
    response = path_diluted_potential_channel_acceleration(
        galaxies.g_bar_m_s2.to_numpy(float),
        galaxies.potential_depth.to_numpy(float),
        galaxies.potential_path_ratio.to_numpy(float),
        a0_m_s2=a0,
        transition_depth=transition_depth,
        transition_power=transition_power,
        extra_spatial_channels=extra_channels,
        path_power=path_power,
    )
    velocity = np.sqrt(
        response["predicted_acceleration_m_s2"]
        * galaxies.radius_adjusted_kpc.to_numpy(float)
        * KPC_M
        / 1.0e6
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
    for name in (
        "potential_onset",
        "path_survival",
        "channel_exponent",
        "base_qumond_boost",
        "enhancement",
        "predicted_acceleration_m_s2",
    ):
        output[name] = response[name]
    output["velocity_predicted_kms"] = velocity
    output["velocity_residual_kms"] = velocity - output.velocity_observed_adjusted_kms
    return output


def solar_predictions(
    *,
    a0: float,
    extra_channels: float,
    transition_power: float,
    path_power: float,
    transition_depth: float,
) -> pd.DataFrame:
    radii = {
        "solar_limb": R_SUN_M,
        "Mercury": 0.387098 * AU_M,
        "Earth": AU_M,
        "Saturn": 9.5826 * AU_M,
    }
    radius = np.asarray(list(radii.values()), dtype=float)
    gbar = G_SI * M_SUN_KG / np.square(radius)
    depth = G_SI * M_SUN_KG / (radius * C_M_S**2)
    response = path_diluted_potential_channel_acceleration(
        gbar,
        depth,
        np.ones_like(gbar),
        a0_m_s2=a0,
        transition_depth=transition_depth,
        transition_power=transition_power,
        extra_spatial_channels=extra_channels,
        path_power=path_power,
    )
    return pd.DataFrame(
        {
            "location": list(radii),
            "radius_m": radius,
            "baryonic_acceleration_m_s2": gbar,
            "potential_depth": depth,
            "potential_path_ratio": 1.0,
            "channel_exponent": response["channel_exponent"],
            "enhancement": response["enhancement"],
            "fractional_force_change": response["enhancement"] - 1.0,
        }
    )


def evaluate_candidate(
    galaxies: pd.DataFrame,
    clusters: list[dict],
    protocol: dict,
    *,
    extra_channels: float,
    transition_power: float,
    path_power: float,
    transition_depth: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a0 = float(protocol["equation"]["a0_m_s2"])
    galaxy = galaxy_predictions(
        galaxies,
        a0=a0,
        extra_channels=extra_channels,
        transition_power=transition_power,
        path_power=path_power,
        transition_depth=transition_depth,
    )
    galaxy_rmse = equal_system_rmse(galaxy, "galaxy", "velocity_residual_kms")
    cluster = predict_cluster_set(
        clusters,
        protocol,
        lambda item: path_diluted_potential_channel_acceleration(
            item["gbar_grid"],
            item["potential_depth_grid"],
            item["potential_path_ratio_grid"],
            a0_m_s2=a0,
            transition_depth=transition_depth,
            transition_power=transition_power,
            extra_spatial_channels=extra_channels,
            path_power=path_power,
        )["predicted_acceleration_m_s2"],
    )
    lens = cluster_scores(cluster)
    solar = solar_predictions(
        a0=a0,
        extra_channels=extra_channels,
        transition_power=transition_power,
        path_power=path_power,
        transition_depth=transition_depth,
    )
    row = {
        "extra_spatial_channels": extra_channels,
        "transition_power_n": transition_power,
        "path_power_q": path_power,
        "chi_t": transition_depth,
        "galaxy_equal_RMSE_km_s": galaxy_rmse,
        **{f"cluster_{key}": value for key, value in lens.items()},
        **{
            f"solar_{location}_fractional_force_change": float(
                solar.loc[solar.location.eq(location), "fractional_force_change"].iloc[0]
            )
            for location in solar.location
        },
    }
    return row, galaxy, cluster, solar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0684_formula_score":
        raise RuntimeError("P0684 protocol is not frozen")
    parent_path = ROOT / protocol["failure_parent"]
    parent = read_json(parent_path)
    expected = protocol["predeclared_integrity_gates"]
    if parent.get("status") != expected["P0683_status"]:
        raise RuntimeError("P0683 status changed")
    if bool(parent.get("candidate_advanced_to_3D_topology")) != bool(
        expected["P0683_candidate_advanced_to_3D_topology"]
    ):
        raise RuntimeError("P0683 advancement state changed")

    galaxy_path = ROOT / protocol["spent_galaxy_test"]["input"]
    galaxies = prepare_galaxies(galaxy_path)
    clusters = prepare_clusters(protocol)
    fixed_rar_galaxy = float(
        parent["comparators"]["galaxy_equal_RMSE_km_s"]["fixed_RAR_a0"]
    )
    fixed_rar_cluster = float(
        parent["comparators"]["cluster_log_RMS_dex"]["fixed_RAR_a0"][
            "all5_log_RMS_dex"
        ]
    )

    grid = protocol["diagnostic_sensitivity_grid"]
    rows = []
    total = int(grid["rows"])
    counter = 0
    for extra_channels in grid["extra_spatial_channels"]:
        for transition_power in grid["transition_power_n"]:
            for path_power in grid["path_power_q"]:
                for transition_depth in grid["chi_t"]:
                    counter += 1
                    if counter == 1 or counter % 50 == 0 or counter == total:
                        print(f"candidate={counter}/{total}", flush=True)
                    row, _, _, _ = evaluate_candidate(
                        galaxies,
                        clusters,
                        protocol,
                        extra_channels=float(extra_channels),
                        transition_power=float(transition_power),
                        path_power=float(path_power),
                        transition_depth=float(transition_depth),
                    )
                    rows.append(row)
    scores = pd.DataFrame(rows)
    equation = protocol["equation"]
    scores["is_primary_dimension_fixed_formula"] = (
        np.isclose(
            scores.extra_spatial_channels,
            float(equation["primary_extra_spatial_channels"]),
        )
        & np.isclose(
            scores.transition_power_n,
            float(equation["primary_transition_power_n"]),
        )
        & np.isclose(scores.path_power_q, float(equation["primary_path_power_q"]))
    )
    gates = protocol["predeclared_advancement_gates"]
    scores["galaxy_ratio_to_fixed_RAR"] = (
        scores.galaxy_equal_RMSE_km_s / fixed_rar_galaxy
    )
    scores["all5_gap_fraction_closed_from_fixed_RAR"] = 1.0 - (
        scores.cluster_all5_log_RMS_dex / fixed_rar_cluster
    )
    scores["galaxy_gate_pass"] = scores.galaxy_ratio_to_fixed_RAR <= float(
        gates["galaxy_RMSE_ratio_to_fixed_RAR_max"]
    )
    scores["solar_gate_pass"] = (
        (scores.solar_solar_limb_fractional_force_change <= float(gates["solar_limb_fractional_force_change_max"]))
        & (scores.solar_Earth_fractional_force_change <= float(gates["Earth_fractional_force_change_max"]))
        & (scores.solar_Saturn_fractional_force_change <= float(gates["Saturn_fractional_force_change_max"]))
    )
    scores["all5_cluster_gate_pass"] = scores.cluster_all5_log_RMS_dex <= float(
        gates["all5_cluster_log_RMS_dex_max"]
    )
    scores["reliable3_cluster_gate_pass"] = (
        scores.cluster_reliable3_log_RMS_dex
        <= float(gates["reliable3_cluster_log_RMS_dex_max"])
    )
    scores["gap_closed_gate_pass"] = (
        scores.all5_gap_fraction_closed_from_fixed_RAR
        >= float(gates["minimum_all5_gap_fraction_closed_from_fixed_RAR"])
    )
    eligible = scores[
        scores.is_primary_dimension_fixed_formula
        & scores.galaxy_gate_pass
        & scores.solar_gate_pass
    ].copy()
    if eligible.empty:
        raise RuntimeError("no primary row passed galaxy and Solar gates")
    minimum = float(eligible.cluster_all5_log_RMS_dex.min())
    selected_score = eligible[
        eligible.cluster_all5_log_RMS_dex <= minimum + 1.0e-6
    ].sort_values("chi_t", ascending=False).iloc[0]
    scores["selected_primary_row"] = (
        scores.is_primary_dimension_fixed_formula
        & np.isclose(scores.chi_t, float(selected_score.chi_t))
    )
    selected_row, selected_galaxy, selected_cluster, selected_solar = evaluate_candidate(
        galaxies,
        clusters,
        protocol,
        extra_channels=float(selected_score.extra_spatial_channels),
        transition_power=float(selected_score.transition_power_n),
        path_power=float(selected_score.path_power_q),
        transition_depth=float(selected_score.chi_t),
    )
    advancement = {
        "primary_dimension_fixed_formula": True,
        "galaxy_gate_pass": bool(selected_score.galaxy_gate_pass),
        "solar_force_proxy_gate_pass": bool(selected_score.solar_gate_pass),
        "all5_cluster_gate_pass": bool(selected_score.all5_cluster_gate_pass),
        "reliable3_cluster_gate_pass": bool(selected_score.reliable3_cluster_gate_pass),
        "gap_closed_gate_pass": bool(selected_score.gap_closed_gate_pass),
        "path_ratio_baryon_only": True,
        "no_per_object_gravity_parameters": True,
        "one_new_universal_parameter": True,
        "no_raw_image_root_score": True,
        "sealed_targets_untouched": True,
    }
    advancement["all_gates_pass"] = bool(all(advancement.values()))
    integrity = {
        "galaxy_systems": int(galaxies.galaxy.nunique()),
        "galaxy_points": len(galaxies),
        "cluster_systems": len(clusters),
        "primary_cluster_systems": int(sum(item["primary"] for item in clusters)),
        "reliable_cluster_systems": int(sum(item["reliable"] for item in clusters)),
        "grid_rows": len(scores),
        "primary_grid_rows": int(scores.is_primary_dimension_fixed_formula.sum()),
        "path_ratio_uses_only_baryonic_profile": True,
        "all_predictions_finite_and_positive": bool(
            np.all(np.isfinite(scores.select_dtypes(include=[np.number]).to_numpy()))
            and np.all(selected_galaxy.predicted_acceleration_m_s2 > 0.0)
            and np.all(selected_cluster.predicted_radial_deflection_arcsec > 0.0)
        ),
        "gravity_parameters_fit_per_object": 0,
        "new_raw_image_root_score_computed": False,
        "full_3D_field_solve_computed": False,
        "sealed_target_outcomes_opened": False,
    }
    integrity["all_gates_pass"] = bool(
        integrity["galaxy_systems"] == int(expected["galaxy_systems"])
        and integrity["galaxy_points"] == int(expected["galaxy_points"])
        and integrity["cluster_systems"] == int(expected["cluster_systems"])
        and integrity["primary_cluster_systems"] == int(expected["primary_cluster_systems"])
        and integrity["reliable_cluster_systems"] == int(expected["reliable_cluster_systems"])
        and integrity["grid_rows"] == int(expected["diagnostic_grid_rows"])
        and integrity["primary_grid_rows"] == int(expected["primary_grid_rows"])
        and integrity["path_ratio_uses_only_baryonic_profile"]
        and integrity["all_predictions_finite_and_positive"]
        and integrity["gravity_parameters_fit_per_object"] == 0
        and not integrity["new_raw_image_root_score_computed"]
        and not integrity["full_3D_field_solve_computed"]
        and not integrity["sealed_target_outcomes_opened"]
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    selected_galaxy.to_csv(output / protocol["outputs"]["galaxy_predictions"], index=False)
    selected_cluster.to_csv(output / protocol["outputs"]["cluster_predictions"], index=False)
    selected_solar.to_csv(output / protocol["outputs"]["solar"], index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    primary_surface = scores[
        np.isclose(scores.extra_spatial_channels, 3.0)
        & np.isclose(scores.transition_power_n, 2.0)
    ]
    cluster_pivot = primary_surface.pivot(
        index="path_power_q", columns="chi_t", values="cluster_all5_log_RMS_dex"
    )
    galaxy_pivot = primary_surface.pivot(
        index="path_power_q", columns="chi_t", values="galaxy_ratio_to_fixed_RAR"
    )
    image0 = axes[0, 0].imshow(cluster_pivot, aspect="auto", origin="lower")
    axes[0, 0].set_xticks(np.arange(len(cluster_pivot.columns)), [f"{v:.1e}" for v in cluster_pivot.columns], rotation=45)
    axes[0, 0].set_yticks(np.arange(len(cluster_pivot.index)), [f"{v:g}" for v in cluster_pivot.index])
    axes[0, 0].set(xlabel="chi_t", ylabel="path power q", title="Cluster log RMS (N=3,n=2)")
    figure.colorbar(image0, ax=axes[0, 0])
    image1 = axes[0, 1].imshow(galaxy_pivot, aspect="auto", origin="lower")
    axes[0, 1].set_xticks(np.arange(len(galaxy_pivot.columns)), [f"{v:.1e}" for v in galaxy_pivot.columns], rotation=45)
    axes[0, 1].set_yticks(np.arange(len(galaxy_pivot.index)), [f"{v:g}" for v in galaxy_pivot.index])
    axes[0, 1].set(xlabel="chi_t", ylabel="path power q", title="Galaxy RMSE / RAR (N=3,n=2)")
    figure.colorbar(image1, ax=axes[0, 1])
    for label, group in selected_cluster.groupby("label", sort=True):
        line = axes[1, 0].plot(group.radius_midpoint_kpc, group.predicted_radial_deflection_arcsec, label=label)[0]
        axes[1, 0].plot(group.radius_midpoint_kpc, group.target_total_radial_deflection_arcsec, linestyle="--", color=line.get_color(), alpha=0.7)
    axes[1, 0].set(xscale="log", xlabel="radius (kpc)", ylabel="reduced radial deflection (arcsec)", title="Selected primary: solid prediction, dashed target")
    axes[1, 0].legend(fontsize=8)
    eta = np.geomspace(1.0, 30.0, 300)
    for onset, style in [(0.25, ":"), (0.5, "--"), (1.0, "-")]:
        exponent = 1.0 + 3.0 * onset / np.sqrt(eta)
        axes[1, 1].plot(eta, exponent, linestyle=style, label=f"potential onset={onset:g}")
    axes[1, 1].set(xscale="log", xlabel="potential path ratio eta", ylabel="channel exponent p", title="Frozen inverse-square-root path dilution")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    passing_diagnostics = scores[
        scores.galaxy_gate_pass
        & scores.all5_cluster_gate_pass
        & scores.reliable3_cluster_gate_pass
        & scores.gap_closed_gate_pass
        & scores.solar_gate_pass
    ].copy()
    best_diagnostic = scores.sort_values(
        ["cluster_all5_log_RMS_dex", "galaxy_ratio_to_fixed_RAR"]
    ).iloc[0]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "pass" if integrity["all_gates_pass"] else "fail",
        "candidate_advanced_to_3D_topology": bool(
            integrity["all_gates_pass"] and advancement["all_gates_pass"]
        ),
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "input_hashes": {
            "failure_parent": sha256(parent_path),
            "galaxies": sha256(galaxy_path),
            "baryonic_profiles": sha256(ROOT / protocol["spent_cluster_test"]["baryonic_profiles"]),
            "target_profiles": sha256(ROOT / protocol["spent_cluster_test"]["target_profiles"]),
            "target_metrics": sha256(ROOT / protocol["spent_cluster_test"]["target_metrics"]),
        },
        "integrity_audit": integrity,
        "fixed_RAR_comparators": {
            "galaxy_equal_RMSE_km_s": fixed_rar_galaxy,
            "cluster_all5_log_RMS_dex": fixed_rar_cluster,
        },
        "selected_primary": selected_score.to_dict(),
        "selected_recomputed_metrics": selected_row,
        "advancement_gates": advancement,
        "diagnostic_rows_passing_numeric_gates": len(passing_diagnostics),
        "best_unrestricted_diagnostic": best_diagnostic.to_dict(),
        "selection": {
            "advance": bool(integrity["all_gates_pass"] and advancement["all_gates_pass"]),
            "next_action_if_pass": "Freeze the exact primary operator in registered 3D QUMOND and run spent RXJ2129 topology plus resolution and baryonic-map robustness.",
            "next_action_if_fail": "Use only predeclared response diagnostics to identify failure; do not promote a diagnostic row or open sealed outcomes."
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe({
        "integrity": integrity,
        "selected_primary": selected_score.to_dict(),
        "advancement": advancement,
        "diagnostic_rows_passing_numeric_gates": len(passing_diagnostics),
        "best_unrestricted_diagnostic": best_diagnostic.to_dict(),
    }), indent=2))


if __name__ == "__main__":
    main()

