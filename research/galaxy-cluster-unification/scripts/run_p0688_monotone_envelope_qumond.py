#!/usr/bin/env python3
"""Run the frozen P0688 inward monotone-majorant QUMOND screen."""

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

from run_p0683_potential_channel_qumond_reconnaissance import (
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

from voidscreen.potential_channel_qumond import (
    inward_monotone_majorant,
    path_diluted_potential_channel_acceleration,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0688_monotone_envelope_qumond.json"
VARIANTS = ("inward_monotone_majorant_primary", "local_P0685_control")


def ordered_response(
    radius,
    gbar,
    depth,
    local_path,
    protocol: dict,
    variant: str,
) -> dict[str, np.ndarray]:
    radius_values, gbar_values, depth_values, path_values = np.broadcast_arrays(
        np.asarray(radius, dtype=float),
        np.asarray(gbar, dtype=float),
        np.asarray(depth, dtype=float),
        np.asarray(local_path, dtype=float),
    )
    if radius_values.ndim != 1 or np.any(np.diff(np.sort(radius_values)) <= 0.0):
        raise ValueError("each P0688 profile must contain distinct one-dimensional radii")
    order = np.argsort(radius_values)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    equation = protocol["equation"]
    local = path_diluted_potential_channel_acceleration(
        gbar_values[order],
        depth_values[order],
        path_values[order],
        a0_m_s2=float(equation["a0_m_s2"]),
        transition_depth=float(equation["chi_t"]),
        transition_power=float(equation["transition_power_n"]),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        path_power=float(equation["path_power_q"]),
    )
    local_exponent = local["channel_exponent"]
    if variant == "inward_monotone_majorant_primary":
        used_exponent = inward_monotone_majorant(local_exponent)
    elif variant == "local_P0685_control":
        used_exponent = local_exponent.copy()
    else:
        raise ValueError(f"unknown P0688 variant: {variant}")
    enhancement = np.power(local["base_qumond_boost"], used_exponent)
    predicted = gbar_values[order] * enhancement
    return {
        "local_channel_exponent": local_exponent[inverse],
        "channel_exponent": used_exponent[inverse],
        "base_qumond_boost": local["base_qumond_boost"][inverse],
        "enhancement": enhancement[inverse],
        "predicted_acceleration_m_s2": predicted[inverse],
        "profile_nonincreasing": bool(np.all(np.diff(used_exponent) <= 1e-12)),
        "pointwise_majorizes_local": bool(np.all(used_exponent + 1e-12 >= local_exponent)),
    }


def galaxy_predictions(
    galaxies: pd.DataFrame,
    protocol: dict,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    frames = []
    checks = {}
    for label, group in galaxies.groupby("galaxy", sort=True):
        values = group.sort_values("radius_adjusted_kpc").copy()
        response = ordered_response(
            values.radius_adjusted_kpc.to_numpy(float),
            values.g_bar_m_s2.to_numpy(float),
            values.potential_depth.to_numpy(float),
            values.potential_path_ratio.to_numpy(float),
            protocol,
            variant,
        )
        values["variant"] = variant
        for name in (
            "local_channel_exponent",
            "channel_exponent",
            "base_qumond_boost",
            "enhancement",
            "predicted_acceleration_m_s2",
        ):
            values[name] = response[name]
        values["velocity_predicted_kms"] = np.sqrt(
            values.predicted_acceleration_m_s2.to_numpy(float)
            * values.radius_adjusted_kpc.to_numpy(float)
            * KPC_M
            / 1e6
        )
        values["velocity_residual_kms"] = (
            values.velocity_predicted_kms - values.velocity_observed_adjusted_kms
        )
        checks[label] = bool(
            response["profile_nonincreasing"] and response["pointwise_majorizes_local"]
        )
        frames.append(values)
    return pd.concat(frames, ignore_index=True), checks


def cluster_predictions(
    clusters: list[dict],
    protocol: dict,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    response_by_label = {}
    checks = {}
    for item in clusters:
        response = ordered_response(
            item["radius_grid"],
            item["gbar_grid"],
            item["potential_depth_grid"],
            item["potential_path_ratio_grid"],
            protocol,
            variant,
        )
        response_by_label[item["label"]] = response
        checks[item["label"]] = bool(
            response["profile_nonincreasing"] and response["pointwise_majorizes_local"]
        )
    predicted = predict_cluster_set(
        clusters,
        protocol,
        lambda item: response_by_label[item["label"]]["predicted_acceleration_m_s2"],
    )
    predicted["variant"] = variant
    return predicted, checks


def solar_predictions(protocol: dict, variant: str) -> tuple[pd.DataFrame, bool]:
    radii = {
        "solar_limb": R_SUN_M,
        "Mercury": 0.387098 * AU_M,
        "Earth": AU_M,
        "Saturn": 9.5826 * AU_M,
    }
    radius = np.asarray(list(radii.values()), dtype=float)
    gbar = G_SI * M_SUN_KG / np.square(radius)
    depth = G_SI * M_SUN_KG / (radius * C_M_S**2)
    response = ordered_response(
        radius,
        gbar,
        depth,
        np.ones_like(radius),
        protocol,
        variant,
    )
    output = pd.DataFrame(
        {
            "location": list(radii),
            "radius_m": radius,
            "baryonic_acceleration_m_s2": gbar,
            "potential_depth": depth,
            "local_channel_exponent": response["local_channel_exponent"],
            "channel_exponent": response["channel_exponent"],
            "enhancement": response["enhancement"],
            "fractional_force_change": response["enhancement"] - 1.0,
        }
    )
    passed = bool(response["profile_nonincreasing"] and response["pointwise_majorizes_local"])
    return output, passed


def evaluate(
    galaxies: pd.DataFrame,
    clusters: list[dict],
    protocol: dict,
    variant: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    galaxy, galaxy_checks = galaxy_predictions(galaxies, protocol, variant)
    cluster, cluster_checks = cluster_predictions(clusters, protocol, variant)
    solar, solar_check = solar_predictions(protocol, variant)
    lens = cluster_scores(cluster)
    row = {
        "variant": variant,
        "galaxy_equal_RMSE_km_s": equal_system_rmse(
            galaxy,
            "galaxy",
            "velocity_residual_kms",
        ),
        **{f"cluster_{key}": value for key, value in lens.items()},
        "all_profiles_nonincreasing_and_majorizing": bool(
            all(galaxy_checks.values()) and all(cluster_checks.values()) and solar_check
        ),
        **{
            f"solar_{location}_fractional_force_change": float(
                solar.loc[solar.location.eq(location), "fractional_force_change"].iloc[0]
            )
            for location in solar.location
        },
    }
    return row, galaxy, cluster, solar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0688_formula_score":
        raise RuntimeError("P0688 protocol is not frozen")
    parent_path = ROOT / protocol["failure_parent"]
    parent = read_json(parent_path)
    expected = protocol["predeclared_integrity_gates"]
    if parent.get("status") != expected["P0687_status"]:
        raise RuntimeError("P0687 status changed")
    if bool(parent.get("candidate_advanced_to_new_3D_topology_freeze")) != bool(
        expected["P0687_candidate_advanced_to_new_3D_topology_freeze"]
    ):
        raise RuntimeError("P0687 advancement state changed")

    galaxy_path = ROOT / protocol["spent_galaxy_test"]["input"]
    galaxies = prepare_galaxies(galaxy_path)
    clusters = prepare_clusters(protocol)
    comparator_path = ROOT / protocol["radial_comparator_report"]
    comparator = read_json(comparator_path)
    fixed_rar_galaxy = float(comparator["fixed_RAR_comparators"]["galaxy_equal_RMSE_km_s"])
    fixed_rar_cluster = float(comparator["fixed_RAR_comparators"]["cluster_all5_log_RMS_dex"])

    rows = []
    outputs = {}
    for variant in VARIANTS:
        print(f"evaluating {variant}", flush=True)
        row, galaxy, cluster, solar = evaluate(galaxies, clusters, protocol, variant)
        rows.append(row)
        outputs[variant] = (galaxy, cluster, solar)
    scores = pd.DataFrame(rows)
    scores["galaxy_ratio_to_fixed_RAR"] = scores.galaxy_equal_RMSE_km_s / fixed_rar_galaxy
    scores["all5_gap_fraction_closed_from_fixed_RAR"] = 1.0 - (
        scores.cluster_all5_log_RMS_dex / fixed_rar_cluster
    )
    gates = protocol["predeclared_advancement_gates"]
    scores["galaxy_gate_pass"] = scores.galaxy_ratio_to_fixed_RAR <= float(
        gates["galaxy_RMSE_ratio_to_fixed_RAR_max"]
    )
    scores["all5_cluster_gate_pass"] = scores.cluster_all5_log_RMS_dex <= float(
        gates["all5_cluster_log_RMS_dex_max"]
    )
    scores["reliable3_cluster_gate_pass"] = scores.cluster_reliable3_log_RMS_dex <= float(
        gates["reliable3_cluster_log_RMS_dex_max"]
    )
    scores["gap_closed_gate_pass"] = scores.all5_gap_fraction_closed_from_fixed_RAR >= float(
        gates["minimum_all5_gap_fraction_closed_from_fixed_RAR"]
    )
    scores["solar_gate_pass"] = (
        (
            scores.solar_solar_limb_fractional_force_change
            <= float(gates["solar_limb_fractional_force_change_max"])
        )
        & (
            scores.solar_Earth_fractional_force_change
            <= float(gates["Earth_fractional_force_change_max"])
        )
        & (
            scores.solar_Saturn_fractional_force_change
            <= float(gates["Saturn_fractional_force_change_max"])
        )
    )
    primary_name = str(gates["primary_variant"])
    primary = scores.set_index("variant").loc[primary_name]
    primary_galaxy, primary_cluster, primary_solar = outputs[primary_name]
    advancement = {
        "primary_variant_fixed": str(primary.name) == primary_name,
        "galaxy_gate_pass": bool(primary.galaxy_gate_pass),
        "all5_cluster_gate_pass": bool(primary.all5_cluster_gate_pass),
        "reliable3_cluster_gate_pass": bool(primary.reliable3_cluster_gate_pass),
        "gap_closed_gate_pass": bool(primary.gap_closed_gate_pass),
        "solar_gate_pass": bool(primary.solar_gate_pass),
        "all_profiles_nonincreasing": bool(primary.all_profiles_nonincreasing_and_majorizing),
        "envelope_pointwise_majorizes_local": bool(
            primary.all_profiles_nonincreasing_and_majorizing
        ),
        "no_new_constants": int(protocol["equation"]["new_constants"]) == 0,
        "no_per_object_fitted_gravity_parameters": True,
        "no_raw_image_root_score": True,
        "sealed_targets_untouched": True,
    }
    advancement["all_gates_pass"] = bool(all(advancement.values()))
    finite = bool(
        np.all(np.isfinite(scores.select_dtypes(include=[np.number]).to_numpy()))
        and np.all(primary_galaxy.predicted_acceleration_m_s2 > 0.0)
        and np.all(primary_cluster.predicted_radial_deflection_arcsec > 0.0)
    )
    integrity = {
        "galaxy_systems": int(galaxies.galaxy.nunique()),
        "galaxy_points": len(galaxies),
        "cluster_systems": len(clusters),
        "primary_cluster_systems": int(sum(item["primary"] for item in clusters)),
        "reliable_cluster_systems": int(sum(item["reliable"] for item in clusters)),
        "formula_variants": len(scores),
        "all_profiles_sorted_before_envelope": True,
        "all_predictions_finite_and_positive": finite,
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
        and integrity["formula_variants"] == int(expected["formula_variants"])
        and integrity["all_profiles_sorted_before_envelope"]
        and integrity["all_predictions_finite_and_positive"]
        and integrity["gravity_parameters_fit_per_object"] == 0
        and not integrity["new_raw_image_root_score_computed"]
        and not integrity["full_3D_field_solve_computed"]
        and not integrity["sealed_target_outcomes_opened"]
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    primary_galaxy.to_csv(output / protocol["outputs"]["galaxy_predictions"], index=False)
    primary_cluster.to_csv(output / protocol["outputs"]["cluster_predictions"], index=False)
    primary_solar.to_csv(output / protocol["outputs"]["solar"], index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(scores.variant, scores.galaxy_ratio_to_fixed_RAR)
    axes[0, 0].axhline(
        float(gates["galaxy_RMSE_ratio_to_fixed_RAR_max"]), color="black", linestyle="--"
    )
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].set(ylabel="RMSE / fixed RAR", title="Spent galaxies")
    axes[0, 1].bar(scores.variant, scores.cluster_all5_log_RMS_dex)
    axes[0, 1].axhline(float(gates["all5_cluster_log_RMS_dex_max"]), color="black", linestyle="--")
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].set(ylabel="all-five log RMS (dex)", title="Spent clusters")
    for label, group in primary_cluster.groupby("label", sort=True):
        line = axes[1, 0].plot(
            group.radius_midpoint_kpc,
            group.predicted_radial_deflection_arcsec,
            label=label,
        )[0]
        axes[1, 0].plot(
            group.radius_midpoint_kpc,
            group.target_total_radial_deflection_arcsec,
            linestyle="--",
            color=line.get_color(),
            alpha=0.7,
        )
    axes[1, 0].set(
        xscale="log",
        xlabel="radius (kpc)",
        ylabel="reduced deflection (arcsec)",
        title="Envelope: solid prediction, dashed target",
    )
    axes[1, 0].legend(fontsize=8)
    example = primary_galaxy[primary_galaxy.galaxy.eq(primary_galaxy.galaxy.iloc[0])]
    axes[1, 1].plot(
        example.radius_adjusted_kpc,
        example.local_channel_exponent,
        marker="o",
        label="local",
    )
    axes[1, 1].plot(
        example.radius_adjusted_kpc,
        example.channel_exponent,
        marker="s",
        label="monotone envelope",
    )
    axes[1, 1].set(xlabel="radius (kpc)", ylabel="channel exponent", title="Example profile repair")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": protocol["protocol_version"],
        "status": "pass" if integrity["all_gates_pass"] else "fail",
        "candidate_advanced_to_3D_potential_shell_freeze": bool(
            integrity["all_gates_pass"] and advancement["all_gates_pass"]
        ),
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "input_hashes": {
            "failure_parent": sha256(parent_path),
            "radial_comparator_report": sha256(comparator_path),
            "galaxies": sha256(galaxy_path),
            "baryonic_profiles": sha256(ROOT / protocol["spent_cluster_test"]["baryonic_profiles"]),
            "target_profiles": sha256(ROOT / protocol["spent_cluster_test"]["target_profiles"]),
        },
        "integrity_audit": integrity,
        "fixed_RAR_comparators": {
            "galaxy_equal_RMSE_km_s": fixed_rar_galaxy,
            "cluster_all5_log_RMS_dex": fixed_rar_cluster,
        },
        "variant_scores": scores.to_dict(orient="records"),
        "primary_advancement_gates": advancement,
        "selection": {
            "advance": bool(integrity["all_gates_pass"] and advancement["all_gates_pass"]),
            "next_action_if_pass": "Freeze a 3D potential-depth-shell implementation and raw RXJ2129 topology audit with no retuning.",
            "next_action_if_fail": "Retire further local-path topology patches and return to a different field variable or source operator.",
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )
    summary = f"""# P0688 monotone-envelope QUMOND

- Integrity: **{"PASS" if integrity["all_gates_pass"] else "FAIL"}**.
- Primary advances: **{"YES" if report["candidate_advanced_to_3D_potential_shell_freeze"] else "NO"}**.
- Primary galaxy RMSE / fixed RAR: **{primary.galaxy_ratio_to_fixed_RAR:.4g}**.
- Primary all-five / reliable-three cluster log RMS: **{primary.cluster_all5_log_RMS_dex:.4g} / {primary.cluster_reliable3_log_RMS_dex:.4g} dex**.
- Primary fixed-RAR cluster gap closed: **{100 * primary.all5_gap_fraction_closed_from_fixed_RAR:.3g}%**.
- All primary profiles nonincreasing and pointwise-majorizing: **{bool(primary.all_profiles_nonincreasing_and_majorizing)}**.
- Sealed P0633/P0640 targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
