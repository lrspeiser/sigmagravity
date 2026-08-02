#!/usr/bin/env python3
"""Run the frozen P0687 baryonic system path-coordinate screen."""

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
    C_M_S,
    KPC_M,
    cluster_scores,
    equal_system_rmse,
    json_safe,
    predict_cluster_set,
    prepare_clusters,
    prepare_galaxies,
    read_json,
    sha256,
)
from run_p0684_path_diluted_potential_channel_qumond import solar_predictions

from voidscreen.potential_channel_qumond import (
    path_diluted_potential_channel_acceleration,
    system_potential_path_coordinate,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0687_system_path_coordinate_qumond.json"
VARIANTS = ("global_system_primary", "local_P0685_control", "capped_local_diagnostic")


def used_path_ratio(local: np.ndarray, system: float, variant: str) -> np.ndarray:
    values = np.asarray(local, dtype=float)
    if variant == "global_system_primary":
        return np.full_like(values, float(system))
    if variant == "local_P0685_control":
        return values
    if variant == "capped_local_diagnostic":
        return np.minimum(values, float(system))
    raise ValueError(f"unknown P0687 variant: {variant}")


def prepare_galaxy_coordinates(galaxies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    rows = []
    for label, group in galaxies.groupby("galaxy", sort=True):
        values = group.copy()
        coordinate = system_potential_path_coordinate(
            values.potential_depth.to_numpy(float),
            values.radius_adjusted_kpc.to_numpy(float) * KPC_M,
            values.g_bar_m_s2.to_numpy(float),
            light_speed_m_s=C_M_S,
        )
        values["system_path_coordinate"] = coordinate
        frames.append(values)
        rows.append(
            {"object_class": "galaxy", "label": label, "system_path_coordinate": coordinate}
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


def cluster_system_coordinate(item: dict) -> float:
    return system_potential_path_coordinate(
        item["potential_depth_grid"],
        item["radius_grid"] * KPC_M,
        item["gbar_grid"],
        light_speed_m_s=C_M_S,
    )


def galaxy_predictions(
    galaxies: pd.DataFrame,
    protocol: dict,
    variant: str,
) -> pd.DataFrame:
    equation = protocol["equation"]
    path = np.empty(len(galaxies), dtype=float)
    for indices in galaxies.groupby("galaxy", sort=False).groups.values():
        positions = galaxies.index.get_indexer(indices)
        group = galaxies.loc[indices]
        system = float(group.system_path_coordinate.iloc[0])
        path[positions] = used_path_ratio(
            group.potential_path_ratio.to_numpy(float),
            system,
            variant,
        )
    response = path_diluted_potential_channel_acceleration(
        galaxies.g_bar_m_s2.to_numpy(float),
        galaxies.potential_depth.to_numpy(float),
        path,
        a0_m_s2=float(equation["a0_m_s2"]),
        transition_depth=float(equation["chi_t"]),
        transition_power=float(equation["transition_power_n"]),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        path_power=float(equation["path_power_q"]),
    )
    output = galaxies[
        [
            "galaxy",
            "radius_adjusted_kpc",
            "g_bar_m_s2",
            "potential_depth",
            "potential_path_ratio",
            "system_path_coordinate",
            "velocity_observed_adjusted_kms",
        ]
    ].copy()
    output["variant"] = variant
    output["used_path_ratio"] = path
    for name in (
        "potential_onset",
        "path_survival",
        "channel_exponent",
        "base_qumond_boost",
        "enhancement",
        "predicted_acceleration_m_s2",
    ):
        output[name] = response[name]
    output["velocity_predicted_kms"] = np.sqrt(
        response["predicted_acceleration_m_s2"]
        * output.radius_adjusted_kpc.to_numpy(float)
        * KPC_M
        / 1e6
    )
    output["velocity_residual_kms"] = (
        output.velocity_predicted_kms - output.velocity_observed_adjusted_kms
    )
    return output


def cluster_predictions(
    clusters: list[dict],
    protocol: dict,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    equation = protocol["equation"]

    def acceleration(item: dict) -> np.ndarray:
        system = cluster_system_coordinate(item)
        path = used_path_ratio(item["potential_path_ratio_grid"], system, variant)
        return path_diluted_potential_channel_acceleration(
            item["gbar_grid"],
            item["potential_depth_grid"],
            path,
            a0_m_s2=float(equation["a0_m_s2"]),
            transition_depth=float(equation["chi_t"]),
            transition_power=float(equation["transition_power_n"]),
            extra_spatial_channels=float(equation["extra_spatial_channels"]),
            path_power=float(equation["path_power_q"]),
        )["predicted_acceleration_m_s2"]

    predicted = predict_cluster_set(clusters, protocol, acceleration)
    slopes = {}
    for item in clusters:
        system = cluster_system_coordinate(item)
        path = used_path_ratio(item["potential_path_ratio_grid"], system, variant)
        response = path_diluted_potential_channel_acceleration(
            item["gbar_grid"],
            item["potential_depth_grid"],
            path,
            a0_m_s2=float(equation["a0_m_s2"]),
            transition_depth=float(equation["chi_t"]),
            transition_power=float(equation["transition_power_n"]),
            extra_spatial_channels=float(equation["extra_spatial_channels"]),
            path_power=float(equation["path_power_q"]),
        )
        slopes[item["label"]] = bool(np.all(np.diff(response["channel_exponent"]) <= 1e-12))
    predicted["variant"] = variant
    return predicted, slopes


def evaluate(
    galaxies: pd.DataFrame,
    clusters: list[dict],
    protocol: dict,
    variant: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    galaxy = galaxy_predictions(galaxies, protocol, variant)
    cluster, slopes = cluster_predictions(clusters, protocol, variant)
    scores = cluster_scores(cluster)
    equation = protocol["equation"]
    solar = solar_predictions(
        a0=float(equation["a0_m_s2"]),
        extra_channels=float(equation["extra_spatial_channels"]),
        transition_power=float(equation["transition_power_n"]),
        path_power=float(equation["path_power_q"]),
        transition_depth=float(equation["chi_t"]),
    )
    row = {
        "variant": variant,
        "galaxy_equal_RMSE_km_s": equal_system_rmse(
            galaxy,
            "galaxy",
            "velocity_residual_kms",
        ),
        **{f"cluster_{key}": value for key, value in scores.items()},
        "all_cluster_exponent_profiles_nonincreasing": bool(all(slopes.values())),
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
    if protocol.get("status") != "frozen_before_any_P0687_formula_score":
        raise RuntimeError("P0687 protocol is not frozen")
    failure_path = ROOT / protocol["failure_parent"]
    failure = read_json(failure_path)
    expected = protocol["predeclared_integrity_gates"]
    if failure.get("status") != expected["P0686_status"]:
        raise RuntimeError("P0686 status changed")
    if bool(failure.get("candidate_advanced_to_resolution_and_baryon_robustness")) != bool(
        expected["P0686_candidate_advanced_to_resolution_and_baryon_robustness"]
    ):
        raise RuntimeError("P0686 advancement state changed")

    galaxy_path = ROOT / protocol["spent_galaxy_test"]["input"]
    galaxies, galaxy_coordinates = prepare_galaxy_coordinates(prepare_galaxies(galaxy_path))
    clusters = prepare_clusters(protocol)
    cluster_coordinates = pd.DataFrame(
        [
            {
                "object_class": "cluster",
                "label": item["label"],
                "system_path_coordinate": cluster_system_coordinate(item),
            }
            for item in clusters
        ]
    )
    coordinates = pd.concat([galaxy_coordinates, cluster_coordinates], ignore_index=True)
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
    primary = scores.set_index("variant").loc[str(gates["primary_variant"])]
    primary_galaxy, primary_cluster, primary_solar = outputs[str(gates["primary_variant"])]
    advancement = {
        "primary_variant_fixed": str(primary.name) == str(gates["primary_variant"]),
        "galaxy_gate_pass": bool(primary.galaxy_gate_pass),
        "all5_cluster_gate_pass": bool(primary.all5_cluster_gate_pass),
        "reliable3_cluster_gate_pass": bool(primary.reliable3_cluster_gate_pass),
        "gap_closed_gate_pass": bool(primary.gap_closed_gate_pass),
        "solar_gate_pass": bool(primary.solar_gate_pass),
        "cluster_exponent_profiles_nonincreasing": bool(
            primary.all_cluster_exponent_profiles_nonincreasing
        ),
        "system_coordinate_baryon_only": True,
        "no_per_object_fitted_gravity_parameters": True,
        "no_raw_image_root_score": True,
        "sealed_targets_untouched": True,
    }
    advancement["all_gates_pass"] = bool(all(advancement.values()))
    finite = bool(
        np.all(np.isfinite(scores.select_dtypes(include=[np.number]).to_numpy()))
        and np.all(primary_galaxy.predicted_acceleration_m_s2 > 0.0)
        and np.all(primary_cluster.predicted_radial_deflection_arcsec > 0.0)
        and np.all(np.isfinite(coordinates.system_path_coordinate))
    )
    integrity = {
        "galaxy_systems": int(galaxies.galaxy.nunique()),
        "galaxy_points": len(galaxies),
        "cluster_systems": len(clusters),
        "primary_cluster_systems": int(sum(item["primary"] for item in clusters)),
        "reliable_cluster_systems": int(sum(item["reliable"] for item in clusters)),
        "formula_variants": len(scores),
        "system_path_coordinate_uses_only_baryonic_profiles": True,
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
        and integrity["system_path_coordinate_uses_only_baryonic_profiles"]
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
    coordinates.to_csv(output / protocol["outputs"]["system_coordinates"], index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(scores.variant, scores.galaxy_ratio_to_fixed_RAR)
    axes[0, 0].axhline(
        float(gates["galaxy_RMSE_ratio_to_fixed_RAR_max"]), color="black", linestyle="--"
    )
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].set(ylabel="RMSE / fixed RAR", title="Spent galaxies")
    axes[0, 1].bar(scores.variant, scores.cluster_all5_log_RMS_dex)
    axes[0, 1].axhline(float(gates["all5_cluster_log_RMS_dex_max"]), color="black", linestyle="--")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].set(ylabel="all-five log RMS (dex)", title="Spent clusters")
    cluster_only = coordinates[coordinates.object_class.eq("cluster")]
    axes[1, 0].bar(cluster_only.label, cluster_only.system_path_coordinate)
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set(ylabel="eta_sys", title="Baryonic system coordinate")
    for label, group in primary_cluster.groupby("label", sort=True):
        line = axes[1, 1].plot(
            group.radius_midpoint_kpc,
            group.predicted_radial_deflection_arcsec,
            label=label,
        )[0]
        axes[1, 1].plot(
            group.radius_midpoint_kpc,
            group.target_total_radial_deflection_arcsec,
            linestyle="--",
            color=line.get_color(),
            alpha=0.7,
        )
    axes[1, 1].set(
        xscale="log",
        xlabel="radius (kpc)",
        ylabel="reduced deflection (arcsec)",
        title="Primary: solid prediction, dashed target",
    )
    axes[1, 1].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": protocol["protocol_version"],
        "status": "pass" if integrity["all_gates_pass"] else "fail",
        "candidate_advanced_to_new_3D_topology_freeze": bool(
            integrity["all_gates_pass"] and advancement["all_gates_pass"]
        ),
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "input_hashes": {
            "failure_parent": sha256(failure_path),
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
        "system_coordinate_summary": {
            "galaxy_median": float(galaxy_coordinates.system_path_coordinate.median()),
            "galaxy_min": float(galaxy_coordinates.system_path_coordinate.min()),
            "galaxy_max": float(galaxy_coordinates.system_path_coordinate.max()),
            "clusters": dict(
                zip(
                    cluster_coordinates.label,
                    cluster_coordinates.system_path_coordinate,
                    strict=True,
                )
            ),
        },
        "selection": {
            "advance": bool(integrity["all_gates_pass"] and advancement["all_gates_pass"]),
            "next_action_if_pass": "Freeze the exact global-system operator in registered 3D QUMOND and repeat the spent RXJ2129 topology audit without retuning.",
            "next_action_if_fail": "Retain the topology-derived nonhollow constraint, diagnose the frozen variants, and do not promote a diagnostic row or open sealed outcomes.",
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )
    summary = f"""# P0687 system path-coordinate QUMOND

- Integrity: **{"PASS" if integrity["all_gates_pass"] else "FAIL"}**.
- Primary advances: **{"YES" if report["candidate_advanced_to_new_3D_topology_freeze"] else "NO"}**.
- Primary galaxy RMSE / fixed RAR: **{primary.galaxy_ratio_to_fixed_RAR:.4g}**.
- Primary all-five / reliable-three cluster log RMS: **{primary.cluster_all5_log_RMS_dex:.4g} / {primary.cluster_reliable3_log_RMS_dex:.4g} dex**.
- Primary fixed-RAR cluster gap closed: **{100 * primary.all5_gap_fraction_closed_from_fixed_RAR:.3g}%**.
- Primary cluster exponent profiles nonincreasing: **{bool(primary.all_cluster_exponent_profiles_nonincreasing)}**.
- Sealed P0633/P0640 targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
