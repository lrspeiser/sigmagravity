#!/usr/bin/env python3
"""Run the frozen P0683 potential-channel QUMOND reconnaissance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.arc_invariants import spherical_profile_invariants  # noqa: E402
from voidscreen.phenomenology import (  # noqa: E402
    fixed_rar_enhancement,
    simple_mond_enhancement,
)
from voidscreen.potential_channel_qumond import (  # noqa: E402
    potential_channel_acceleration,
)
from voidscreen.raw_lensing import (  # noqa: E402
    KPC_M,
    RAD_TO_ARCSEC,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0683_potential_channel_qumond_reconnaissance.json"
G_SI = 6.67430e-11
C_M_S = 299_792_458.0
M_SUN_KG = 1.98847e30
R_SUN_M = 6.957e8
AU_M = 149_597_870_700.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def equal_system_rmse(frame: pd.DataFrame, group: str, residual: str) -> float:
    per_system = frame.groupby(group, sort=True)[residual].apply(
        lambda values: float(np.sqrt(np.mean(np.square(values.to_numpy(float)))))
    )
    return float(np.sqrt(np.mean(np.square(per_system.to_numpy(float)))))


def prepare_galaxies(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    frame["potential_depth"] = np.nan
    frame["potential_path_ratio"] = np.nan
    for _, group in frame.groupby("galaxy", sort=True):
        ordered = group.sort_values("radius_adjusted_kpc")
        invariants = spherical_profile_invariants(
            ordered.radius_adjusted_kpc.to_numpy(float),
            ordered.g_bar_m_s2.to_numpy(float),
        )
        frame.loc[ordered.index, "potential_depth"] = invariants["potential_depth"]
        frame.loc[ordered.index, "potential_path_ratio"] = invariants[
            "potential_path_ratio"
        ]
    if frame[["potential_depth", "potential_path_ratio"]].isna().any().any():
        raise RuntimeError("galaxy potential construction left missing rows")
    return frame


def galaxy_predictions(
    galaxies: pd.DataFrame,
    *,
    a0: float,
    endpoint: float,
    transition_power: float,
    transition_depth: float,
) -> pd.DataFrame:
    response = potential_channel_acceleration(
        galaxies.g_bar_m_s2.to_numpy(float),
        galaxies.potential_depth.to_numpy(float),
        a0_m_s2=a0,
        transition_depth=transition_depth,
        transition_power=transition_power,
        endpoint_exponent=endpoint,
    )
    predicted_velocity = np.sqrt(
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
            "velocity_observed_adjusted_kms",
        ]
    ].copy()
    output["channel_exponent"] = response["channel_exponent"]
    output["base_qumond_boost"] = response["base_qumond_boost"]
    output["enhancement"] = response["enhancement"]
    output["predicted_acceleration_m_s2"] = response[
        "predicted_acceleration_m_s2"
    ]
    output["velocity_predicted_kms"] = predicted_velocity
    output["velocity_residual_kms"] = (
        predicted_velocity - output.velocity_observed_adjusted_kms
    )
    return output


def comparator_galaxy_score(galaxies: pd.DataFrame, acceleration: np.ndarray) -> float:
    velocity = np.sqrt(
        np.asarray(acceleration, dtype=float)
        * galaxies.radius_adjusted_kpc.to_numpy(float)
        * KPC_M
        / 1.0e6
    )
    work = pd.DataFrame(
        {
            "galaxy": galaxies.galaxy,
            "residual": velocity - galaxies.velocity_observed_adjusted_kms.to_numpy(float),
        }
    )
    return equal_system_rmse(work, "galaxy", "residual")


def prepare_clusters(protocol: dict) -> list[dict]:
    settings = protocol["spent_cluster_test"]
    tian = pd.read_csv(
        ROOT / settings["baryonic_profiles"],
        sep=r"\s+",
        names=["label", "radius_kpc", "log_gbar", "log_gobs", "err_gbar", "err_gobs"],
    )
    targets = pd.read_csv(ROOT / settings["target_profiles"])
    metrics = pd.read_csv(ROOT / settings["target_metrics"]).set_index("label")
    grid_points = int(settings["acceleration_grid_points"])
    maximum = float(settings["maximum_radius_kpc"])
    prepared = []
    for label in settings["systems"]:
        anchors = tian[tian.label.eq(label)].sort_values("radius_kpc")
        target = targets[targets.label.eq(label)].sort_values("radial_bin").copy()
        if len(anchors) < 3 or len(target) != 12:
            raise RuntimeError(f"cluster profile coverage changed for {label}")
        radius_grid = np.geomspace(0.1, maximum, grid_points)
        gbar_grid = loglog_interpolate_with_tails(
            radius_grid,
            anchors.radius_kpc.to_numpy(float),
            np.power(10.0, anchors.log_gbar.to_numpy(float)),
            outer_slope=-2.0,
        )
        invariants = spherical_profile_invariants(radius_grid, gbar_grid)
        potential = invariants["potential_depth"]
        row = metrics.loc[label]
        target["target_total_radial_deflection_arcsec"] = (
            target.mean_baryon_radial_deflection_arcsec
            + target.mean_halo_radial_deflection_arcsec
        )
        prepared.append(
            {
                "label": label,
                "radius_grid": radius_grid,
                "gbar_grid": gbar_grid,
                "potential_depth_grid": potential,
                "potential_path_ratio_grid": invariants["potential_path_ratio"],
                "target": target,
                "distance_ratio": float(row.reference_distance_ratio),
                "primary": not bool(row.parameter_at_boundary),
                "reliable": bool(row.reliable_predictor_target),
            }
        )
    return prepared


def predict_cluster_set(
    clusters: list[dict],
    protocol: dict,
    acceleration_builder,
) -> pd.DataFrame:
    settings = protocol["spent_cluster_test"]
    rows = []
    for cluster in clusters:
        acceleration_grid = np.asarray(acceleration_builder(cluster), dtype=float)
        if np.any(~np.isfinite(acceleration_grid)) or np.any(acceleration_grid <= 0.0):
            raise RuntimeError(f"invalid acceleration for {cluster['label']}")

        def lookup(radius):
            return np.exp(
                np.interp(
                    np.log(radius),
                    np.log(cluster["radius_grid"]),
                    np.log(acceleration_grid),
                )
            )

        target = cluster["target"]
        physical = spherical_deflection_radians(
            target.radius_midpoint_kpc.to_numpy(float),
            lookup,
            maximum_radius_kpc=float(settings["maximum_radius_kpc"]),
            integration_points=int(settings["line_of_sight_integration_points"]),
        )
        predicted = physical * cluster["distance_ratio"] * RAD_TO_ARCSEC
        observed = target.target_total_radial_deflection_arcsec.to_numpy(float)
        for index, (prediction, target_value) in enumerate(
            zip(predicted, observed, strict=True)
        ):
            rows.append(
                {
                    "label": cluster["label"],
                    "radial_bin": int(target.iloc[index].radial_bin),
                    "radius_midpoint_kpc": float(target.iloc[index].radius_midpoint_kpc),
                    "target_total_radial_deflection_arcsec": target_value,
                    "predicted_radial_deflection_arcsec": float(prediction),
                    "log10_prediction_to_target": float(
                        np.log10(prediction / target_value)
                    ),
                    "primary_system": cluster["primary"],
                    "reliable_system": cluster["reliable"],
                }
            )
    return pd.DataFrame(rows)


def cluster_scores(predictions: pd.DataFrame) -> dict:
    primary = predictions[predictions.primary_system.astype(bool)]
    reliable = predictions[predictions.reliable_system.astype(bool)]
    return {
        "all5_log_RMS_dex": equal_system_rmse(
            primary, "label", "log10_prediction_to_target"
        ),
        "reliable3_log_RMS_dex": equal_system_rmse(
            reliable, "label", "log10_prediction_to_target"
        ),
        "all6_stress_log_RMS_dex": equal_system_rmse(
            predictions, "label", "log10_prediction_to_target"
        ),
    }


def solar_predictions(
    *,
    a0: float,
    endpoint: float,
    transition_power: float,
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
    response = potential_channel_acceleration(
        gbar,
        depth,
        a0_m_s2=a0,
        transition_depth=transition_depth,
        transition_power=transition_power,
        endpoint_exponent=endpoint,
    )
    return pd.DataFrame(
        {
            "location": list(radii),
            "radius_m": radius,
            "baryonic_acceleration_m_s2": gbar,
            "potential_depth": depth,
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
    endpoint: float,
    transition_power: float,
    transition_depth: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a0 = float(protocol["equation"]["a0_m_s2"])
    galaxy = galaxy_predictions(
        galaxies,
        a0=a0,
        endpoint=endpoint,
        transition_power=transition_power,
        transition_depth=transition_depth,
    )
    galaxy_rmse = equal_system_rmse(galaxy, "galaxy", "velocity_residual_kms")
    cluster = predict_cluster_set(
        clusters,
        protocol,
        lambda item: potential_channel_acceleration(
            item["gbar_grid"],
            item["potential_depth_grid"],
            a0_m_s2=a0,
            transition_depth=transition_depth,
            transition_power=transition_power,
            endpoint_exponent=endpoint,
        )["predicted_acceleration_m_s2"],
    )
    lens_scores = cluster_scores(cluster)
    solar = solar_predictions(
        a0=a0,
        endpoint=endpoint,
        transition_power=transition_power,
        transition_depth=transition_depth,
    )
    row = {
        "p_infinity": endpoint,
        "transition_power_n": transition_power,
        "chi_t": transition_depth,
        "galaxy_equal_RMSE_km_s": galaxy_rmse,
        **{f"cluster_{key}": value for key, value in lens_scores.items()},
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
    if protocol.get("status") != "frozen_before_any_P0683_formula_score":
        raise RuntimeError("P0683 protocol is not frozen")
    parent_path = ROOT / protocol["parent"]
    parent = read_json(parent_path)
    if parent.get("status") != protocol["predeclared_integrity_gates"]["P0682_status"]:
        raise RuntimeError("P0682 parent status changed")

    galaxy_path = ROOT / protocol["spent_galaxy_test"]["input"]
    galaxies = prepare_galaxies(galaxy_path)
    clusters = prepare_clusters(protocol)
    a0 = float(protocol["equation"]["a0_m_s2"])
    gbar = galaxies.g_bar_m_s2.to_numpy(float)
    comparator_galaxy = {
        "baryons_Newton": comparator_galaxy_score(galaxies, gbar),
        "fixed_RAR_a0": comparator_galaxy_score(
            galaxies, gbar * fixed_rar_enhancement(gbar, a0)
        ),
        "simple_AQUAL_algebraic_a0": comparator_galaxy_score(
            galaxies, gbar * simple_mond_enhancement(gbar, a0)
        ),
    }
    comparator_cluster_predictions = {
        "baryons_Newton": predict_cluster_set(
            clusters, protocol, lambda item: item["gbar_grid"]
        ),
        "fixed_RAR_a0": predict_cluster_set(
            clusters,
            protocol,
            lambda item: item["gbar_grid"]
            * fixed_rar_enhancement(item["gbar_grid"], a0),
        ),
        "simple_AQUAL_algebraic_a0": predict_cluster_set(
            clusters,
            protocol,
            lambda item: item["gbar_grid"]
            * simple_mond_enhancement(item["gbar_grid"], a0),
        ),
    }
    comparator_cluster = {
        name: cluster_scores(frame)
        for name, frame in comparator_cluster_predictions.items()
    }

    grid = protocol["diagnostic_sensitivity_grid"]
    rows = []
    for endpoint in grid["p_infinity"]:
        for transition_power in grid["transition_power_n"]:
            for transition_depth in grid["chi_t"]:
                print(
                    f"p={endpoint:g} n={transition_power:g} chi_t={transition_depth:.3g}",
                    flush=True,
                )
                row, _, _, _ = evaluate_candidate(
                    galaxies,
                    clusters,
                    protocol,
                    endpoint=float(endpoint),
                    transition_power=float(transition_power),
                    transition_depth=float(transition_depth),
                )
                rows.append(row)
    scores = pd.DataFrame(rows)
    scores["is_primary_dimension_fixed_formula"] = (
        np.isclose(scores.p_infinity, float(protocol["equation"]["primary_p_infinity"]))
        & np.isclose(
            scores.transition_power_n,
            float(protocol["equation"]["primary_transition_power_n"]),
        )
    )
    gates = protocol["predeclared_advancement_gates"]
    scores["galaxy_ratio_to_fixed_RAR"] = (
        scores.galaxy_equal_RMSE_km_s / comparator_galaxy["fixed_RAR_a0"]
    )
    scores["all5_gap_fraction_closed_from_fixed_RAR"] = 1.0 - (
        scores.cluster_all5_log_RMS_dex
        / comparator_cluster["fixed_RAR_a0"]["all5_log_RMS_dex"]
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
    primary_eligible = scores[
        scores.is_primary_dimension_fixed_formula
        & scores.galaxy_gate_pass
        & scores.solar_gate_pass
    ].copy()
    if primary_eligible.empty:
        raise RuntimeError("no dimension-fixed primary row passed galaxy and Solar gates")
    minimum_cluster = float(primary_eligible.cluster_all5_log_RMS_dex.min())
    tied = primary_eligible[
        primary_eligible.cluster_all5_log_RMS_dex <= minimum_cluster + 1.0e-6
    ].sort_values("chi_t", ascending=False)
    selected_score = tied.iloc[0]
    scores["selected_primary_row"] = (
        scores.is_primary_dimension_fixed_formula
        & np.isclose(scores.chi_t, float(selected_score.chi_t))
    )
    selected_row, selected_galaxy, selected_cluster, selected_solar = evaluate_candidate(
        galaxies,
        clusters,
        protocol,
        endpoint=float(selected_score.p_infinity),
        transition_power=float(selected_score.transition_power_n),
        transition_depth=float(selected_score.chi_t),
    )
    advancement = {
        "primary_dimension_fixed_formula": True,
        "galaxy_gate_pass": bool(selected_score.galaxy_gate_pass),
        "solar_force_proxy_gate_pass": bool(selected_score.solar_gate_pass),
        "all5_cluster_gate_pass": bool(selected_score.all5_cluster_gate_pass),
        "reliable3_cluster_gate_pass": bool(selected_score.reliable3_cluster_gate_pass),
        "gap_closed_gate_pass": bool(selected_score.gap_closed_gate_pass),
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
    expected = protocol["predeclared_integrity_gates"]
    integrity["all_gates_pass"] = bool(
        integrity["galaxy_systems"] == int(expected["galaxy_systems"])
        and integrity["galaxy_points"] == int(expected["galaxy_points"])
        and integrity["cluster_systems"] == int(expected["cluster_systems"])
        and integrity["primary_cluster_systems"] == int(expected["primary_cluster_systems"])
        and integrity["reliable_cluster_systems"] == int(expected["reliable_cluster_systems"])
        and integrity["grid_rows"] == int(expected["diagnostic_grid_rows"])
        and integrity["primary_grid_rows"] == int(expected["primary_grid_rows"])
        and integrity["all_predictions_finite_and_positive"]
        and integrity["gravity_parameters_fit_per_object"] == 0
        and not integrity["new_raw_image_root_score_computed"]
        and not integrity["full_3D_field_solve_computed"]
        and not integrity["sealed_target_outcomes_opened"]
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    selected_galaxy.to_csv(
        output / protocol["outputs"]["galaxy_predictions"], index=False
    )
    selected_cluster.to_csv(
        output / protocol["outputs"]["cluster_predictions"], index=False
    )
    selected_solar.to_csv(output / protocol["outputs"]["solar"], index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    n2 = scores[np.isclose(scores.transition_power_n, 2.0)].copy()
    pivot_cluster = n2.pivot(index="p_infinity", columns="chi_t", values="cluster_all5_log_RMS_dex")
    pivot_galaxy = n2.pivot(index="p_infinity", columns="chi_t", values="galaxy_ratio_to_fixed_RAR")
    image0 = axes[0, 0].imshow(pivot_cluster.to_numpy(), aspect="auto", origin="lower")
    axes[0, 0].set_xticks(np.arange(len(pivot_cluster.columns)), [f"{v:.1e}" for v in pivot_cluster.columns], rotation=45)
    axes[0, 0].set_yticks(np.arange(len(pivot_cluster.index)), [f"{v:g}" for v in pivot_cluster.index])
    axes[0, 0].set(xlabel="chi_t", ylabel="p_infinity", title="Cluster target log RMS (n=2)")
    figure.colorbar(image0, ax=axes[0, 0])
    image1 = axes[0, 1].imshow(pivot_galaxy.to_numpy(), aspect="auto", origin="lower")
    axes[0, 1].set_xticks(np.arange(len(pivot_galaxy.columns)), [f"{v:.1e}" for v in pivot_galaxy.columns], rotation=45)
    axes[0, 1].set_yticks(np.arange(len(pivot_galaxy.index)), [f"{v:g}" for v in pivot_galaxy.index])
    axes[0, 1].set(xlabel="chi_t", ylabel="p_infinity", title="Galaxy RMSE / fixed RAR (n=2)")
    figure.colorbar(image1, ax=axes[0, 1])
    for label, group in selected_cluster.groupby("label", sort=True):
        line = axes[1, 0].plot(
            group.radius_midpoint_kpc,
            group.predicted_radial_deflection_arcsec,
            label=label,
        )[0]
        axes[1, 0].plot(
            group.radius_midpoint_kpc,
            group.target_total_radial_deflection_arcsec,
            linestyle="--",
            alpha=0.7,
            color=line.get_color(),
        )
    axes[1, 0].set(xscale="log", xlabel="radius (kpc)", ylabel="reduced radial deflection (arcsec)", title="Selected primary: solid prediction, dashed target")
    axes[1, 0].legend(fontsize=8)
    depth = np.geomspace(1.0e-10, 3.0e-5, 400)
    exponent = 1.0 + 3.0 * np.square(depth / float(selected_score.chi_t)) / (1.0 + np.square(depth / float(selected_score.chi_t)))
    axes[1, 1].plot(depth, exponent, color="black")
    axes[1, 1].axvspan(float(galaxies.potential_depth.min()), float(galaxies.potential_depth.max()), color="C0", alpha=0.2, label="spent galaxies")
    cluster_depth = np.concatenate(
        [
            np.exp(
                np.interp(
                    np.log(item["target"].radius_midpoint_kpc.to_numpy(float)),
                    np.log(item["radius_grid"]),
                    np.log(item["potential_depth_grid"]),
                )
            )
            for item in clusters
        ]
    )
    axes[1, 1].axvspan(float(np.min(cluster_depth)), float(np.max(cluster_depth)), color="C3", alpha=0.2, label="spent cluster annuli")
    axes[1, 1].set(xscale="log", xlabel="baryonic potential depth |Phi_b|/c^2", ylabel="channel exponent p", title=f"Selected chi_t={selected_score.chi_t:.2g}")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

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
            "parent": sha256(parent_path),
            "galaxies": sha256(galaxy_path),
            "baryonic_profiles": sha256(ROOT / protocol["spent_cluster_test"]["baryonic_profiles"]),
            "target_profiles": sha256(ROOT / protocol["spent_cluster_test"]["target_profiles"]),
            "target_metrics": sha256(ROOT / protocol["spent_cluster_test"]["target_metrics"]),
        },
        "integrity_audit": integrity,
        "comparators": {
            "galaxy_equal_RMSE_km_s": comparator_galaxy,
            "cluster_log_RMS_dex": comparator_cluster,
        },
        "selected_primary": selected_score.to_dict(),
        "selected_recomputed_metrics": selected_row,
        "advancement_gates": advancement,
        "best_unrestricted_diagnostic": best_diagnostic.to_dict(),
        "selection": {
            "advance": bool(integrity["all_gates_pass"] and advancement["all_gates_pass"]),
            "next_action_if_pass": "Freeze the selected one-setting operator, implement it in the registered 3D QUMOND solver, and run spent RXJ2129 topology plus resolution/baryon sensitivities.",
            "next_action_if_fail": "Use the frozen endpoint/power response map to identify which dimensional premise failed; do not retune thresholds or open sealed outcomes.",
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
        "best_unrestricted_diagnostic": best_diagnostic.to_dict(),
    }), indent=2))


if __name__ == "__main__":
    main()
