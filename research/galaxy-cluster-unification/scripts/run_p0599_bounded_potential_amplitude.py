#!/usr/bin/env python3
"""Test bounded potential amplitude carriers on SPARC and absolute CLASH data."""

from __future__ import annotations

import hashlib
import itertools
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

from run_arc_invariant_absolute_lensing import prepare_clusters, prepare_galaxies  # noqa: E402
from run_p0593_diffusion_cross_domain import acceleration_velocity  # noqa: E402
from voidscreen.conservative_diffusion import (  # noqa: E402
    low_acceleration_activation,
    radial_shape_activation,
    redistributed_cumulative_mass,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.unified import G_SI, M_SUN_KG, rar_acceleration  # noqa: E402


def balanced_folds(names, folds: int) -> dict[str, int]:
    ordered = sorted(set(map(str, names)), key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest())
    return {name: index % int(folds) for index, name in enumerate(ordered)}


def per_system_mse(residual: np.ndarray, systems: np.ndarray) -> pd.Series:
    return pd.Series(np.square(residual), index=np.asarray(systems, dtype=str)).groupby(level=0).mean()


def spatial_base(
    frame: pd.DataFrame,
    *,
    system_column: str,
    radius_column: str,
    gbar_column: str,
    protocol: dict,
) -> dict[str, np.ndarray]:
    config = protocol["fixed_spatial_parameters"]
    local = frame[gbar_column].to_numpy(float)
    diffuse = np.empty(len(frame), dtype=float)
    shape = np.empty(len(frame), dtype=float)
    screen = np.empty(len(frame), dtype=float)
    for _, raw_indices in frame.groupby(system_column, sort=False).indices.items():
        raw_indices = np.asarray(raw_indices, dtype=int)
        order = np.argsort(frame.loc[raw_indices, radius_column].to_numpy(float), kind="stable")
        indices = raw_indices[order]
        radius = frame.loc[indices, radius_column].to_numpy(float)
        gbar = frame.loc[indices, gbar_column].to_numpy(float)
        mass = np.maximum.accumulate(gbar * np.square(radius * KPC_M) / (G_SI * M_SUN_KG))
        r80 = float(frame.loc[indices[0], "force_equivalent_r80_kpc"])
        concentration = float(
            frame.loc[indices[0], "force_equivalent_concentration_r50_over_r80"]
        )
        source_g = float(np.interp(r80, radius, gbar))
        source_screen = low_acceleration_activation(
            source_g,
            a0_m_s2=config["a0_m_s2"],
            power=config["source_acceleration_gate_power"],
        )
        source_shape = radial_shape_activation(
            concentration,
            midpoint=config["shape_midpoint"],
            width=config["shape_width"],
        )
        fraction = config["route_fraction_max"] * source_shape * source_screen
        routed, _ = redistributed_cumulative_mass(
            radius,
            mass,
            r80=r80,
            position_scale=1.0,
            width_over_r80=config["q_R80"],
            bins=1024,
        )
        effective_mass = (1.0 - fraction) * mass + fraction * routed
        diffuse[indices] = G_SI * M_SUN_KG * effective_mass / np.square(radius * KPC_M)
        shape[indices] = source_shape
        screen[indices] = source_screen
    return {"local": local, "shape_diffusion": diffuse, "shape": shape, "screen": screen}


def carrier_weight(name: str, shape: np.ndarray, path: np.ndarray) -> np.ndarray:
    bounded_path = path / (1.0 + path)
    if name == "potential":
        return np.ones_like(shape)
    if name == "potential_shape":
        return shape
    if name == "potential_inverse_shape":
        return 1.0 - shape
    if name == "potential_path":
        return bounded_path
    if name == "potential_shape_path":
        return shape * bounded_path
    raise ValueError(f"unknown carrier {name}")


def candidate_id(spatial: str, carrier: str, amplitude: float, threshold: float, power: float) -> str:
    return f"{spatial}__{carrier}__A{amplitude:g}__ct{threshold:.0e}__p{power:g}"


def domain_metrics(
    galaxy: pd.DataFrame,
    galaxy_velocity: np.ndarray,
    cluster: pd.DataFrame,
    cluster_acceleration: np.ndarray,
) -> dict[str, float]:
    galaxy_residual = galaxy_velocity - galaxy.velocity_observed_adjusted_km_s.to_numpy(float)
    cluster_residual = np.log10(cluster_acceleration) - cluster.log_gtot.to_numpy(float)
    galaxy_mse = per_system_mse(galaxy_residual, galaxy.galaxy.to_numpy(str))
    cluster_mse = per_system_mse(cluster_residual, cluster.system.to_numpy(str))
    return {
        "galaxy_equal_RMSE_km_s": float(np.sqrt(galaxy_mse.mean())),
        "galaxy_pooled_RMSE_km_s": float(np.sqrt(np.mean(np.square(galaxy_residual)))),
        "cluster_equal_RMSE_dex": float(np.sqrt(cluster_mse.mean())),
        "cluster_pooled_RMSE_dex": float(np.sqrt(np.mean(np.square(cluster_residual)))),
        "cluster_mean_residual_dex": float(np.mean(cluster_residual)),
        "cluster_median_observed_over_predicted": float(
            np.median(cluster.observed_g_m_s2.to_numpy(float) / cluster_acceleration)
        ),
    }


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0599_bounded_potential_amplitude_protocol.json").read_text(encoding="utf-8")
    )
    arc_protocol = json.loads((ROOT / protocol["data"]["arc_protocol"]).read_text(encoding="utf-8"))
    a0 = protocol["fixed_spatial_parameters"]["a0_m_s2"]
    galaxy_all, _ = prepare_galaxies(arc_protocol, a0)
    galaxy = galaxy_all[galaxy_all.split == "outer_holdout"].copy().reset_index(drop=True)
    cluster, _ = prepare_clusters(arc_protocol)
    cluster = cluster.copy().reset_index(drop=True)
    if (
        galaxy.galaxy.nunique() != protocol["data"]["galaxies"]
        or len(galaxy) != protocol["data"]["galaxy_outer_points"]
        or cluster.system.nunique() != protocol["data"]["clusters"]
        or len(cluster) != protocol["data"]["cluster_points"]
    ):
        raise RuntimeError("P0599 data coverage changed")
    fold_count = protocol["validation"]["folds"]
    galaxy_folds = balanced_folds(galaxy.galaxy.unique(), fold_count)
    cluster_folds = balanced_folds(cluster.system.unique(), fold_count)
    galaxy["cv_fold"] = galaxy.galaxy.map(galaxy_folds)
    cluster["cv_fold"] = cluster.system.map(cluster_folds)
    galaxy_base = spatial_base(
        galaxy_all.reset_index(drop=True),
        system_column="galaxy",
        radius_column="radius_adjusted_kpc",
        gbar_column="g_bar_m_s2",
        protocol=protocol,
    )
    outer_mask = galaxy_all.reset_index(drop=True).split.to_numpy(str) == "outer_holdout"
    galaxy_base = {key: values[outer_mask] for key, values in galaxy_base.items()}
    cluster_base = spatial_base(
        cluster,
        system_column="system",
        radius_column="radius_kpc",
        gbar_column="gbar_m_s2",
        protocol=protocol,
    )
    reference_g_acceleration = rar_acceleration(galaxy_base["local"], a0)
    reference_g_velocity = acceleration_velocity(
        galaxy.radius_adjusted_kpc.to_numpy(float), reference_g_acceleration
    )
    reference_c_acceleration = rar_acceleration(cluster_base["local"], a0)
    spatial_g_acceleration = rar_acceleration(galaxy_base["shape_diffusion"], a0)
    spatial_g_velocity = acceleration_velocity(
        galaxy.radius_adjusted_kpc.to_numpy(float), spatial_g_acceleration
    )
    spatial_c_acceleration = rar_acceleration(cluster_base["shape_diffusion"], a0)
    references = {
        "fixed_RAR": domain_metrics(
            galaxy, reference_g_velocity, cluster, reference_c_acceleration
        ),
        "P0598_spatial_RAR": domain_metrics(
            galaxy, spatial_g_velocity, cluster, spatial_c_acceleration
        ),
    }
    reference_galaxy_mse = per_system_mse(
        reference_g_velocity - galaxy.velocity_observed_adjusted_km_s.to_numpy(float),
        galaxy.galaxy.to_numpy(str),
    )

    specs = []
    for spatial, carrier, amplitude, threshold, power in itertools.product(
        protocol["grid"]["spatial_mode"],
        protocol["grid"]["carrier"],
        protocol["grid"]["amplitude_A"],
        protocol["grid"]["potential_threshold_chi"],
        protocol["grid"]["potential_power"],
    ):
        specs.append(
            {
                "candidate_id": candidate_id(spatial, carrier, amplitude, threshold, power),
                "spatial_mode": spatial,
                "carrier": carrier,
                "amplitude_A": float(amplitude),
                "potential_threshold_chi": float(threshold),
                "potential_power": float(power),
            }
        )
    if len(specs) != protocol["grid"]["candidate_count"]:
        raise RuntimeError("P0599 candidate count changed")
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    candidate_rows = []
    fold_rows = []
    for spec in specs:
        galaxy_potential = 1.0 / (
            1.0
            + np.power(
                spec["potential_threshold_chi"]
                / np.maximum(galaxy.potential_depth.to_numpy(float), np.finfo(float).tiny),
                spec["potential_power"],
            )
        )
        cluster_potential = 1.0 / (
            1.0
            + np.power(
                spec["potential_threshold_chi"]
                / np.maximum(cluster.potential_depth.to_numpy(float), np.finfo(float).tiny),
                spec["potential_power"],
            )
        )
        galaxy_weight = carrier_weight(
            spec["carrier"], galaxy_base["shape"], galaxy.potential_path_ratio.to_numpy(float)
        )
        cluster_weight = carrier_weight(
            spec["carrier"], cluster_base["shape"], cluster.potential_path_ratio.to_numpy(float)
        )
        galaxy_scalar = rar_acceleration(galaxy_base[spec["spatial_mode"]], a0)
        cluster_scalar = rar_acceleration(cluster_base[spec["spatial_mode"]], a0)
        galaxy_acceleration = galaxy_scalar * (
            1.0
            + spec["amplitude_A"]
            * galaxy_base["screen"]
            * galaxy_potential
            * galaxy_weight
        )
        cluster_acceleration = cluster_scalar * (
            1.0
            + spec["amplitude_A"]
            * cluster_base["screen"]
            * cluster_potential
            * cluster_weight
        )
        galaxy_velocity = acceleration_velocity(
            galaxy.radius_adjusted_kpc.to_numpy(float), galaxy_acceleration
        )
        predictions[spec["candidate_id"]] = (galaxy_velocity, cluster_acceleration)
        metrics = domain_metrics(galaxy, galaxy_velocity, cluster, cluster_acceleration)
        candidate_rows.append({**spec, **metrics})
        galaxy_residual = galaxy_velocity - galaxy.velocity_observed_adjusted_km_s.to_numpy(float)
        cluster_residual = np.log10(cluster_acceleration) - cluster.log_gtot.to_numpy(float)
        galaxy_mse = per_system_mse(galaxy_residual, galaxy.galaxy.to_numpy(str))
        cluster_mse = per_system_mse(cluster_residual, cluster.system.to_numpy(str))
        for fold in range(fold_count):
            train_g = [name for name, value in galaxy_folds.items() if value != fold]
            test_g = [name for name, value in galaxy_folds.items() if value == fold]
            train_c = [name for name, value in cluster_folds.items() if value != fold]
            test_c = [name for name, value in cluster_folds.items() if value == fold]
            train_g_rmse = float(np.sqrt(galaxy_mse.loc[train_g].mean()))
            test_g_rmse = float(np.sqrt(galaxy_mse.loc[test_g].mean()))
            train_c_rmse = float(np.sqrt(cluster_mse.loc[train_c].mean()))
            test_c_rmse = float(np.sqrt(cluster_mse.loc[test_c].mean()))
            rar_train = float(np.sqrt(reference_galaxy_mse.loc[train_g].mean()))
            rar_test = float(np.sqrt(reference_galaxy_mse.loc[test_g].mean()))
            fold_rows.append(
                {
                    **spec,
                    "fold": fold,
                    "training_galaxies": len(train_g),
                    "test_galaxies": len(test_g),
                    "training_clusters": len(train_c),
                    "test_clusters": len(test_c),
                    "training_galaxy_equal_RMSE_km_s": train_g_rmse,
                    "test_galaxy_equal_RMSE_km_s": test_g_rmse,
                    "training_galaxy_RMSE_ratio_to_fixed_RAR": train_g_rmse / rar_train,
                    "test_galaxy_RMSE_ratio_to_fixed_RAR": test_g_rmse / rar_test,
                    "training_cluster_equal_RMSE_dex": train_c_rmse,
                    "test_cluster_equal_RMSE_dex": test_c_rmse,
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    fold_scores = pd.DataFrame(fold_rows)
    galaxy_oof = np.empty(len(galaxy), dtype=float)
    cluster_oof = np.empty(len(cluster), dtype=float)
    selection_rows = []
    complexity = {
        "potential": 0,
        "potential_shape": 1,
        "potential_inverse_shape": 1,
        "potential_path": 1,
        "potential_shape_path": 2,
    }
    for fold in range(fold_count):
        block = fold_scores[
            (fold_scores.fold == fold)
            & (fold_scores.training_galaxy_RMSE_ratio_to_fixed_RAR <= 1.02)
        ].copy()
        if block.empty:
            raise RuntimeError(f"P0599 fold {fold} has no galaxy-preserving candidate")
        block["carrier_complexity"] = block.carrier.map(complexity)
        selected = block.sort_values(
            [
                "training_cluster_equal_RMSE_dex",
                "carrier_complexity",
                "amplitude_A",
                "potential_power",
                "potential_threshold_chi",
                "candidate_id",
            ],
            kind="stable",
        ).iloc[0]
        g_prediction, c_prediction = predictions[str(selected.candidate_id)]
        g_mask = galaxy.cv_fold.to_numpy(int) == fold
        c_mask = cluster.cv_fold.to_numpy(int) == fold
        galaxy_oof[g_mask] = g_prediction[g_mask]
        cluster_oof[c_mask] = c_prediction[c_mask]
        selection_rows.append(selected.to_dict())
    selections = pd.DataFrame(selection_rows)
    oof_metrics = domain_metrics(galaxy, galaxy_oof, cluster, cluster_oof)
    galaxy["prediction_km_s"] = galaxy_oof
    galaxy["fixed_RAR_km_s"] = reference_g_velocity
    galaxy["selected_candidate_id"] = galaxy.cv_fold.map(
        selections.set_index("fold").candidate_id.to_dict()
    )
    cluster["prediction_m_s2"] = cluster_oof
    cluster["fixed_RAR_m_s2"] = reference_c_acceleration
    cluster["residual_dex"] = np.log10(cluster_oof) - cluster.log_gtot
    cluster["fixed_RAR_residual_dex"] = np.log10(reference_c_acceleration) - cluster.log_gtot
    cluster["selected_candidate_id"] = cluster.cv_fold.map(
        selections.set_index("fold").candidate_id.to_dict()
    )

    eligible = candidates[
        candidates.galaxy_equal_RMSE_km_s
        <= 1.02 * references["fixed_RAR"]["galaxy_equal_RMSE_km_s"]
    ].copy()
    impact_rows = []
    for parameter in (
        "spatial_mode",
        "carrier",
        "amplitude_A",
        "potential_threshold_chi",
        "potential_power",
    ):
        grouped = eligible.groupby(parameter).cluster_equal_RMSE_dex.median().sort_values()
        impact_rows.append(
            {
                "parameter": parameter,
                "best_level": str(grouped.index[0]),
                "worst_level": str(grouped.index[-1]),
                "median_cluster_RMSE_span_dex": float(grouped.iloc[-1] - grouped.iloc[0]),
                "best_median_cluster_RMSE_dex": float(grouped.iloc[0]),
                "worst_median_cluster_RMSE_dex": float(grouped.iloc[-1]),
            }
        )
    impacts = pd.DataFrame(impact_rows).sort_values(
        "median_cluster_RMSE_span_dex", ascending=False
    )
    cluster_reference_mse = per_system_mse(
        np.log10(reference_c_acceleration) - cluster.log_gtot.to_numpy(float),
        cluster.system.to_numpy(str),
    )
    cluster_candidate_mse = per_system_mse(
        np.log10(cluster_oof) - cluster.log_gtot.to_numpy(float),
        cluster.system.to_numpy(str),
    )
    clusters_improved = int(np.sum(cluster_candidate_mse < cluster_reference_mse))
    cluster_gain = 1.0 - (
        oof_metrics["cluster_equal_RMSE_dex"]
        / references["fixed_RAR"]["cluster_equal_RMSE_dex"]
    )
    fold_cluster_improved = 0
    for fold in range(fold_count):
        held = [name for name, value in cluster_folds.items() if value == fold]
        candidate_rmse = float(np.sqrt(cluster_candidate_mse.loc[held].mean()))
        reference_rmse = float(np.sqrt(cluster_reference_mse.loc[held].mean()))
        fold_cluster_improved += int(candidate_rmse < reference_rmse)
    solar_g_r80 = 254.55253269745893
    solar_screen = low_acceleration_activation(
        solar_g_r80,
        a0_m_s2=a0,
        power=protocol["fixed_spatial_parameters"]["source_acceleration_gate_power"],
    )
    maximum_solar_fraction = (
        max(protocol["grid"]["amplitude_A"]) * solar_screen
    )
    gates_cfg = protocol["interpretation_gates"]
    gates = {
        "galaxy_preservation_pass": bool(
            oof_metrics["galaxy_equal_RMSE_km_s"]
            / references["fixed_RAR"]["galaxy_equal_RMSE_km_s"]
            <= gates_cfg["oof_galaxy_RMSE_ratio_to_fixed_RAR_max"]
        ),
        "cluster_improvement_pass": bool(
            cluster_gain
            >= gates_cfg["oof_cluster_RMSE_improvement_fraction_vs_fixed_RAR_min"]
        ),
        "cluster_fold_count_pass": bool(
            fold_cluster_improved >= gates_cfg["cluster_folds_improved_min"]
        ),
        "cluster_system_count_pass": bool(
            clusters_improved >= gates_cfg["clusters_improved_min"]
        ),
        "Solar_amplitude_screen_pass": bool(
            maximum_solar_fraction <= gates_cfg["Solar_amplitude_fraction_max"]
        ),
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    fold_scores.to_csv(output / protocol["outputs"]["candidate_fold_scores"], index=False)
    selections.to_csv(output / protocol["outputs"]["fold_selections"], index=False)
    galaxy.to_csv(output / protocol["outputs"]["galaxy_oof_predictions"], index=False)
    cluster.to_csv(output / protocol["outputs"]["cluster_oof_predictions"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].bar(
        ["fixed RAR", "screened spatial", "bounded potential OOF"],
        [
            references["fixed_RAR"]["cluster_equal_RMSE_dex"],
            references["P0598_spatial_RAR"]["cluster_equal_RMSE_dex"],
            oof_metrics["cluster_equal_RMSE_dex"],
        ],
    )
    axes[0].set(ylabel="equal-cluster RMSE (dex)", title="absolute CLASH amplitude")
    display = impacts.sort_values("median_cluster_RMSE_span_dex")
    axes[1].barh(display.parameter, display.median_cluster_RMSE_span_dex)
    axes[1].set(xlabel="median CLASH RMSE span (dex)", title="impact under galaxy-preservation gate")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    report = {
        "report_version": "P0599-BOUNDED-POTENTIAL-AMPLITUDE-RESULTS-0.1.0",
        "status": "complete_whole_object_cross_validation",
        "coverage": {
            "candidates": len(candidates),
            "galaxies": galaxy.galaxy.nunique(),
            "galaxy_outer_points": len(galaxy),
            "clusters": cluster.system.nunique(),
            "cluster_points": len(cluster),
            "folds": fold_count,
        },
        "measured_support": {
            "galaxy_potential_depth_range": [float(galaxy.potential_depth.min()), float(galaxy.potential_depth.max())],
            "cluster_potential_depth_range": [float(cluster.potential_depth.min()), float(cluster.potential_depth.max())],
            "support_gap_ratio": float(cluster.potential_depth.min() / galaxy.potential_depth.max()),
        },
        "references": references,
        "oof_metrics": oof_metrics,
        "cluster_improvement_fraction_vs_fixed_RAR": cluster_gain,
        "galaxy_RMSE_ratio_to_fixed_RAR": float(
            oof_metrics["galaxy_equal_RMSE_km_s"]
            / references["fixed_RAR"]["galaxy_equal_RMSE_km_s"]
        ),
        "clusters_improved_vs_fixed_RAR": clusters_improved,
        "cluster_folds_improved_vs_fixed_RAR": fold_cluster_improved,
        "fold_selections": selections.to_dict("records"),
        "unique_selected_candidates": int(selections.candidate_id.nunique()),
        "parameter_impacts": impacts.to_dict("records"),
        "Solar": {
            "source_g_R80_m_s2": solar_g_r80,
            "source_screen_activation": solar_screen,
            "maximum_tested_amplitude_fraction_upper_bound": maximum_solar_fraction,
        },
        "gates": gates,
        "all_interpretation_gates_pass": bool(all(gates.values())),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0599 bounded potential amplitude\n\n"
        f"Whole-object cross-validation changed equal-cluster absolute RMSE from "
        f"{references['fixed_RAR']['cluster_equal_RMSE_dex']:.3f} to "
        f"{oof_metrics['cluster_equal_RMSE_dex']:.3f} dex ({100.0 * cluster_gain:+.1f}%) while galaxy "
        f"equal-weighted RMSE was {oof_metrics['galaxy_equal_RMSE_km_s']:.3f} km/s, a ratio of "
        f"{report['galaxy_RMSE_ratio_to_fixed_RAR']:.3f} to fixed RAR. It improved "
        f"{clusters_improved}/20 clusters and {fold_cluster_improved}/5 cluster folds. The strongest "
        f"parameter impact was {impacts.iloc[0].parameter} with a median CLASH span of "
        f"{impacts.iloc[0].median_cluster_RMSE_span_dex:.3f} dex. SPARC and CLASH potential support "
        f"is separated by a factor {report['measured_support']['support_gap_ratio']:.2f}, so intermediate "
        "BCG data are required before interpreting the threshold as universal physics.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
