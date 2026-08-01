#!/usr/bin/env python3
"""Replay one simplified radial-shape diffusion law across all current domains."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons, lens_source_map  # noqa: E402
from run_p0568_baryon_only_tensor_forward import build_contexts  # noqa: E402
from run_p0580_conservative_return_sparc import galaxy_force_profile, score  # noqa: E402
from run_p0592_diffusive_propagator_transfer import normalize_inside  # noqa: E402
from run_p0593_diffusion_cross_domain import acceleration_velocity  # noqa: E402
from run_p0595_diffusion_boundary_cv import galaxy_fold  # noqa: E402
from voidscreen.conservative_diffusion import (  # noqa: E402
    gaussian_tail_upper_bound,
    radial_shape_activation,
    redistributed_cumulative_mass,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.unified import G_SI, M_SUN_KG, rar_acceleration  # noqa: E402


AU_M = 149_597_870_700.0


def radial_radii(positions: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weight = np.asarray(weights, dtype=float)
    weight /= weight.sum()
    xy = np.asarray(positions, dtype=float)
    center = np.sum(weight[:, None] * xy, axis=0)
    radius = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1])
    order = np.argsort(radius)
    cumulative = np.cumsum(weight[order])
    values = []
    for fraction in (0.5, 0.8):
        index = min(np.searchsorted(cumulative, fraction), len(order) - 1)
        values.append(float(radius[order[index]]))
    return values[0], values[1]


def galaxy_equal_rmse(frame: pd.DataFrame, column: str) -> float:
    residual = frame[column].to_numpy(float) - frame.velocity_observed_adjusted_km_s.to_numpy(float)
    mse = pd.Series(np.square(residual), index=frame.galaxy.to_numpy(str)).groupby(level=0).mean()
    return float(np.sqrt(mse.mean()))


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0597_simple_shape_law_cross_domain_protocol.json").read_text(encoding="utf-8")
    )
    parameters = protocol["parameters"]
    galaxy_protocol = json.loads(
        (ROOT / protocol["source_protocols"]["galaxy"]).read_text(encoding="utf-8")
    )
    galaxy_cfg = galaxy_protocol["galaxy_test"]
    raw = pd.read_csv(ROOT / galaxy_cfg["points"])
    points = raw[(raw.model == galaxy_cfg["model"]) & (raw.scenario == galaxy_cfg["scenario"])].copy()
    points["source_point_index"] = points.index
    outer = points[points.split == galaxy_cfg["split"]].copy().reset_index(drop=True)
    outer["cv_fold"] = outer.galaxy.map(lambda value: galaxy_fold(value, 5))
    profiles = {galaxy: galaxy_force_profile(block) for galaxy, block in points.groupby("galaxy", sort=False)}
    prediction = np.empty(len(outer), dtype=float)
    galaxy_shape = {}
    for galaxy, indices in outer.groupby("galaxy", sort=False).indices.items():
        profile = profiles[galaxy]
        shape = radial_shape_activation(
            profile["concentration_R50_over_R80"],
            midpoint=parameters["shape_midpoint"],
            width=parameters["shape_width"],
        )
        fraction = parameters["route_fraction_max"] * shape
        galaxy_shape[galaxy] = {
            "C_R50_over_R80": profile["concentration_R50_over_R80"],
            "shape_activation": shape,
            "effective_route_fraction": fraction,
            "R80_kpc": profile["R80_kpc"],
        }
        routed, _ = redistributed_cumulative_mass(
            profile["radius_kpc"],
            profile["mass_solar"],
            r80=profile["R80_kpc"],
            position_scale=1.0,
            width_over_r80=parameters["q_R80"],
            bins=galaxy_protocol["constants"]["radial_bins"],
        )
        effective_mass = (1.0 - fraction) * profile["mass_solar"] + fraction * routed
        radius = profile["radius_kpc"]
        g_eff = G_SI * M_SUN_KG * effective_mass / np.square(radius * KPC_M)
        full_velocity = acceleration_velocity(
            radius, rar_acceleration(g_eff, galaxy_protocol["constants"]["a0_m_s2"])
        )
        frame = profile["frame"]
        mask = frame.split.to_numpy(str) == galaxy_cfg["split"]
        velocity_by_source = dict(zip(frame.loc[mask, "source_point_index"], full_velocity[mask]))
        target = np.asarray(indices, dtype=int)
        prediction[target] = [
            velocity_by_source[source]
            for source in outer.loc[target, "source_point_index"].to_numpy(int)
        ]
    outer["simple_shape_prediction_km_s"] = prediction
    outer["fixed_RAR_km_s"] = outer.velocity_RAR_same_nuisance_km_s
    for key in ("C_R50_over_R80", "shape_activation", "effective_route_fraction", "R80_kpc"):
        outer[key] = outer.galaxy.map(lambda name: galaxy_shape[name][key])
    galaxy_rows = []
    for galaxy, block in outer.groupby("galaxy"):
        obs = block.velocity_observed_adjusted_km_s.to_numpy(float)
        simple_mse = float(np.mean(np.square(block.simple_shape_prediction_km_s.to_numpy(float) - obs)))
        rar_mse = float(np.mean(np.square(block.fixed_RAR_km_s.to_numpy(float) - obs)))
        galaxy_rows.append(
            {
                "galaxy": galaxy,
                "cv_fold": int(block.cv_fold.iloc[0]),
                **galaxy_shape[galaxy],
                "simple_shape_RMSE_km_s": np.sqrt(simple_mse),
                "fixed_RAR_RMSE_km_s": np.sqrt(rar_mse),
                "delta_MSE_km_s2": rar_mse - simple_mse,
            }
        )
    galaxy_scores = pd.DataFrame(galaxy_rows)
    galaxy_metrics = {
        "simple_shape": score(outer, prediction),
        "fixed_RAR": score(outer, outer.fixed_RAR_km_s.to_numpy(float)),
    }
    equal_improvement = 1.0 - (
        galaxy_metrics["simple_shape"]["outer_equal_galaxy_RMSE_km_s"]
        / galaxy_metrics["fixed_RAR"]["outer_equal_galaxy_RMSE_km_s"]
    )
    fold_rows = []
    for fold in range(5):
        block = outer[outer.cv_fold == fold]
        simple = galaxy_equal_rmse(block, "simple_shape_prediction_km_s")
        reference = galaxy_equal_rmse(block, "fixed_RAR_km_s")
        fold_rows.append(
            {"fold": fold, "galaxies": block.galaxy.nunique(), "simple_shape_equal_RMSE_km_s": simple, "fixed_RAR_equal_RMSE_km_s": reference, "improvement_fraction": 1.0 - simple / reference}
        )
    galaxy_folds = pd.DataFrame(fold_rows)
    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    delta = galaxy_scores.delta_MSE_km_s2.to_numpy(float)
    samples = rng.integers(0, len(delta), size=(protocol["bootstrap"]["draws"], len(delta)))
    galaxy_bootstrap = float(np.mean(np.mean(delta[samples], axis=1) > 0.0))

    cluster_protocol = json.loads(
        (ROOT / protocol["source_protocols"]["cluster"]).read_text(encoding="utf-8")
    )
    p0568 = json.loads((ROOT / cluster_protocol["data"]["p0568_protocol"]).read_text(encoding="utf-8"))
    p0567 = json.loads((ROOT / cluster_protocol["data"]["p0567_protocol"]).read_text(encoding="utf-8"))
    contexts = build_contexts(p0568, p0567)
    development = set(cluster_protocol["data"]["development_systems"])
    spacing = float(cluster_protocol["preprocessing"]["grid_spacing_kpc"])
    cluster_rows, uncertainty_rows, glafic_rows = [], [], []
    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        r50, r80 = radial_radii(context.data.positions, context.data.weights)
        shape = radial_shape_activation(
            r50 / r80, midpoint=parameters["shape_midpoint"], width=parameters["shape_width"]
        )
        fraction = parameters["route_fraction_max"] * shape
        local = normalize_inside(
            deposit_baryons(context.data, cluster_protocol["preprocessing"]["base_baryon_smoothing_kpc"]),
            context.aperture,
        )
        endpoint = normalize_inside(
            gaussian_filter(local, parameters["q_R80"] * r80 / spacing, mode="constant"),
            context.aperture,
        )
        simple = normalize_inside((1.0 - fraction) * local + fraction * endpoint, context.aperture)
        cluster_endpoint = normalize_inside(
            gaussian_filter(local, 0.5 * r80 / spacing, mode="constant"), context.aperture
        )
        cluster_locked = cluster_endpoint
        predictions = {"local": local, "simple_shape": simple, "cluster_locked": cluster_locked}
        for name, values in predictions.items():
            cluster_rows.append(
                {"system": label, "cohort": cohort, "candidate": name, "R50_kpc": r50, "R80_kpc": r80, "C_R50_over_R80": r50 / r80, "shape_activation": shape, "effective_route_fraction": fraction, **shape_metrics(values, context.target, context.aperture)}
            )
            glafic_rows.append(
                {"system": label, "cohort": cohort, "candidate": name, **shape_metrics(values, context.glafic_target, context.aperture)}
            )
        for realization, raw_map in enumerate(context.data.range_maps):
            target = lens_source_map(raw_map, context.data.radius, spacing, 20.0, (250.0, 300.0))
            values = {
                name: shape_metrics(candidate, target, context.aperture)["jensen_shannon"]
                for name, candidate in predictions.items()
            }
            uncertainty_rows.append(
                {"system": label, "cohort": cohort, "realization": realization, **{f"{name}_jsd": value for name, value in values.items()}, "simple_improves_local": values["simple_shape"] < values["local"], "simple_beats_cluster_locked": values["simple_shape"] < values["cluster_locked"]}
            )
    cluster_scores = pd.DataFrame(cluster_rows)
    uncertainty = pd.DataFrame(uncertainty_rows)
    glafic = pd.DataFrame(glafic_rows)
    holdout_systems = cluster_scores[cluster_scores.cohort == "holdout"].pivot(index="system", columns="candidate", values="jensen_shannon")
    holdout_means = holdout_systems.mean()
    cluster_gain = 1.0 - holdout_means.simple_shape / holdout_means.local
    cluster_locked_gain = 1.0 - holdout_means.cluster_locked / holdout_means.local
    systems_improved = int(np.sum(holdout_systems.simple_shape < holdout_systems.local))
    holdout_uncertainty = uncertainty[uncertainty.cohort == "holdout"]
    realizations_improved = float(holdout_uncertainty.simple_improves_local.mean())
    glafic_holdout = glafic[glafic.cohort == "holdout"].groupby("candidate").jensen_shannon.mean()
    glafic_gain = 1.0 - glafic_holdout.simple_shape / glafic_holdout.local

    solar_radius = galaxy_protocol["solar_test"]["solar_radius_m"]
    solar_r80 = solar_radius * 0.8 ** (1.0 / 3.0)
    solar_r50 = solar_radius * 0.5 ** (1.0 / 3.0)
    solar_shape = radial_shape_activation(
        solar_r50 / solar_r80, midpoint=parameters["shape_midpoint"], width=parameters["shape_width"]
    )
    solar_sigma = parameters["q_R80"] * solar_r80
    mercury_tail = gaussian_tail_upper_bound(
        evaluation_radius=galaxy_protocol["solar_test"]["mercury_perihelion_AU"] * AU_M,
        source_radius=solar_radius,
        sigma=solar_sigma,
    )
    solar = {
        "uniform_sphere_C_R50_over_R80": solar_r50 / solar_r80,
        "shape_activation": solar_shape,
        "effective_route_fraction": parameters["route_fraction_max"] * solar_shape,
        "diffusion_sigma_m": solar_sigma,
        "Mercury_exterior_mass_tail_upper_bound": mercury_tail,
        "planetary_exterior_force_change_upper_bound": mercury_tail,
        "PPN_Cassini_defined": False,
        "stellar_interior_tested": False,
    }
    cfg = protocol["interpretation_gates"]
    gates = {
        "galaxy_equal_RMSE_improvement_pass": bool(equal_improvement >= cfg["galaxy_equal_RMSE_improvement_fraction_min"]),
        "galaxy_fold_count_pass": bool(np.sum(galaxy_folds.improvement_fraction > 0.0) >= cfg["galaxy_folds_improved_min"]),
        "galaxy_bootstrap_pass": bool(galaxy_bootstrap >= cfg["galaxy_bootstrap_probability_min"]),
        "cluster_holdout_improvement_pass": bool(cluster_gain >= cfg["cluster_holdout_improvement_fraction_min"]),
        "cluster_holdout_system_count_pass": bool(systems_improved >= cfg["cluster_holdout_systems_improved_min"]),
        "cluster_realization_fraction_pass": bool(realizations_improved >= cfg["cluster_holdout_realizations_improved_fraction_min"]),
        "cluster_GLAFIC_improvement_pass": bool(glafic_gain >= cfg["cluster_GLAFIC_improvement_fraction_min"]),
        "solar_exterior_tail_pass": bool(mercury_tail <= cfg["solar_exterior_tail_fraction_max"]),
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    outer.to_csv(output / protocol["outputs"]["galaxy_predictions"], index=False)
    galaxy_scores.to_csv(output / protocol["outputs"]["galaxy_scores"], index=False)
    cluster_scores.to_csv(output / protocol["outputs"]["cluster_system_scores"], index=False)
    uncertainty.to_csv(output / protocol["outputs"]["cluster_uncertainty"], index=False)
    glafic.to_csv(output / protocol["outputs"]["cluster_glafic_scores"], index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].bar(
        ["fixed RAR", "simple shape"],
        [galaxy_metrics["fixed_RAR"]["outer_equal_galaxy_RMSE_km_s"], galaxy_metrics["simple_shape"]["outer_equal_galaxy_RMSE_km_s"]],
    )
    axes[0].set(ylabel="equal-galaxy RMSE (km/s)", title="131 galaxies (post-hoc fixed law)")
    holdout_systems.plot(kind="bar", ax=axes[1])
    axes[1].set(ylabel="Jensen-Shannon distance", title="three cluster holdouts")
    axes[1].tick_params(axis="x", rotation=20)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    report = {
        "report_version": "P0597-SIMPLE-SHAPE-LAW-CROSS-DOMAIN-RESULTS-0.1.0",
        "status": "complete_posthoc_cross_domain_replay",
        "selection_disclosure": protocol["selection_disclosure"],
        "parameters": parameters,
        "coverage": {"galaxies": len(galaxy_scores), "galaxy_outer_points": len(outer), "clusters": len(contexts), "cluster_holdouts": 3, "lenstool_realizations": len(uncertainty)},
        "galaxy": {"metrics": galaxy_metrics, "equal_galaxy_improvement_fraction_vs_fixed_RAR": equal_improvement, "galaxies_improved": int(np.sum(galaxy_scores.delta_MSE_km_s2 > 0.0)), "galaxies_improved_fraction": float(np.mean(galaxy_scores.delta_MSE_km_s2 > 0.0)), "folds": galaxy_folds.to_dict("records"), "folds_improved": int(np.sum(galaxy_folds.improvement_fraction > 0.0)), "bootstrap_probability_equal_galaxy_MSE_improvement": galaxy_bootstrap},
        "cluster": {"holdout_mean_jsd": {key: float(value) for key, value in holdout_means.items()}, "holdout_improvement_fraction_vs_local": float(cluster_gain), "cluster_locked_improvement_fraction_vs_local": float(cluster_locked_gain), "cluster_locked_gain_retained_fraction": float(cluster_gain / cluster_locked_gain), "holdout_systems_improved": systems_improved, "holdout_realizations_improved_fraction": realizations_improved, "holdout_realizations_beating_cluster_locked_fraction": float(holdout_uncertainty.simple_beats_cluster_locked.mean()), "GLAFIC_holdout_mean_jsd": {key: float(value) for key, value in glafic_holdout.items()}, "GLAFIC_improvement_fraction_vs_local": float(glafic_gain)},
        "solar": solar,
        "gates": gates,
        "all_interpretation_gates_pass": bool(all(gates.values())),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0597 simple radial-shape law cross-domain replay\n\n"
        f"The post-hoc fixed law changes galaxy equal-weighted RMSE by {100.0 * equal_improvement:+.2f}% "
        f"versus fixed RAR and improves {report['galaxy']['galaxies_improved']}/131 galaxies. All "
        f"{report['galaxy']['folds_improved']}/5 fixed-formula folds improve; galaxy bootstrap probability "
        f"is {100.0 * galaxy_bootstrap:.1f}%. On three cluster holdouts it changes normalized morphology "
        f"JSD by {100.0 * cluster_gain:+.2f}% versus local baryon light, improves {systems_improved}/3 systems, "
        f"and changes GLAFIC by {100.0 * glafic_gain:+.2f}%. The formula was chosen after P0596 disclosure "
        "and is therefore a candidate for future independent testing, not a validated discovery.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
