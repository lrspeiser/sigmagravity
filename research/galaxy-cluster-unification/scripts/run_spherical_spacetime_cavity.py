#!/usr/bin/env python3
"""Test closed-space Gauss flow and hard matter-cavity interpretations."""

from __future__ import annotations

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
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sparc_independent_nuisance_refit import metrics as sparc_metrics  # noqa: E402
from run_vector_completion_full_test import at_boundary  # noqa: E402
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.spherical_spacetime import (  # noqa: E402
    C_M_S,
    global_closed_acceleration,
    hard_cavity_best_axis_enhancement,
    hard_cavity_isotropic_rms_enhancement,
    local_mass_curvature_acceleration,
    stellar_area_covering_fraction,
)


G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
R_SUN_M = 6.957e8
AU_M = 149_597_870_700.0


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
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def assign_folds(frame: pd.DataFrame, folds: int, seed: int) -> pd.Series:
    names = np.asarray(sorted(frame["galaxy"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    mapping = {str(name): int(index % folds) for index, name in enumerate(names)}
    return frame["galaxy"].map(mapping).astype(int)


def model_acceleration(
    model: str,
    gbar,
    radius_kpc,
    parameters,
    maximum_x: float,
) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if model.startswith("closed_global"):
        return global_closed_acceleration(
            gbar,
            radius_kpc,
            10.0 ** values[0],
            maximum_x=maximum_x,
        )
    if model == "local_GR_curvature":
        return local_mass_curvature_acceleration(
            gbar, radius_kpc, 0.0, maximum_x=maximum_x
        )
    if model == "local_amplified_unscreened":
        return local_mass_curvature_acceleration(
            gbar, radius_kpc, 10.0 ** values[0], maximum_x=maximum_x
        )
    if model == "local_amplified_screened":
        return local_mass_curvature_acceleration(
            gbar,
            radius_kpc,
            10.0 ** values[0],
            acceleration_screen_m_s2=10.0 ** values[1],
            screen_power=values[2],
            maximum_x=maximum_x,
        )
    raise ValueError(model)


def predicted_velocity(frame: pd.DataFrame, model: str, parameters, maximum_x: float) -> np.ndarray:
    acceleration = model_acceleration(
        model,
        frame["g_bar_m_s2"].to_numpy(float),
        frame["radius_adjusted_kpc"].to_numpy(float),
        parameters,
        maximum_x,
    )
    return np.sqrt(
        acceleration * frame["radius_adjusted_kpc"].to_numpy(float) * KPC_M / 1.0e6
    )


def add_prediction(frame: pd.DataFrame, model: str, velocity: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "sphere_model", model)
    output["velocity_predicted_km_s"] = velocity
    inclination_factor = (
        output["velocity_observed_adjusted_km_s"].to_numpy(float)
        / output["velocity_observed_catalog_kms"].to_numpy(float)
    )
    output["velocity_predicted_catalog_km_s"] = velocity / inclination_factor
    output["coherence"] = np.nan
    return output


def equal_galaxy_rar_loss(frame: pd.DataFrame, velocity: np.ndarray) -> float:
    residual = np.log10(
        velocity / frame["velocity_RAR_same_nuisance_km_s"].to_numpy(float)
    )
    table = frame[["galaxy"]].copy()
    table["squared"] = np.square(residual)
    return float(table.groupby("galaxy")["squared"].mean().mean())


def solar_diagnostic(model: str, parameters, maximum_x: float, protocol: dict) -> dict:
    radii_m = np.asarray([R_SUN_M, AU_M, 9.5826 * AU_M])
    radii_kpc = radii_m / KPC_M
    gbar = G_SI * M_SUN_KG / np.square(radii_m)
    try:
        predicted = model_acceleration(model, gbar, radii_kpc, parameters, maximum_x)
        fractional = predicted / gbar - 1.0
    except ValueError:
        fractional = np.full(3, np.inf)
    gates = protocol["solar_gates"]
    maximum = float(np.max(np.abs(fractional)))
    earth = float(fractional[1])
    return {
        "Sun_limb_fractional_change": float(fractional[0]),
        "Earth_orbit_fractional_change": earth,
        "Saturn_orbit_fractional_change": float(fractional[2]),
        "maximum_fractional_change_limb_to_Saturn": maximum,
        "pass": bool(
            maximum <= float(gates["maximum_fractional_change_limb_to_Saturn"])
            and abs(earth) <= float(gates["maximum_Earth_orbit_fractional_change"])
        ),
    }


def fit_model(
    frame: pd.DataFrame,
    validity_frame: pd.DataFrame,
    model: str,
    specification: dict,
    maximum_x: float,
    protocol: dict,
    seed: int,
) -> np.ndarray:
    bounds = list(map(tuple, specification["bounds"]))
    if not bounds:
        return np.asarray([], dtype=float)

    def objective(values) -> float:
        try:
            # Domain validity may use the known radii and baryonic fields, but
            # never the held-out observed velocities.
            predicted_velocity(validity_frame, model, values, maximum_x)
            velocity = predicted_velocity(frame, model, values, maximum_x)
            loss = equal_galaxy_rar_loss(frame, velocity)
            solar = solar_diagnostic(model, values, maximum_x, protocol)
            gates = protocol["solar_gates"]
            excess = max(
                0.0,
                solar["maximum_fractional_change_limb_to_Saturn"]
                / float(gates["maximum_fractional_change_limb_to_Saturn"])
                - 1.0,
            )
            earth = max(
                0.0,
                abs(solar["Earth_orbit_fractional_change"])
                / float(gates["maximum_Earth_orbit_fractional_change"])
                - 1.0,
            )
            return loss + float(gates["penalty_coefficient"]) * (excess**2 + earth**2)
        except (FloatingPointError, OverflowError, ValueError):
            return 1.0e100

    settings = protocol["fit"]
    global_fit = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=int(settings["differential_evolution_maxiter"]),
        popsize=int(settings["differential_evolution_popsize"]),
        tol=1.0e-10,
        polish=False,
        workers=1,
    )
    local = minimize(
        objective,
        global_fit.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    return np.asarray(local.x if local.success else global_fit.x, dtype=float)


def environmental_metrics(model: str, parameters, sample: pd.DataFrame, maximum_x: float) -> dict:
    results = {}
    for domain, selected in sample.groupby("domain", sort=True):
        gbar = np.power(10.0, selected["log_gbar"].to_numpy(float))
        try:
            predicted = model_acceleration(
                model, gbar, selected["radius_kpc"].to_numpy(float), parameters, maximum_x
            )
            residual = np.log10(predicted) - selected["log_gobs"].to_numpy(float)
            results[str(domain)] = {
                "points": len(selected),
                "RMSE_dex": float(np.sqrt(np.mean(np.square(residual)))),
                "mean_residual_dex": float(np.mean(residual)),
                "median_predicted_to_observed": float(np.median(np.power(10.0, residual))),
            }
        except ValueError:
            results[str(domain)] = {
                "points": len(selected),
                "RMSE_dex": math.inf,
                "mean_residual_dex": math.nan,
                "median_predicted_to_observed": math.nan,
            }
    return results


def domain_diagnostic(model: str, parameters, maximum_x: float, protocol: dict) -> dict:
    radius = np.geomspace(0.1, float(protocol["domain"]["maximum_cluster_radius_kpc"]), 512)
    representative_mass_solar = 1.0e14
    gbar = G_SI * representative_mass_solar * M_SUN_KG / np.square(radius * KPC_M)
    try:
        predicted = model_acceleration(model, gbar, radius, parameters, maximum_x)
        return {
            "pass": bool(np.all(np.isfinite(predicted)) and np.all(predicted > 0.0)),
            "maximum_radius_kpc": float(radius[-1]),
            "maximum_enhancement": float(np.max(predicted / gbar)),
        }
    except ValueError as error:
        return {"pass": False, "maximum_radius_kpc": float(radius[-1]), "error": str(error)}


def hard_cavity_diagnostics(frame: pd.DataFrame, morphology: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    columns = ["galaxy", "disk_scale_kpc", "stellar_mass_solar"]
    selected = frame[frame["split"].eq("outer_holdout")].merge(
        morphology[columns], on="galaxy", how="inner", validate="many_to_one"
    )
    selected["disk_scale_adjusted_kpc"] = (
        selected["disk_scale_kpc"] * selected["distance_scale"]
    )
    selected = selected[
        selected["radius_adjusted_kpc"] >= selected["disk_scale_adjusted_kpc"]
    ].copy()
    ratio = (
        selected["disk_scale_adjusted_kpc"].to_numpy(float)
        / selected["radius_adjusted_kpc"].to_numpy(float)
    )
    axis = hard_cavity_best_axis_enhancement(ratio)
    rms = hard_cavity_isotropic_rms_enhancement(ratio)
    radius_m = selected["radius_adjusted_kpc"].to_numpy(float) * KPC_M
    velocity_bar = np.sqrt(selected["g_bar_m_s2"].to_numpy(float) * radius_m / 1.0e6)
    observed = selected["velocity_observed_adjusted_km_s"].to_numpy(float)
    required = np.square(observed / velocity_bar)
    cover = stellar_area_covering_fraction(
        selected["stellar_mass_solar"].to_numpy(float),
        selected["disk_scale_adjusted_kpc"].to_numpy(float),
    )
    selected["cavity_radius_over_radius"] = ratio
    selected["hard_cavity_axis_acceleration_factor"] = axis
    selected["hard_cavity_isotropic_RMS_factor"] = rms
    selected["observed_to_baryon_acceleration_factor"] = required
    selected["stellar_projected_covering_fraction_upper"] = cover
    selected["hard_cavity_axis_velocity_km_s"] = velocity_bar * np.sqrt(axis)
    selected["hard_cavity_RMS_velocity_km_s"] = velocity_bar * np.sqrt(rms)
    return {
        "points_outside_one_disk_scale": len(selected),
        "galaxies": int(selected["galaxy"].nunique()),
        "axis_factor_quantiles": dict(
            zip(["p05", "median", "p95"], map(float, np.quantile(axis, [0.05, 0.5, 0.95])), strict=True)
        ),
        "isotropic_RMS_factor_quantiles": dict(
            zip(["p05", "median", "p95"], map(float, np.quantile(rms, [0.05, 0.5, 0.95])), strict=True)
        ),
        "required_factor_quantiles": dict(
            zip(["p05", "median", "p95"], map(float, np.quantile(required, [0.05, 0.5, 0.95])), strict=True)
        ),
        "fraction_axis_upper_bound_meets_required": float(np.mean(axis >= required)),
        "axis_velocity_RMSE_km_s": float(np.sqrt(np.mean(np.square(velocity_bar * np.sqrt(axis) - observed)))),
        "isotropic_velocity_RMSE_km_s": float(np.sqrt(np.mean(np.square(velocity_bar * np.sqrt(rms) - observed)))),
        "stellar_covering_fraction_quantiles": dict(
            zip(["p05", "median", "p95", "maximum"], map(float, np.quantile(cover, [0.05, 0.5, 0.95, 1.0])), strict=True)
        ),
        "analytic_net_force_in_inviscid_flow": 0.0,
        "boundary_conflict": "The impermeable solution has zero normal flow at the cavity surface; it cannot simultaneously describe gravity passing through and striking the body normally."
    }, selected


def make_figure(report: dict, cavity_points: pd.DataFrame, output: Path) -> None:
    names = list(report["models"])
    labels = [name.replace("closed_", "c:").replace("local_", "l:") for name in names]
    outer = [report["models"][name]["SPARC"]["outer_holdout"]["RMSE_km_s"] for name in names]
    bcg = [report["models"][name]["environment"]["BCG"]["RMSE_dex"] for name in names]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].bar(labels, outer)
    axes[0].axhline(report["reference"]["fixed_RAR_outer_RMSE_km_s"], color="black", linestyle="--", label="fixed RAR")
    axes[0].set(title="Untouched SPARC outer radii", ylabel="RMSE (km/s)")
    axes[0].legend()
    axes[1].bar(labels, bcg)
    axes[1].axhline(0.17, color="gray", linestyle=":", label="gate")
    axes[1].set(title="Cluster-central matter dynamics", ylabel="RMSE (dex)")
    axes[1].legend()
    axes[2].scatter(
        cavity_points["observed_to_baryon_acceleration_factor"],
        cavity_points["hard_cavity_axis_acceleration_factor"],
        s=9,
        alpha=0.45,
    )
    limit = max(2.0, float(np.quantile(cavity_points["observed_to_baryon_acceleration_factor"], 0.95)))
    axes[2].plot([1.0, limit], [1.0, limit], color="black", linestyle="--")
    axes[2].set(xlim=(0.5, limit), ylim=(0.5, limit), xlabel="required acceleration factor", ylabel="most favorable cavity factor", title="Hard-cavity upper bound")
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    config_path = ROOT / "configs/spherical_spacetime_cavity_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_spherical_spacetime_scores":
        raise RuntimeError("protocol is not frozen before scoring")
    maximum_x = float(protocol["domain"]["maximum_closed_sphere_x"])
    source_path = ROOT / protocol["inputs"]["SPARC_points"]
    source = pd.read_csv(source_path)
    frame = source[
        source["model"].eq("fixed_RAR") & source["scenario"].eq("invariant")
    ].copy().reset_index(drop=True)
    frame["fold"] = assign_folds(
        frame, int(protocol["fit"]["galaxy_folds"]), int(protocol["fit"]["fold_seed"])
    )
    reference_report = json.loads((ROOT / protocol["inputs"]["SPARC_report"]).read_text())
    rar = reference_report["scores"]["fixed_RAR:invariant"]
    sample = pd.read_csv(ROOT / protocol["inputs"]["cluster_sample"])
    morphology = pd.read_csv(ROOT / protocol["inputs"]["SPARC_morphology"])
    model_results = {}
    prediction_tables = []
    tolerance = float(protocol["fit"]["boundary_fraction_tolerance"])

    for model, specification in protocol["models"].items():
        print(f"spherical-spacetime model={model}", flush=True)
        fold_predictions = np.full(len(frame), np.nan)
        fold_fits = []
        for fold in range(int(protocol["fit"]["galaxy_folds"])):
            training = frame[frame["fold"].ne(fold) & frame["split"].eq("inner_train")]
            heldout = frame[frame["fold"].eq(fold)]
            fitted = fit_model(
                training,
                frame,
                model,
                specification,
                maximum_x,
                protocol,
                int(protocol["fit"]["fold_seed"]) + fold,
            )
            fold_predictions[heldout.index.to_numpy()] = predicted_velocity(
                heldout, model, fitted, maximum_x
            )
            fold_fits.append(
                dict(zip(specification["parameters"], map(float, fitted), strict=True))
            )
        full = fit_model(
            frame[frame["split"].eq("inner_train")],
            frame,
            model,
            specification,
            maximum_x,
            protocol,
            int(protocol["fit"]["fold_seed"]) + 100,
        )
        velocity = predicted_velocity(frame, model, full, maximum_x)
        points = add_prediction(frame, model, velocity)
        prediction_tables.append(points)
        solar = solar_diagnostic(model, full, maximum_x, protocol)
        domain = domain_diagnostic(model, full, maximum_x, protocol)
        environment = environmental_metrics(model, full, sample, maximum_x)
        boundary = dict(
            zip(
                specification["parameters"],
                at_boundary(full, specification["bounds"], tolerance),
                strict=True,
            )
        )
        inner = sparc_metrics(points, "inner_train")
        outer = sparc_metrics(points, "outer_holdout")
        gates = protocol["advance_gates"]
        advance = bool(
            specification["eligible_to_advance"]
            and outer["RMSE_km_s"] <= float(gates["outer_RMSE_relative_to_fixed_RAR_max"]) * rar["outer_holdout"]["RMSE_km_s"]
            and outer["equal_galaxy_RMSE_km_s"] <= float(gates["outer_equal_galaxy_RMSE_relative_to_fixed_RAR_max"]) * rar["outer_holdout"]["equal_galaxy_RMSE_km_s"]
            and environment["BCG"]["RMSE_dex"] <= float(gates["BCG_RMSE_dex_max"])
            and environment["cluster"]["RMSE_dex"] <= float(gates["cluster_RMSE_dex_max"])
            and solar["pass"]
            and domain["pass"]
            and not any(boundary.values())
        )
        model_results[model] = {
            "parameters": dict(zip(specification["parameters"], map(float, full), strict=True)),
            "parameter_vector": list(map(float, full)),
            "parameter_at_boundary": boundary,
            "fold_fits": fold_fits,
            "fold_RAR_log_velocity_RMSE_dex": float(np.sqrt(equal_galaxy_rar_loss(frame, fold_predictions))),
            "SPARC": {"inner_train": inner, "outer_holdout": outer},
            "environment": environment,
            "solar": solar,
            "cluster_domain": domain,
            "eligible_to_advance": specification["eligible_to_advance"],
            "advance": advance,
        }

    cavity, cavity_points = hard_cavity_diagnostics(frame, morphology)
    advanced = [name for name, result in model_results.items() if result["advance"]]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed galaxy, environment, Solar-System, and cavity-flow stage",
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(config_path)},
        "input_hashes": {
            protocol["inputs"][key]: sha256(ROOT / protocol["inputs"][key])
            for key in ["SPARC_points", "SPARC_report", "SPARC_morphology", "cluster_sample"]
        },
        "reference": {
            "fixed_RAR_outer_RMSE_km_s": rar["outer_holdout"]["RMSE_km_s"],
            "fixed_RAR_outer_equal_galaxy_RMSE_km_s": rar["outer_holdout"]["equal_galaxy_RMSE_km_s"],
        },
        "models": model_results,
        "hard_cavity": cavity,
        "advanced_to_raw_lensing": advanced,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = (ROOT / protocol["outputs"]["galaxy_report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / protocol["outputs"]["galaxy_report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["galaxy_predictions"], index=False
    )
    cavity_points.to_csv(output / "hard_cavity_points.csv", index=False)
    make_figure(report, cavity_points, ROOT / protocol["outputs"]["figure"])
    lines = [
        "# Spherical spacetime and matter-cavity test",
        "",
        "| model | outer RMSE (km/s) | BCG RMSE (dex) | cluster RMSE (dex) | solar | advance |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, result in model_results.items():
        lines.append(
            f"| {name} | {result['SPARC']['outer_holdout']['RMSE_km_s']:.3f} | "
            f"{result['environment']['BCG']['RMSE_dex']:.4f} | "
            f"{result['environment']['cluster']['RMSE_dex']:.4f} | "
            f"{result['solar']['pass']} | {result['advance']} |"
        )
    lines.extend(
        [
            "",
            f"Hard-cavity best-axis median enhancement: **{cavity['axis_factor_quantiles']['median']:.6f}**.",
            f"Hard-cavity upper bound meets observed factor at **{100*cavity['fraction_axis_upper_bound_meets_required']:.2f}%** of eligible points.",
            f"Advanced to raw lensing: **{', '.join(advanced) if advanced else 'none'}**.",
        ]
    )
    (ROOT / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
