#!/usr/bin/env python3
"""Full bounded vector-completion test on BCG, CLASH, SPARC, and RX J2129."""

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
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_cpr0_accept_clash_bridge import (  # noqa: E402
    assign_group_folds,
    domain_metrics,
    equal_group_domain_mse,
)
from run_sigma_field_exploration import run_diagnostic_lensing  # noqa: E402
from run_sparc_independent_nuisance_refit import (  # noqa: E402
    bounds_for,
    build_frame,
    optimizer_settings,
    starts_for,
)
from run_sparc_independent_nuisance_refit import (  # noqa: E402
    metrics as sparc_metrics,
)

from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.sparc_refit import effective_prediction, fit_galaxy  # noqa: E402
from voidscreen.tensor_completion import (  # noqa: E402
    TENSOR_MODELS,
    predict_tensor_acceleration,
    spherical_profile_tidal_eigenvalues,
    spherical_tidal_eigenvalues,
    tensor_completion,
)
from voidscreen.unbounded_running import (  # noqa: E402
    M_SUN_KG,
    RUNNING_MODELS,
    TENSOR_RUNNING_MODELS,
    VARIABLE_EXPONENT_DENSITY_MODELS,
    equivalent_enclosed_baryonic_mass_msun,
    predict_running_acceleration,
)
from voidscreen.vector_completion import (  # noqa: E402
    bounded_completion,
    predict_completion_acceleration,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def predict_bridge(frame: pd.DataFrame, model: str, parameters) -> np.ndarray:
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(float))
    if model in TENSOR_MODELS:
        eigenvalues = spherical_tidal_eigenvalues(
            gbar,
            frame["radius_kpc"].to_numpy(float),
            frame["local_density_g_cm3"].to_numpy(float),
        )
        result = predict_tensor_acceleration(
            gbar,
            eigenvalues,
            model,
            parameters,
            direction_components=(1.0, 0.0, 0.0),
        )
        return np.log10(result["predicted_acceleration_m_s2"])
    coherence = (
        frame["coherence"].to_numpy(float)
        if model == "coherence_completion"
        else None
    )
    result = predict_completion_acceleration(
        gbar,
        frame["radius_kpc"].to_numpy(float),
        parameters,
        coherence=coherence,
    )
    return np.log10(result["predicted_acceleration_m_s2"])


def fit_bridge(
    frame: pd.DataFrame,
    model: str,
    bounds,
    protocol: dict,
    seed: int,
) -> np.ndarray:
    objective = lambda values: equal_group_domain_mse(
        frame, predict_bridge(frame, model, values)
    )
    settings = protocol["optimization"]
    global_fit = differential_evolution(
        objective,
        list(map(tuple, bounds)),
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
        bounds=list(map(tuple, bounds)),
        options={"maxiter": 8000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    return np.asarray(local.x if local.success else global_fit.x, dtype=float)


def at_boundary(values, bounds, tolerance: float) -> list[bool]:
    flags = []
    for value, (lower, upper) in zip(values, bounds, strict=True):
        width = float(upper) - float(lower)
        flags.append(
            bool(
                value <= float(lower) + tolerance * width
                or value >= float(upper) - tolerance * width
            )
        )
    return flags


def cross_validate_bridge(
    frame: pd.DataFrame, model: str, specification: dict, protocol: dict
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    folds = int(protocol["sample"]["bridge_folds"])
    seed = int(protocol["sample"]["bridge_seed"])
    bounds = specification["bounds"]
    names = specification["parameters"]
    tolerance = float(protocol["optimization"]["boundary_fraction_tolerance"])
    predictions = np.full(len(frame), np.nan)
    records = []
    for fold in range(folds):
        training = frame[frame["fold"] != fold]
        heldout = frame[frame["fold"] == fold]
        values = fit_bridge(training, model, bounds, protocol, seed + fold)
        predictions[heldout.index] = predict_bridge(heldout, model, values)
        records.append(
            {
                "fold": fold,
                "heldout": {
                    domain: sorted(block["system"].astype(str).unique().tolist())
                    for domain, block in heldout.groupby("domain", sort=True)
                },
                "parameters": dict(zip(names, map(float, values), strict=True)),
                "at_boundary": dict(
                    zip(names, at_boundary(values, bounds, tolerance), strict=True)
                ),
            }
        )
    if np.any(~np.isfinite(predictions)):
        raise RuntimeError(f"{model} left missing heldout predictions")
    full = fit_bridge(frame, model, bounds, protocol, seed + 100)
    return predictions, records, full


def run_sparc_transfer(
    model: str,
    parameters: np.ndarray,
    sparc_protocol: dict,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if model in TENSOR_MODELS or model in RUNNING_MODELS:
        internal_model = model
        if (
            model in TENSOR_MODELS
            or model in TENSOR_RUNNING_MODELS
            or model in VARIABLE_EXPONENT_DENSITY_MODELS
        ):
            geometry_name = "primary"
            density_geometry = next(
                item
                for item in sparc_protocol["candidate_density_scenarios"]
                if item["name"] == geometry_name
            )
        else:
            geometry_name = "not_used"
            density_geometry = None
    else:
        internal_model = (
            "vector_completion_coherence"
            if model == "coherence_completion"
            else "vector_completion"
        )
        geometry_name = "not_used"
        density_geometry = None
    settings = optimizer_settings(sparc_protocol)
    bounds = bounds_for(internal_model, sparc_protocol)
    point_blocks = []
    fit_rows = []
    for galaxy_index, block in frame.groupby("galaxy_index", sort=True):
        inner = block[block["split"].eq("inner_train")]
        fitted = fit_galaxy(
            inner,
            model=internal_model,
            settings=settings,
            starts=starts_for(internal_model, sparc_protocol, int(galaxy_index)),
            bounds=bounds,
            candidate_parameters=parameters,
            density_geometry=density_geometry,
            max_iterations=int(sparc_protocol["nuisance_fit"]["max_iterations"]),
        )
        predicted = effective_prediction(
            block,
            fitted.theta,
            model=internal_model,
            settings=settings,
            candidate_parameters=parameters,
            density_geometry=density_geometry,
        )
        points = block[
            [
                "galaxy",
                "galaxy_index",
                "split",
                "radius_catalog_kpc",
                "velocity_observed_catalog_kms",
                "velocity_error_catalog_kms",
            ]
        ].copy()
        for name, values in predicted.items():
            points[name] = values
        points.insert(0, "model", model)
        point_blocks.append(points)
        fit_rows.append(
            {
                "model": model,
                "galaxy": str(block["galaxy"].iloc[0]),
                "finite": fitted.finite,
                "success": fitted.success,
                "objective": fitted.objective,
                "any_nuisance_at_boundary": bool(
                    any(
                        abs(value - low) <= 1.0e-4 * max(1.0, abs(low))
                        or abs(value - high) <= 1.0e-4 * max(1.0, abs(high))
                        for value, (low, high) in zip(fitted.theta, bounds, strict=True)
                    )
                ),
            }
        )
    points = pd.concat(point_blocks, ignore_index=True)
    fits = pd.DataFrame(fit_rows)
    diagnostics = {
        "inner_train": sparc_metrics(points, "inner_train"),
        "outer_holdout": sparc_metrics(points, "outer_holdout"),
        "finite_fit_fraction": float(fits["finite"].mean()),
        "optimizer_success_fraction": float(fits["success"].mean()),
        "nuisance_boundary_fraction": float(fits["any_nuisance_at_boundary"].mean()),
        "density_geometry": geometry_name,
    }
    return points, diagnostics


def raw_lensing_profile(model: str, parameters: np.ndarray, protocol: dict) -> pd.DataFrame:
    source = pd.read_csv(ROOT / protocol["inputs"]["RXJ_profile"])
    cluster = source[
        source["model"].eq("sigma_refracted_AQUAL")
        & source["domain"].eq("RXJ2129")
    ].sort_values("radius_kpc")
    if model in TENSOR_MODELS:
        gbar = cluster["gbar_m_s2"].to_numpy(float)
        radius = cluster["radius_kpc"].to_numpy(float)
        eigenvalues = spherical_profile_tidal_eigenvalues(gbar, radius)
        completion = predict_tensor_acceleration(
            gbar,
            eigenvalues,
            model,
            parameters,
            direction_components=(1.0, 0.0, 0.0),
        )
    elif model in RUNNING_MODELS:
        gbar = cluster["gbar_m_s2"].to_numpy(float)
        radius = cluster["radius_kpc"].to_numpy(float)
        eigenvalues = (
            spherical_profile_tidal_eigenvalues(gbar, radius)
            if model in TENSOR_RUNNING_MODELS
            else None
        )
        local_density = None
        if model in VARIABLE_EXPONENT_DENSITY_MODELS:
            radius_m = radius * KPC_M
            mass_kg = (
                equivalent_enclosed_baryonic_mass_msun(gbar, radius) * M_SUN_KG
            )
            density_kg_m3 = np.gradient(mass_kg, radius_m, edge_order=2) / (
                4.0 * math.pi * np.square(radius_m)
            )
            local_density = np.maximum(density_kg_m3 * 1.0e-3, 1.0e-35)
        completion = predict_running_acceleration(
            gbar,
            radius,
            model,
            parameters,
            tidal_eigenvalues_s2=eigenvalues,
            local_density_g_cm3=local_density,
        )
    else:
        completion = predict_completion_acceleration(
            cluster["gbar_m_s2"].to_numpy(float),
            cluster["radius_kpc"].to_numpy(float),
            parameters,
            coherence=(
                np.zeros(len(cluster)) if model == "coherence_completion" else None
            ),
        )
    return pd.DataFrame(
        {
            "domain": "RXJ2129",
            "radius_kpc": cluster["radius_kpc"].to_numpy(float),
            "gSigma_m_s2": completion["predicted_acceleration_m_s2"],
        }
    )


def solar_diagnostics(model: str, parameters: np.ndarray, protocol: dict) -> dict:
    solar = protocol["solar_diagnostics"]
    gm = float(solar["solar_GM_m3_s2"])
    earth_radius = float(solar["Earth_orbit_m"])
    tidal_earth = gm / earth_radius**3
    if model in TENSOR_MODELS:
        result = tensor_completion(
            [[-2.0 * tidal_earth, tidal_earth, tidal_earth]],
            [[1.0, 0.0, 0.0]],
            model,
            parameters,
        )
        earth_enhancement = float(result["enhancement_relative_to_local_G"][0])
        earth_norm = float(result["tidal_norm_s2"][0])
    else:
        result = bounded_completion(
            [tidal_earth],
            solar_completion=float(parameters[0]),
            tidal_transition_s2=float(10.0 ** parameters[1]),
            transition_power=float(parameters[2]),
        )
        earth_enhancement = float(result["enhancement_relative_to_local_G"][0])
        earth_norm = tidal_earth
    transition_radius_m = (gm / (10.0 ** float(parameters[1]))) ** (1.0 / 3.0)
    return {
        "Earth_tidal_curvature_s2": tidal_earth,
        "Earth_tensor_norm_s2": earth_norm,
        "Earth_enhancement_relative_to_local_G": earth_enhancement,
        "Earth_fractional_change": earth_enhancement - 1.0,
        "solar_point_mass_transition_radius_AU": float(
            transition_radius_m / float(solar["astronomical_unit_m"])
        ),
    }


def completion_summary(
    model: str, parameters: np.ndarray, frame: pd.DataFrame
) -> dict:
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(float))
    if model in TENSOR_MODELS:
        eigenvalues = spherical_tidal_eigenvalues(
            gbar,
            frame["radius_kpc"].to_numpy(float),
            frame["local_density_g_cm3"].to_numpy(float),
        )
        result = predict_tensor_acceleration(
            gbar,
            eigenvalues,
            model,
            parameters,
            direction_components=(1.0, 0.0, 0.0),
        )
        completion = result["projected_completion_fraction"]
        tensor_values = result["completion_tensor_eigenvalues"]
        enhancement = result["enhancement_relative_to_local_G"]
        return {
            "G_max_over_G_solar": float(1.0 / parameters[0]),
            "minimum_completion_fraction": float(np.min(tensor_values)),
            "median_completion_fraction": float(np.median(completion)),
            "maximum_completion_fraction": float(np.max(tensor_values)),
            "minimum_projected_completion_fraction": float(np.min(completion)),
            "maximum_projected_completion_fraction": float(np.max(completion)),
            "median_enhancement_relative_to_local_G": float(np.median(enhancement)),
            "maximum_enhancement_relative_to_local_G": float(np.max(enhancement)),
            "median_projected_availability": float(np.median(result["projected_availability"])),
        }
    result = predict_completion_acceleration(
        gbar,
        frame["radius_kpc"].to_numpy(float),
        parameters,
        coherence=(
            frame["coherence"].to_numpy(float)
            if model == "coherence_completion"
            else None
        ),
    )
    completion = result["completion_fraction"]
    enhancement = result["enhancement_relative_to_local_G"]
    return {
        "G_max_over_G_solar": float(1.0 / parameters[0]),
        "minimum_completion_fraction": float(np.min(completion)),
        "median_completion_fraction": float(np.median(completion)),
        "maximum_completion_fraction": float(np.max(completion)),
        "median_enhancement_relative_to_local_G": float(np.median(enhancement)),
        "maximum_enhancement_relative_to_local_G": float(np.max(enhancement)),
    }


def make_figure(report: dict, bridge_predictions: pd.DataFrame, output: Path) -> None:
    models = list(report["models"])
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    labels = ["Newtonian", "simple MOND", "prior Sigma", *models]
    bridge_values = [
        report["references"]["bridge"]["Newtonian"]["equal_domain_RMSE_dex"],
        report["references"]["bridge"]["simple_MOND"]["equal_domain_RMSE_dex"],
        report["references"]["bridge"]["prior_Sigma"]["equal_domain_RMSE_dex"],
        *[report["models"][name]["bridge_metrics"]["equal_domain_RMSE_dex"] for name in models],
    ]
    model_colors = [plt.get_cmap("tab10")(index) for index in range(len(models))]
    axes[0, 0].bar(
        labels,
        bridge_values,
        color=["#999999", "#7570b3", "#1b9e77", *model_colors],
    )
    axes[0, 0].set(title="Held-out BCG + cluster bridge", ylabel="equal-domain RMSE (dex)")
    axes[0, 0].tick_params(axis="x", rotation=25)

    sparc_labels = ["RAR", "MOND", "prior Sigma", *models]
    sparc_values = [
        report["references"]["SPARC"]["fixed_RAR_outer_RMSE_km_s"],
        report["references"]["SPARC"]["simple_MOND_outer_RMSE_km_s"],
        report["references"]["SPARC"]["prior_Sigma_outer_RMSE_km_s"],
        *[report["models"][name]["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"] for name in models],
    ]
    axes[0, 1].bar(
        sparc_labels,
        sparc_values,
        color=["#666666", "#7570b3", "#1b9e77", *model_colors],
    )
    axes[0, 1].set(title="SPARC outer-radius transfer", ylabel="RMSE (km/s)")
    axes[0, 1].tick_params(axis="x", rotation=25)

    tidal = np.logspace(-35, -12, 500)
    for name, color in zip(models, model_colors, strict=True):
        values = report["models"][name]["full_fit_parameter_vector"]
        if name in TENSOR_MODELS:
            result = tensor_completion(
                np.stack([-2.0 * tidal, tidal, tidal], axis=-1),
                np.broadcast_to([1.0, 0.0, 0.0], (len(tidal), 3)),
                name,
                values,
            )
        else:
            result = bounded_completion(
                tidal,
                solar_completion=values[0],
                tidal_transition_s2=10.0 ** values[1],
                transition_power=values[2],
            )
        axes[1, 0].semilogx(
            tidal,
            result["enhancement_relative_to_local_G"],
            color=color,
            label=name,
        )
    axes[1, 0].axvline(3.96e-14, color="black", linestyle="--", label="Earth orbit")
    axes[1, 0].invert_xaxis()
    axes[1, 0].set(title="Bounded completion law", xlabel="tidal curvature (s^-2)", ylabel="g / g using local G")
    axes[1, 0].legend(fontsize=8)

    raw_labels = ["zero slip", "compact halo", *models]
    raw_values = [
        report["references"]["raw_lensing"]["zero_slip_heldout_RMS_arcsec"],
        report["references"]["raw_lensing"]["compact_halo_heldout_RMS_arcsec"],
        *[report["models"][name]["raw_lensing"]["heldout"]["exact_radial_RMS_arcsec"] for name in models],
    ]
    axes[1, 1].bar(
        raw_labels,
        raw_values,
        color=["#1b9e77", "#444444", *model_colors],
    )
    axes[1, 1].axhline(1.0, color="black", linestyle="--")
    axes[1, 1].set(title="RX J2129 raw image positions", ylabel="heldout RMS (arcsec)")
    axes[1, 1].tick_params(axis="x", rotation=25)

    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/vector_completion_full_test_protocol.json",
    )
    args = parser.parse_args()
    config_path = args.protocol.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] not in {
        "frozen_before_bounded_completion_scores",
        "frozen_before_tensor_scores",
    }:
        raise RuntimeError("completion protocol was not frozen before scoring")

    sample = pd.read_csv(ROOT / protocol["inputs"]["bridge_sample"])
    sample = assign_group_folds(
        sample.drop(columns=["fold"], errors="ignore"),
        int(protocol["sample"]["bridge_folds"]),
        int(protocol["sample"]["bridge_seed"]),
    )
    reference_bridge = json.loads(
        (ROOT / protocol["inputs"]["bridge_reference_report"]).read_text()
    )
    sparc_protocol = json.loads(
        (ROOT / protocol["inputs"]["SPARC_protocol"]).read_text()
    )
    sparc_reference = json.loads(
        (ROOT / protocol["inputs"]["SPARC_reference_report"]).read_text()
    )
    raw_reference = json.loads(
        (ROOT / protocol["inputs"]["raw_lensing_reference_report"]).read_text()
    )

    model_results = {}
    bridge_tables = []
    full_parameters = {}
    for model, specification in protocol["models"].items():
        print(f"cross-validating bridge model={model}", flush=True)
        predicted, fold_fits, parameters = cross_validate_bridge(
            sample, model, specification, protocol
        )
        full_parameters[model] = parameters
        metrics = domain_metrics(sample, predicted)
        table = sample.copy()
        table["model"] = model
        table["predicted_log_gobs"] = predicted
        table["residual_dex"] = predicted - table["log_gobs"]
        bridge_tables.append(table)
        boundary = dict(
            zip(
                specification["parameters"],
                at_boundary(
                    parameters,
                    specification["bounds"],
                    float(protocol["optimization"]["boundary_fraction_tolerance"]),
                ),
                strict=True,
            )
        )
        model_results[model] = {
            "bridge_metrics": metrics,
            "fold_fits": fold_fits,
            "full_fit_parameters": dict(
                zip(specification["parameters"], map(float, parameters), strict=True)
            ),
            "full_fit_parameter_vector": list(map(float, parameters)),
            "full_fit_at_boundary": boundary,
            "completion": completion_summary(model, parameters, sample),
            "solar": solar_diagnostics(model, parameters, protocol),
        }

    bridge_predictions = pd.concat(bridge_tables, ignore_index=True)
    selected_model = min(
        model_results,
        key=lambda name: model_results[name]["bridge_metrics"]["equal_domain_RMSE_dex"],
    )

    print("building frozen SPARC frame", flush=True)
    sparc_frame = build_frame(
        sparc_protocol,
        ROOT / protocol["inputs"]["SPARC_raw"],
        ROOT / protocol["inputs"]["SPARC_morphology"],
    )
    sparc_tables = []
    for model, parameters in full_parameters.items():
        print(f"independent SPARC nuisance refit model={model}", flush=True)
        points, diagnostics = run_sparc_transfer(
            model, parameters, sparc_protocol, sparc_frame
        )
        sparc_tables.append(points)
        model_results[model]["SPARC_metrics"] = diagnostics
    sparc_predictions = pd.concat(sparc_tables, ignore_index=True)

    raw_tables = []
    for model, parameters in full_parameters.items():
        print(f"raw RXJ2129 lensing model={model}", flush=True)
        profile = raw_lensing_profile(model, parameters, protocol)
        predictions, summary = run_diagnostic_lensing(
            pd.Series(model_results[model]["full_fit_parameters"]), protocol, profile
        )
        predictions["completion_model"] = model
        raw_tables.append(predictions)
        model_results[model]["raw_lensing"] = summary
    raw_predictions = pd.concat(raw_tables, ignore_index=True)

    references = {
        "bridge": {
            "Newtonian": reference_bridge["metrics"]["Newtonian"],
            "simple_MOND": reference_bridge["metrics"]["simple_MOND"],
            "prior_Sigma": reference_bridge["metrics"][
                "RAR_sharp_coherence_gated_RG"
            ],
        },
        "SPARC": {
            "fixed_RAR_outer_RMSE_km_s": sparc_reference["scores"][
                "fixed_RAR:invariant"
            ]["outer_holdout"]["RMSE_km_s"],
            "simple_MOND_outer_RMSE_km_s": sparc_reference["scores"][
                "simple_MOND:invariant"
            ]["outer_holdout"]["RMSE_km_s"],
            "prior_Sigma_outer_RMSE_km_s": sparc_reference["scores"][
                "RAR_sharp_coherence_gated_RG:primary"
            ]["outer_holdout"]["RMSE_km_s"],
            "NFW_outer_RMSE_km_s": sparc_reference["scores"]["NFW:invariant"][
                "outer_holdout"
            ]["RMSE_km_s"],
        },
        "raw_lensing": {
            "zero_slip_heldout_RMS_arcsec": raw_reference["raw_lensing"]["zero_slip"][
                "heldout"
            ]["exact_radial_RMS_arcsec"],
            "radial_slip_heldout_RMS_arcsec": raw_reference["raw_lensing"][
                "radial_selected"
            ]["heldout"]["exact_radial_RMS_arcsec"],
            "compact_halo_heldout_RMS_arcsec": raw_reference["raw_lensing"][
                "compact_halo_reference_heldout_RMS_arcsec"
            ],
        },
    }

    gates = protocol["advance_gates"]
    for model, result in model_results.items():
        bridge = result["bridge_metrics"]
        outer = result["SPARC_metrics"]["outer_holdout"]
        raw_rms = result["raw_lensing"]["heldout"]["exact_radial_RMS_arcsec"]
        audit = {
            "bridge_equal_domain_pass": bridge["equal_domain_RMSE_dex"]
            <= float(gates["bridge_equal_domain_RMSE_dex_max"]),
            "BCG_pass": bridge["BCG"]["equal_system_RMSE_dex"]
            <= float(gates["BCG_equal_system_RMSE_dex_max"]),
            "cluster_pass": bridge["cluster"]["equal_system_RMSE_dex"]
            <= float(gates["cluster_equal_system_RMSE_dex_max"]),
            "SPARC_transfer_pass": outer["RMSE_km_s"]
            / references["SPARC"]["fixed_RAR_outer_RMSE_km_s"]
            <= float(gates["SPARC_outer_RMSE_relative_to_fixed_RAR_max"]),
            "raw_lensing_pass": raw_rms <= float(gates["raw_heldout_RMS_arcsec_max"]),
            "solar_Earth_pass": abs(result["solar"]["Earth_fractional_change"])
            <= float(gates["solar_Earth_fractional_change_max"]),
            "bounded_completion_pass": result["completion"][
                "maximum_completion_fraction"
            ]
            <= 1.0 + 1.0e-12,
            "full_fit_not_at_boundary_pass": not any(
                result["full_fit_at_boundary"].values()
            ),
        }
        audit["all_observational_gates_pass"] = all(audit.values())
        result["gate_audit"] = audit

    selected_result = model_results[selected_model]
    report = {
        "report_version": protocol["protocol_version"],
        "status": f"completed {protocol.get('family_label', 'bounded vector completion')} full test",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "root_equation": protocol["root_equation"],
        "coverage": {
            "bridge": {"rows": len(sample), "systems": int(sample["system"].nunique())},
            "SPARC": {
                "galaxies": int(sparc_frame["galaxy"].nunique()),
                "inner_points": int(sparc_frame["split"].eq("inner_train").sum()),
                "outer_points": int(sparc_frame["split"].eq("outer_holdout").sum()),
            },
            "raw_lensing_images_per_model": int(
                raw_predictions.groupby("completion_model").size().iloc[0]
            ),
        },
        "selection": {
            "selected_model": selected_model,
            "rule": protocol["selection"]["rule"],
            "selected_bridge_equal_domain_RMSE_dex": selected_result[
                "bridge_metrics"
            ]["equal_domain_RMSE_dex"],
            "parameters_selected_from_SPARC": 0,
            "parameters_selected_from_raw_images": 0,
        },
        "models": model_results,
        "references": references,
        "selected_verdict": {
            "all_observational_gates_pass": selected_result["gate_audit"][
                "all_observational_gates_pass"
            ],
            "covariant_action_derived": False,
            "interpretation": "The ceiling interpretation is valid algebraically only if the bounded law fits all domains with one setting. It is not independently distinguished from an environment-dependent effective G by these observations.",
        },
        "claim_limits": protocol["claim_limits"],
        "input_hashes": {
            key: sha256(ROOT / path)
            for key, path in protocol["inputs"].items()
            if (ROOT / path).is_file()
        },
        "outputs": protocol["outputs"],
    }

    output = ROOT / Path(protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    bridge_predictions.to_csv(ROOT / protocol["outputs"]["bridge_predictions"], index=False)
    sparc_predictions.to_csv(ROOT / protocol["outputs"]["SPARC_predictions"], index=False)
    raw_predictions.to_csv(ROOT / protocol["outputs"]["raw_lensing_predictions"], index=False)
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(report, bridge_predictions, ROOT / protocol["outputs"]["figure"])

    selected = model_results[selected_model]
    summary = f"""# {protocol.get('family_label', 'Bounded vector completion').title()} full test

## Result

Selected without looking at SPARC or raw images: **{selected_model}**.

- Universal parameters: `{json.dumps(json_safe(selected['full_fit_parameters']))}`
- Interpreted maximum coupling: `G_max/G_solar = {selected['completion']['G_max_over_G_solar']:.3f}`
- Held-out BCG+cluster equal-domain RMSE: `{selected['bridge_metrics']['equal_domain_RMSE_dex']:.4f} dex`
- SPARC outer-radius RMSE: `{selected['SPARC_metrics']['outer_holdout']['RMSE_km_s']:.3f} km/s`
- RX J2129 raw heldout RMS: `{selected['raw_lensing']['heldout']['exact_radial_RMS_arcsec']:.3f} arcsec`
- Earth-orbit fractional change: `{selected['solar']['Earth_fractional_change']:.3e}`
- All frozen observational gates pass: `{selected['gate_audit']['all_observational_gates_pass']}`

## Interpretation

The model never raises any completion-tensor eigenvalue above 100%. Apparent extra gravity relative to Newtonian calculations is interpreted as directional recovery of vectors missing from the Solar-calibrated sum. This interpretation is not observationally separable from a bounded environment-dependent tensor coupling without an independent measurement of the proposed maximum coupling.

{('The tidal tensor is reconstructed from symmetry and density data, and the RX J2129 calculation is not yet a full three-dimensional tensor ray trace.' if protocol['status'] == 'frozen_before_tensor_scores' else 'The coherence variant uses provisional, non-identical observables across domains.')} RX J2129 is a spent-holdout diagnostic, and no covariant action has yet been derived.
"""
    (ROOT / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
