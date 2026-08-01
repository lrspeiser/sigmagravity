#!/usr/bin/env python3
"""Test a finite family of bounded, path-memory gravity completion laws."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    metrics as sparc_metrics,
    optimizer_settings,
    starts_for,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.path_completion import (  # noqa: E402
    MASS_PATH_MODELS,
    PATH_MODELS,
    predict_mass_path_completion_frame,
    path_completion_profile,
    predict_path_completion_frame,
)
from voidscreen.sparc_refit import effective_prediction, fit_galaxy  # noqa: E402


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


def bridge_frame_with_gbar(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.reset_index(drop=True).copy()
    output["gbar_m_s2"] = np.power(10.0, output["log_gbar"].to_numpy(float))
    return output


def predict_bridge(frame: pd.DataFrame, model: str, parameters) -> tuple[np.ndarray, dict]:
    fields = predict_path_completion_frame(frame, model, parameters)
    return np.log10(fields["predicted_acceleration_m_s2"]), fields


def fit_bridge(frame: pd.DataFrame, model: str, bounds, protocol: dict, seed: int) -> np.ndarray:
    objective = lambda values: equal_group_domain_mse(
        frame, predict_bridge(frame, model, values)[0]
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


def boundary_flags(values, bounds, tolerance: float) -> list[bool]:
    return [
        bool(
            value <= float(lower) + tolerance * (float(upper) - float(lower))
            or value >= float(upper) - tolerance * (float(upper) - float(lower))
        )
        for value, (lower, upper) in zip(values, bounds, strict=True)
    ]


def cross_validate_bridge(
    frame: pd.DataFrame, model: str, specification: dict, protocol: dict
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    folds = int(protocol["sample"]["bridge_folds"])
    seed = int(protocol["sample"]["bridge_seed"])
    names = specification["parameters"]
    bounds = specification["bounds"]
    tolerance = float(protocol["optimization"]["boundary_fraction_tolerance"])
    prediction = np.full(len(frame), np.nan)
    fit_records = []
    for fold in range(folds):
        training = frame[frame["fold"] != fold]
        heldout = frame[frame["fold"] == fold]
        values = fit_bridge(training, model, bounds, protocol, seed + fold)
        prediction[heldout.index] = predict_bridge(heldout, model, values)[0]
        fit_records.append(
            {
                "fold": fold,
                "heldout": {
                    domain: sorted(group["system"].astype(str).unique().tolist())
                    for domain, group in heldout.groupby("domain", sort=True)
                },
                "parameters": dict(zip(names, map(float, values), strict=True)),
                "at_boundary": dict(
                    zip(
                        names,
                        boundary_flags(values, bounds, tolerance),
                        strict=True,
                    )
                ),
            }
        )
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError(f"{model} left heldout bridge predictions missing")
    full = fit_bridge(frame, model, bounds, protocol, seed + 100)
    return prediction, fit_records, full


def run_sparc_transfer(
    model: str,
    parameters: np.ndarray,
    protocol: dict,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    settings = optimizer_settings(protocol)
    nuisance_bounds = bounds_for(model, protocol)
    blocks = []
    fit_records = []
    for galaxy_index, source in frame.groupby("galaxy_index", sort=True):
        inner = source[source["split"].eq("inner_train")]
        fit = fit_galaxy(
            inner,
            model=model,
            settings=settings,
            starts=starts_for(model, protocol, int(galaxy_index)),
            bounds=nuisance_bounds,
            candidate_parameters=parameters,
            density_geometry=None,
            max_iterations=int(protocol["nuisance_fit"]["max_iterations"]),
        )
        predicted = effective_prediction(
            source,
            fit.theta,
            model=model,
            settings=settings,
            candidate_parameters=parameters,
            density_geometry=None,
        )
        points = source[
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
        profile = pd.DataFrame(
            {
                "system": points["galaxy"],
                "radius_kpc": points["radius_adjusted_kpc"],
                "gbar_m_s2": points["g_bar_m_s2"],
            }
        )
        path = (
            predict_mass_path_completion_frame(profile, model, parameters)
            if model in MASS_PATH_MODELS
            else predict_path_completion_frame(profile, model, parameters)
        )
        for name, values in path.items():
            points[name] = values
        points.insert(0, "model", model)
        blocks.append(points)
        fit_records.append(
            {
                "finite": fit.finite,
                "success": fit.success,
                "at_boundary": any(
                    abs(value - low) <= 1.0e-4 * max(1.0, abs(low))
                    or abs(value - high) <= 1.0e-4 * max(1.0, abs(high))
                    for value, (low, high) in zip(fit.theta, nuisance_bounds, strict=True)
                ),
            }
        )
    points = pd.concat(blocks, ignore_index=True)
    fits = pd.DataFrame(fit_records)
    return points, {
        "inner_train": sparc_metrics(points, "inner_train"),
        "outer_holdout": sparc_metrics(points, "outer_holdout"),
        "finite_fit_fraction": float(fits["finite"].mean()),
        "optimizer_success_fraction": float(fits["success"].mean()),
        "nuisance_boundary_fraction": float(fits["at_boundary"].mean()),
    }


def raw_profile(model: str, parameters: np.ndarray, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(ROOT / protocol["inputs"]["RXJ_profile"])
    cluster = source[
        source["model"].eq("sigma_refracted_AQUAL")
        & source["domain"].eq("RXJ2129")
    ].sort_values("radius_kpc")
    result = path_completion_profile(
        cluster["radius_kpc"].to_numpy(float),
        cluster["gbar_m_s2"].to_numpy(float),
        model,
        parameters,
    )
    lens = pd.DataFrame(
        {
            "domain": "RXJ2129",
            "radius_kpc": cluster["radius_kpc"].to_numpy(float),
            "gSigma_m_s2": result["predicted_acceleration_m_s2"],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "domain": "RXJ2129",
            "system": "RXJ2129",
            "radius_kpc": cluster["radius_kpc"].to_numpy(float),
            "gbar_m_s2": cluster["gbar_m_s2"].to_numpy(float),
            **result,
        }
    )
    return lens, diagnostics


def solar_test(model: str, parameters: np.ndarray, protocol: dict) -> dict:
    settings = protocol["solar_diagnostics"]
    au_m = float(settings["astronomical_unit_m"])
    maximum_au = max(map(float, settings["probe_AU"]))
    radius_m = np.geomspace(float(settings["solar_radius_m"]), maximum_au * au_m, 4000)
    radius_kpc = radius_m / KPC_M
    gbar = float(settings["solar_GM_m3_s2"]) / radius_m**2
    result = path_completion_profile(radius_kpc, gbar, model, parameters)
    records = {}
    for probe in settings["probe_AU"]:
        index = int(np.argmin(np.abs(radius_m / au_m - float(probe))))
        records[f"{float(probe):g}_AU"] = {
            "completion_fraction": float(result["completion_fraction"][index]),
            "enhancement_relative_to_local_G": float(
                result["enhancement_relative_to_local_G"][index]
            ),
            "fractional_change": float(
                result["enhancement_relative_to_local_G"][index] - 1.0
            ),
        }
    return records


def add_slope_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby(["domain", "system"], sort=True):
        ordered = group.sort_values("radius_kpc").copy()
        if len(ordered) >= 2:
            log_radius = np.log(ordered["radius_kpc"].to_numpy(float))
            ordered["effective_acceleration_exponent"] = -np.gradient(
                np.log(ordered["predicted_acceleration_m_s2"].to_numpy(float)),
                log_radius,
            )
            ordered["completion_log_slope"] = np.gradient(
                np.log(ordered["completion_fraction"].to_numpy(float)),
                log_radius,
            )
        else:
            ordered["effective_acceleration_exponent"] = np.nan
            ordered["completion_log_slope"] = np.nan
        pieces.append(ordered)
    return pd.concat(pieces, ignore_index=True)


def slope_summary(frame: pd.DataFrame) -> dict:
    output = {}
    for domain, group in frame.groupby("domain"):
        exponent = group["effective_acceleration_exponent"].dropna()
        completion = group["completion_log_slope"].dropna()
        output[str(domain)] = {
            "rows_with_slope": int(len(exponent)),
            "median_effective_acceleration_exponent": (
                float(exponent.median()) if len(exponent) else None
            ),
            "fraction_with_exponent_between_0p75_and_1p25": (
                float(exponent.between(0.75, 1.25).mean()) if len(exponent) else None
            ),
            "median_completion_log_slope": (
                float(completion.median()) if len(completion) else None
            ),
        }
    return output


def make_figure(report: dict, output: Path) -> None:
    models = list(report["models"])
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    labels = ["MOND", "prior Sigma", *models]
    bridge = [
        report["references"]["bridge"]["simple_MOND"],
        report["references"]["bridge"]["prior_Sigma"],
        *[report["models"][name]["bridge_metrics"]["equal_domain_RMSE_dex"] for name in models],
    ]
    axes[0, 0].bar(labels, bridge, color=["#777777", "#1b9e77", *colors])
    axes[0, 0].set(title="Held-out BCG + cluster bridge", ylabel="equal-domain RMSE (dex)")

    sparc = [
        report["references"]["SPARC"]["fixed_RAR"],
        report["references"]["SPARC"]["prior_Sigma"],
        *[report["models"][name]["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"] for name in models],
    ]
    axes[0, 1].bar(["RAR", "prior Sigma", *models], sparc, color=["#777777", "#1b9e77", *colors])
    axes[0, 1].set(title="SPARC untouched outer radii", ylabel="RMSE (km/s)")

    raw = [
        report["references"]["raw_lensing"]["compact_halo"],
        *[report["models"][name]["raw_lensing"]["heldout"]["exact_radial_RMS_arcsec"] for name in models],
    ]
    axes[1, 0].bar(["compact halo", *models], raw, color=["#444444", *colors])
    axes[1, 0].axhline(1.0, color="black", linestyle="--")
    axes[1, 0].set(title="RX J2129 raw image positions", ylabel="heldout RMS (arcsec)")

    maxima = [report["models"][name]["G_max_over_G_measured"] for name in models]
    earth = [
        abs(report["models"][name]["solar"]["1_AU"]["fractional_change"])
        for name in models
    ]
    bars = axes[1, 1].bar(models, maxima, color=colors)
    axes[1, 1].set(title="Allowed universal maximum", ylabel="G_max / G_measured")
    for bar, change in zip(bars, earth, strict=True):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"Earth Δ={change:.1e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for axis in axes.ravel():
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/path_completion_full_test_protocol.json",
    )
    args = parser.parse_args()
    config_path = args.protocol.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_path_completion_scores":
        raise RuntimeError("path-completion protocol was not frozen before scoring")
    if tuple(protocol["models"]) != PATH_MODELS:
        raise RuntimeError("protocol model order differs from implemented finite family")

    bridge = bridge_frame_with_gbar(
        pd.read_csv(ROOT / protocol["inputs"]["bridge_sample"])
    )
    bridge = assign_group_folds(
        bridge.drop(columns=["fold"], errors="ignore"),
        int(protocol["sample"]["bridge_folds"]),
        int(protocol["sample"]["bridge_seed"]),
    )
    bridge_reference = json.loads(
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
    local_reference = json.loads(
        (ROOT / protocol["inputs"]["local_completion_reference_report"]).read_text()
    )

    results = {}
    parameters = {}
    bridge_tables = []
    profile_tables = []
    for model, specification in protocol["models"].items():
        print(f"cross-validating path model={model}", flush=True)
        prediction, fold_fits, full = cross_validate_bridge(
            bridge, model, specification, protocol
        )
        parameters[model] = full
        _, full_fields = predict_bridge(bridge, model, full)
        table = bridge.copy()
        table["model"] = model
        table["predicted_log_gobs"] = prediction
        table["residual_dex"] = prediction - table["log_gobs"]
        for name, values in full_fields.items():
            table[f"full_fit_{name}"] = values
        bridge_tables.append(table)
        full_profile = bridge[["domain", "system", "radius_kpc", "gbar_m_s2"]].copy()
        for name, values in full_fields.items():
            full_profile[name] = values
        full_profile["model"] = model
        profile_tables.append(add_slope_diagnostics(full_profile))
        flags = dict(
            zip(
                specification["parameters"],
                boundary_flags(
                    full,
                    specification["bounds"],
                    float(protocol["optimization"]["boundary_fraction_tolerance"]),
                ),
                strict=True,
            )
        )
        results[model] = {
            "bridge_metrics": domain_metrics(bridge, prediction),
            "fold_fits": fold_fits,
            "full_fit_parameters": dict(
                zip(specification["parameters"], map(float, full), strict=True)
            ),
            "full_fit_parameter_vector": list(map(float, full)),
            "full_fit_at_boundary": flags,
            "G_max_over_G_measured": float(1.0 / full[0]),
            "bridge_completion": {
                "minimum": float(np.min(full_fields["completion_fraction"])),
                "median": float(np.median(full_fields["completion_fraction"])),
                "maximum": float(np.max(full_fields["completion_fraction"])),
            },
            "solar": solar_test(model, full, protocol),
        }

    bridge_predictions = pd.concat(bridge_tables, ignore_index=True)
    bridge_rank = sorted(
        results,
        key=lambda name: results[name]["bridge_metrics"]["equal_domain_RMSE_dex"],
    )

    print("building frozen SPARC frame", flush=True)
    sparc_source = build_frame(
        sparc_protocol,
        ROOT / protocol["inputs"]["SPARC_raw"],
        ROOT / protocol["inputs"]["SPARC_morphology"],
    )
    sparc_tables = []
    for model in protocol["models"]:
        print(f"SPARC independent nuisance refit model={model}", flush=True)
        points, metrics = run_sparc_transfer(
            model, parameters[model], sparc_protocol, sparc_source
        )
        sparc_tables.append(points)
        results[model]["SPARC_metrics"] = metrics
        profile = pd.DataFrame(
            {
                "domain": np.where(
                    points["split"].eq("outer_holdout"),
                    "SPARC_outer",
                    "SPARC_inner",
                ),
                "system": points["galaxy"],
                "radius_kpc": points["radius_adjusted_kpc"],
                "gbar_m_s2": points["g_bar_m_s2"],
                "tidal_curvature_s2": points["tidal_curvature_s2"],
                "path_weight": points["path_weight"],
                "recovery_optical_depth": points["recovery_optical_depth"],
                "completion_fraction": points["completion_fraction"],
                "enhancement_relative_to_local_G": points[
                    "enhancement_relative_to_local_G"
                ],
                "predicted_acceleration_m_s2": points[
                    "predicted_acceleration_m_s2"
                ],
                "model": model,
            }
        )
        profile_tables.append(add_slope_diagnostics(profile))
    sparc_predictions = pd.concat(sparc_tables, ignore_index=True)

    raw_tables = []
    for model in protocol["models"]:
        print(f"raw RXJ2129 lensing model={model}", flush=True)
        lens_profile, diagnostic = raw_profile(model, parameters[model], protocol)
        predictions, summary = run_diagnostic_lensing(
            pd.Series(results[model]["full_fit_parameters"]),
            protocol,
            lens_profile,
        )
        predictions["path_model"] = model
        raw_tables.append(predictions)
        diagnostic["model"] = model
        profile_tables.append(add_slope_diagnostics(diagnostic))
        results[model]["raw_lensing"] = summary
    raw_predictions = pd.concat(raw_tables, ignore_index=True)
    profile_diagnostics = pd.concat(profile_tables, ignore_index=True)

    references = {
        "bridge": {
            "simple_MOND": bridge_reference["metrics"]["simple_MOND"][
                "equal_domain_RMSE_dex"
            ],
            "prior_Sigma": bridge_reference["metrics"][
                "RAR_sharp_coherence_gated_RG"
            ]["equal_domain_RMSE_dex"],
            "local_completion": local_reference["models"]["isotropic_completion"][
                "bridge_metrics"
            ]["equal_domain_RMSE_dex"],
        },
        "SPARC": {
            "fixed_RAR": sparc_reference["scores"]["fixed_RAR:invariant"][
                "outer_holdout"
            ]["RMSE_km_s"],
            "simple_MOND": sparc_reference["scores"]["simple_MOND:invariant"][
                "outer_holdout"
            ]["RMSE_km_s"],
            "prior_Sigma": sparc_reference["scores"][
                "RAR_sharp_coherence_gated_RG:primary"
            ]["outer_holdout"]["RMSE_km_s"],
            "local_completion": local_reference["models"]["isotropic_completion"][
                "SPARC_metrics"
            ]["outer_holdout"]["RMSE_km_s"],
        },
        "raw_lensing": {
            "zero_slip": raw_reference["raw_lensing"]["zero_slip"]["heldout"][
                "exact_radial_RMS_arcsec"
            ],
            "prior_Sigma_slip": raw_reference["raw_lensing"]["radial_selected"][
                "heldout"
            ]["exact_radial_RMS_arcsec"],
            "compact_halo": raw_reference["raw_lensing"][
                "compact_halo_reference_heldout_RMS_arcsec"
            ],
            "local_completion": local_reference["models"]["isotropic_completion"][
                "raw_lensing"
            ]["heldout"]["exact_radial_RMS_arcsec"],
        },
    }

    gates = protocol["advance_gates"]
    for model, result in results.items():
        bridge_metrics = result["bridge_metrics"]
        outer = result["SPARC_metrics"]["outer_holdout"]
        raw_rms = result["raw_lensing"]["heldout"]["exact_radial_RMS_arcsec"]
        audit = {
            "bridge_equal_domain_pass": bridge_metrics["equal_domain_RMSE_dex"]
            <= float(gates["bridge_equal_domain_RMSE_dex_max"]),
            "BCG_pass": bridge_metrics["BCG"]["equal_system_RMSE_dex"]
            <= float(gates["BCG_equal_system_RMSE_dex_max"]),
            "cluster_pass": bridge_metrics["cluster"]["equal_system_RMSE_dex"]
            <= float(gates["cluster_equal_system_RMSE_dex_max"]),
            "SPARC_transfer_pass": outer["RMSE_km_s"]
            / references["SPARC"]["fixed_RAR"]
            <= float(gates["SPARC_outer_RMSE_relative_to_fixed_RAR_max"]),
            "raw_lensing_pass": raw_rms <= float(gates["raw_heldout_RMS_arcsec_max"]),
            "solar_Earth_pass": abs(result["solar"]["1_AU"]["fractional_change"])
            <= float(gates["solar_Earth_fractional_change_max"]),
            "bounded_completion_pass": result["bridge_completion"]["maximum"]
            <= float(gates["completion_fraction_max"]),
            "full_fit_not_at_boundary_pass": not any(
                result["full_fit_at_boundary"].values()
            ),
        }
        audit["all_frozen_gates_pass"] = all(audit.values())
        result["gate_audit"] = audit
        result_profiles = profile_diagnostics[profile_diagnostics["model"].eq(model)]
        result["profile_slopes"] = slope_summary(result_profiles)

    survivors = [
        model for model in bridge_rank if results[model]["gate_audit"]["all_frozen_gates_pass"]
    ]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed finite path-completion family test",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "root_model": protocol["root_model"],
        "coverage": {
            "bridge_rows": len(bridge),
            "bridge_systems": int(bridge["system"].nunique()),
            "SPARC_galaxies": int(sparc_source["galaxy"].nunique()),
            "SPARC_inner_points": int(sparc_source["split"].eq("inner_train").sum()),
            "SPARC_outer_points": int(sparc_source["split"].eq("outer_holdout").sum()),
            "raw_images_per_model": int(raw_predictions.groupby("path_model").size().iloc[0]),
        },
        "selection": {
            "bridge_rank": bridge_rank,
            "all_gate_survivors": survivors,
            "selected_survivor": survivors[0] if survivors else None,
            "gravity_parameters_fit_to_SPARC": 0,
            "gravity_or_lensing_amplitudes_fit_to_raw_images": 0,
        },
        "models": results,
        "references": references,
        "verdict": {
            "any_universal_path_law_passes": bool(survivors),
            "covariant_action_derived": False,
            "meaning": (
                "At least one predefined bounded path law survived every frozen transfer gate."
                if survivors
                else "No predefined bounded path law simultaneously passed cluster/BCG, galaxy, raw-lensing, Solar, and non-boundary gates."
            ),
        },
        "claim_limits": protocol["claim_limits"],
        "input_hashes": {
            key: sha256(ROOT / value)
            for key, value in protocol["inputs"].items()
            if (ROOT / value).is_file()
        },
        "outputs": protocol["outputs"],
    }

    output_dir = ROOT / Path(protocol["outputs"]["report"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_predictions.to_csv(ROOT / protocol["outputs"]["bridge_predictions"], index=False)
    sparc_predictions.to_csv(ROOT / protocol["outputs"]["SPARC_predictions"], index=False)
    raw_predictions.to_csv(ROOT / protocol["outputs"]["raw_lensing_predictions"], index=False)
    profile_diagnostics.to_csv(ROOT / protocol["outputs"]["profile_diagnostics"], index=False)
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(report, ROOT / protocol["outputs"]["figure"])

    lines = [
        "# Bounded path-completion full test",
        "",
        f"Bridge ranking: `{', '.join(bridge_rank)}`.",
        f"All-gate survivors: `{', '.join(survivors) if survivors else 'none'}`.",
        "",
        "| model | Gmax/Gmeasured | bridge RMSE dex | SPARC outer km/s | raw heldout arcsec | Earth change | all gates |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for model in protocol["models"]:
        result = results[model]
        lines.append(
            f"| {model} | {result['G_max_over_G_measured']:.3f} | "
            f"{result['bridge_metrics']['equal_domain_RMSE_dex']:.4f} | "
            f"{result['SPARC_metrics']['outer_holdout']['RMSE_km_s']:.3f} | "
            f"{result['raw_lensing']['heldout']['exact_radial_RMS_arcsec']:.3f} | "
            f"{result['solar']['1_AU']['fractional_change']:.2e} | "
            f"{result['gate_audit']['all_frozen_gates_pass']} |"
        )
    lines.extend(
        [
            "",
            "Gravity parameters were selected on the BCG/CLASH bridge only. SPARC outer radii and RX J2129 image coordinates were not used to tune them.",
        ]
    )
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
