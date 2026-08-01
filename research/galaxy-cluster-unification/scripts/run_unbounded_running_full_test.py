#!/usr/bin/env python3
"""Broad Cassini-screened test of slowly unbounded effective-G laws."""

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
from run_sparc_independent_nuisance_refit import build_frame  # noqa: E402
from run_vector_completion_full_test import (  # noqa: E402
    at_boundary,
    json_safe,
    raw_lensing_profile,
    run_sparc_transfer,
)

from voidscreen.tensor_completion import spherical_tidal_eigenvalues  # noqa: E402
from voidscreen.unbounded_running import (  # noqa: E402
    RUNNING_MODELS,
    TENSOR_RUNNING_MODELS,
    VARIABLE_EXPONENT_DENSITY_MODELS,
    point_mass_scale_diagnostics,
    predict_running_acceleration,
    solar_system_diagnostics,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_bridge(frame: pd.DataFrame, model: str, parameters) -> np.ndarray:
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(float))
    eigenvalues = None
    if model in TENSOR_RUNNING_MODELS:
        eigenvalues = spherical_tidal_eigenvalues(
            gbar,
            frame["radius_kpc"].to_numpy(float),
            frame["local_density_g_cm3"].to_numpy(float),
        )
    result = predict_running_acceleration(
        gbar,
        frame["radius_kpc"].to_numpy(float),
        model,
        parameters,
        tidal_eigenvalues_s2=eigenvalues,
        local_density_g_cm3=(
            frame["local_density_g_cm3"].to_numpy(float)
            if model in VARIABLE_EXPONENT_DENSITY_MODELS
            else None
        ),
    )
    return np.log10(result["predicted_acceleration_m_s2"])


def solar_penalty(model: str, parameters, protocol: dict) -> tuple[float, dict]:
    gates = protocol["solar_gates"]
    cassini_limit = float(gates["maximum_fractional_coupling_change_limb_to_Saturn"])
    earth_limit = float(gates["maximum_Earth_orbit_fractional_change"])
    diagnostic = solar_system_diagnostics(
        model,
        parameters,
        cassini_limit=cassini_limit,
    )
    cassini_excess = max(
        0.0,
        diagnostic["maximum_fractional_change_limb_to_Saturn"] / cassini_limit - 1.0,
    )
    earth_excess = max(
        0.0,
        abs(diagnostic["Earth_orbit_fractional_change"]) / earth_limit - 1.0,
    )
    return cassini_excess**2 + earth_excess**2, diagnostic


def fit_bridge(frame: pd.DataFrame, model: str, specification: dict, protocol: dict, seed: int):
    bounds = list(map(tuple, specification["bounds"]))
    coefficient = float(protocol["optimization"]["Cassini_violation_penalty"])

    def objective(values):
        try:
            penalty, _ = solar_penalty(model, values, protocol)
            prediction = predict_bridge(frame, model, values)
            fit_loss = equal_group_domain_mse(frame, prediction)
            return fit_loss + coefficient * penalty
        except (FloatingPointError, OverflowError, ValueError):
            return 1.0e100

    settings = protocol["optimization"]
    global_fit = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=int(settings["differential_evolution_maxiter"]),
        popsize=int(settings["differential_evolution_popsize"]),
        tol=1.0e-9,
        polish=False,
        workers=1,
    )
    local = minimize(
        objective,
        global_fit.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 8000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    return np.asarray(local.x if local.success else global_fit.x, dtype=float)


def cross_validate(frame: pd.DataFrame, model: str, specification: dict, protocol: dict):
    predictions = np.full(len(frame), np.nan)
    records = []
    for fold in range(int(protocol["sample"]["bridge_folds"])):
        training = frame[frame["fold"] != fold]
        heldout = frame[frame["fold"] == fold]
        parameters = fit_bridge(
            training,
            model,
            specification,
            protocol,
            int(protocol["sample"]["bridge_seed"]) + fold,
        )
        predictions[heldout.index] = predict_bridge(heldout, model, parameters)
        _, solar = solar_penalty(model, parameters, protocol)
        records.append(
            {
                "fold": fold,
                "parameters": dict(
                    zip(specification["parameters"], map(float, parameters), strict=True)
                ),
                "solar": solar,
            }
        )
    full = fit_bridge(
        frame,
        model,
        specification,
        protocol,
        int(protocol["sample"]["bridge_seed"]) + 100,
    )
    return predictions, records, full


def choose_raw_candidates(results: dict, references: dict, count: int = 3) -> list[str]:
    bridge_ref = references["bridge"]["prior_Sigma"]["equal_domain_RMSE_dex"]
    sparc_ref = references["SPARC"]["fixed_RAR_outer_RMSE_km_s"]
    names = list(results)
    best_bridge = min(names, key=lambda name: results[name]["bridge_metrics"]["equal_domain_RMSE_dex"])
    best_sparc = min(names, key=lambda name: results[name]["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"])
    best_consistency = min(
        names,
        key=lambda name: max(
            results[name]["bridge_metrics"]["equal_domain_RMSE_dex"] / bridge_ref,
            results[name]["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"] / sparc_ref,
        ),
    )
    selected = []
    for name in [best_bridge, best_sparc, best_consistency]:
        if name not in selected:
            selected.append(name)
    ranked = sorted(
        names,
        key=lambda name: math.sqrt(
            results[name]["bridge_metrics"]["equal_domain_RMSE_dex"]
            / bridge_ref
            * results[name]["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"]
            / sparc_ref
        ),
    )
    for name in ranked:
        if name not in selected:
            selected.append(name)
        if len(selected) >= count:
            break
    return selected[:count]


def make_figure(report: dict, output: Path) -> None:
    names = list(report["models"])
    labels = [name.replace("tensor_", "t:").replace("curvature_", "c:").replace("_running", "") for name in names]
    colors = [plt.get_cmap("tab10")(index % 10) for index in range(len(names))]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    bridge = [report["models"][name]["bridge_metrics"]["equal_domain_RMSE_dex"] for name in names]
    sparc = [report["models"][name]["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"] for name in names]
    solar = [report["models"][name]["solar"]["maximum_fractional_change_limb_to_Saturn"] for name in names]
    axes[0, 0].bar(labels, bridge, color=colors)
    axes[0, 0].axhline(report["references"]["bridge"]["prior_Sigma"]["equal_domain_RMSE_dex"], color="black", linestyle="--", label="prior Sigma")
    axes[0, 0].set(title="Held-out BCG + cluster", ylabel="RMSE (dex)")
    axes[0, 0].legend()
    axes[0, 1].bar(labels, sparc, color=colors)
    axes[0, 1].axhline(report["references"]["SPARC"]["fixed_RAR_outer_RMSE_km_s"], color="black", linestyle="--", label="RAR")
    axes[0, 1].axhline(report["references"]["SPARC"]["NFW_outer_RMSE_km_s"], color="gray", linestyle=":", label="NFW")
    axes[0, 1].set(title="SPARC untouched outer radii", ylabel="RMSE (km/s)")
    axes[0, 1].legend()
    axes[1, 0].bar(labels, np.maximum(solar, 1.0e-20), color=colors)
    axes[1, 0].axhline(2.3e-5, color="black", linestyle="--", label="Cassini proxy gate")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Solar-System coupling change", ylabel="maximum |G_eff/G_local - 1|")
    axes[1, 0].legend()
    axes[1, 1].scatter(bridge, sparc, c=colors, s=75)
    for name, x, y in zip(labels, bridge, sparc, strict=True):
        axes[1, 1].annotate(name, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    axes[1, 1].set(title="Galaxy-cluster consistency", xlabel="bridge RMSE (dex)", ylabel="SPARC outer RMSE (km/s)")
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    for axis in axes[:2].ravel():
        axis.tick_params(axis="x", rotation=35, labelsize=8)
    axes[1, 0].tick_params(axis="x", rotation=35, labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/unbounded_running_full_test_protocol.json",
        help="Protocol path relative to the research root",
    )
    arguments = parser.parse_args()
    config_path = ROOT / arguments.config
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_before_"):
        raise RuntimeError("unbounded protocol was not frozen before scoring")
    unknown_models = set(protocol["models"]) - RUNNING_MODELS
    if unknown_models:
        raise RuntimeError(f"unknown running models in protocol: {sorted(unknown_models)}")

    sample = pd.read_csv(ROOT / protocol["inputs"]["bridge_sample"])
    sample = assign_group_folds(
        sample.drop(columns=["fold"], errors="ignore"),
        int(protocol["sample"]["bridge_folds"]),
        int(protocol["sample"]["bridge_seed"]),
    )
    bridge_reference = json.loads((ROOT / protocol["inputs"]["bridge_reference_report"]).read_text())
    sparc_protocol = json.loads((ROOT / protocol["inputs"]["SPARC_protocol"]).read_text())
    sparc_reference = json.loads((ROOT / protocol["inputs"]["SPARC_reference_report"]).read_text())
    raw_reference = json.loads((ROOT / protocol["inputs"]["raw_lensing_reference_report"]).read_text())
    references = {
        "bridge": {
            "prior_Sigma": bridge_reference["metrics"]["RAR_sharp_coherence_gated_RG"],
            "Newtonian": bridge_reference["metrics"]["Newtonian"],
            "simple_MOND": bridge_reference["metrics"]["simple_MOND"],
        },
        "SPARC": {
            "fixed_RAR_outer_RMSE_km_s": sparc_reference["scores"]["fixed_RAR:invariant"]["outer_holdout"]["RMSE_km_s"],
            "NFW_outer_RMSE_km_s": sparc_reference["scores"]["NFW:invariant"]["outer_holdout"]["RMSE_km_s"],
        },
        "raw_lensing": {
            "compact_halo_heldout_RMS_arcsec": raw_reference["raw_lensing"]["compact_halo_reference_heldout_RMS_arcsec"]
        },
    }

    results = {}
    full_parameters = {}
    bridge_tables = []
    for model, specification in protocol["models"].items():
        print(f"bridge cross-validation model={model}", flush=True)
        heldout, fold_fits, parameters = cross_validate(sample, model, specification, protocol)
        full_parameters[model] = parameters
        table = sample.copy()
        table.insert(0, "model", model)
        table["predicted_log_gobs"] = heldout
        table["residual_dex"] = heldout - table["log_gobs"]
        bridge_tables.append(table)
        _, solar = solar_penalty(model, parameters, protocol)
        boundary = dict(zip(specification["parameters"], at_boundary(parameters, specification["bounds"], float(protocol["optimization"]["boundary_fraction_tolerance"])), strict=True))
        results[model] = {
            "bridge_metrics": domain_metrics(sample, heldout),
            "fold_fits": fold_fits,
            "full_fit_parameters": dict(zip(specification["parameters"], map(float, parameters), strict=True)),
            "full_fit_parameter_vector": list(map(float, parameters)),
            "full_fit_at_boundary": boundary,
            "solar": solar,
            "scale_extrapolation": point_mass_scale_diagnostics(model, parameters),
        }

    print("building SPARC transfer frame", flush=True)
    sparc_frame = build_frame(sparc_protocol, ROOT / protocol["inputs"]["SPARC_raw"], ROOT / protocol["inputs"]["SPARC_morphology"])
    sparc_tables = []
    for model, parameters in full_parameters.items():
        print(f"SPARC nuisance refit model={model}", flush=True)
        points, metrics = run_sparc_transfer(model, parameters, sparc_protocol, sparc_frame)
        sparc_tables.append(points)
        results[model]["SPARC_metrics"] = metrics

    raw_models = choose_raw_candidates(results, references, count=3)
    raw_tables = []
    for model in raw_models:
        print(f"raw RXJ2129 diagnostic model={model}", flush=True)
        profile = raw_lensing_profile(model, full_parameters[model], protocol)
        predictions, summary = run_diagnostic_lensing(pd.Series(results[model]["full_fit_parameters"]), protocol, profile)
        predictions["running_model"] = model
        raw_tables.append(predictions)
        results[model]["raw_lensing"] = summary
    for model in results:
        results[model].setdefault("raw_lensing", {"status": "not run; outside three exploratory compromises"})

    bridge_ref = references["bridge"]["prior_Sigma"]["equal_domain_RMSE_dex"]
    galaxy_ref = references["SPARC"]["fixed_RAR_outer_RMSE_km_s"]
    gates = protocol["advance_gates"]
    bridge_limit = float(gates["bridge_equal_domain_RMSE_dex_max"])
    bcg_limit = float(gates["BCG_equal_system_RMSE_dex_max"])
    cluster_limit = float(gates["cluster_equal_system_RMSE_dex_max"])
    if "SPARC_outer_RMSE_km_s_max" in gates:
        sparc_limit = float(gates["SPARC_outer_RMSE_km_s_max"])
    else:
        sparc_limit = galaxy_ref * float(
            gates["SPARC_outer_RMSE_relative_to_fixed_RAR_max"]
        )
    earth_limit = float(protocol["solar_gates"]["maximum_Earth_orbit_fractional_change"])
    for model, result in results.items():
        bridge = result["bridge_metrics"]
        galaxy = result["SPARC_metrics"]["outer_holdout"]
        result["consistency_score_max_reference_ratio"] = max(bridge["equal_domain_RMSE_dex"] / bridge_ref, galaxy["RMSE_km_s"] / galaxy_ref)
        result["gate_audit"] = {
            "bridge_pass": bridge["equal_domain_RMSE_dex"] <= bridge_limit,
            "BCG_pass": bridge["BCG"]["equal_system_RMSE_dex"] <= bcg_limit,
            "cluster_pass": bridge["cluster"]["equal_system_RMSE_dex"] <= cluster_limit,
            "SPARC_pass": galaxy["RMSE_km_s"] <= sparc_limit,
            "Cassini_pass": result["solar"]["Cassini_pass"],
            "Earth_pass": abs(result["solar"]["Earth_orbit_fractional_change"])
            <= earth_limit,
            "not_at_boundary": not any(result["full_fit_at_boundary"].values()),
        }
        result["gate_audit"]["all_primary_gates_pass"] = all(result["gate_audit"].values())

    ranking = sorted(results, key=lambda name: results[name]["consistency_score_max_reference_ratio"])
    report = {
        "report_version": protocol["protocol_version"],
        "status": f"completed {protocol['family_label']} test",
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(config_path)},
        "coverage": {"bridge_rows": len(sample), "bridge_systems": int(sample.system.nunique()), "SPARC_galaxies": int(sparc_frame.galaxy.nunique()), "SPARC_inner_points": int(sparc_frame.split.eq("inner_train").sum()), "SPARC_outer_points": int(sparc_frame.split.eq("outer_holdout").sum()), "raw_models": raw_models},
        "selection": {"bridge_only_best": min(results, key=lambda name: results[name]["bridge_metrics"]["equal_domain_RMSE_dex"]), "post_transfer_consistency_ranking": ranking, "post_transfer_warning": "exploratory ranking; SPARC was inspected"},
        "models": results,
        "references": references,
        "solar_gate_interpretation": protocol["solar_gates"]["interpretation"],
        "verdict": {"any_universal_survivor": any(result["gate_audit"]["all_primary_gates_pass"] for result in results.values()), "best_consistency_model": ranking[0]},
    }
    output = ROOT / Path(protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(bridge_tables, ignore_index=True).to_csv(ROOT / protocol["outputs"]["bridge_predictions"], index=False)
    pd.concat(sparc_tables, ignore_index=True).to_csv(ROOT / protocol["outputs"]["SPARC_predictions"], index=False)
    if raw_tables:
        pd.concat(raw_tables, ignore_index=True).to_csv(ROOT / protocol["outputs"]["raw_lensing_predictions"], index=False)
    (ROOT / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    make_figure(report, ROOT / protocol["outputs"]["figure"])
    lines = [f"# {protocol['family_label']}", "", "All universal parameters were fit on the BCG+cluster bridge; SPARC was transfer-only.", "", "| rank | model | bridge (dex) | SPARC outer (km/s) | Cassini change | boundary |", "|---:|---|---:|---:|---:|---|"]
    for index, model in enumerate(ranking, 1):
        result = results[model]
        boundary = ", ".join(name for name, hit in result["full_fit_at_boundary"].items() if hit) or "none"
        lines.append(f"| {index} | {model} | {result['bridge_metrics']['equal_domain_RMSE_dex']:.4f} | {result['SPARC_metrics']['outer_holdout']['RMSE_km_s']:.3f} | {result['solar']['maximum_fractional_change_limb_to_Saturn']:.2e} | {boundary} |")
    lines.extend(["", f"Any universal survivor: **{report['verdict']['any_universal_survivor']}**.", f"Best exploratory consistency: **{ranking[0]}**."])
    (ROOT / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
