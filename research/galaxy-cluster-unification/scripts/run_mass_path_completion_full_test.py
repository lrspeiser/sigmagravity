#!/usr/bin/env python3
"""Post-failure mass-dependent bounded path-completion tests."""

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
from run_path_completion_full_test import (  # noqa: E402
    boundary_flags,
    bridge_frame_with_gbar,
    json_safe,
    run_sparc_transfer,
    sha256,
)
from run_sigma_field_exploration import run_diagnostic_lensing  # noqa: E402
from run_sparc_independent_nuisance_refit import build_frame  # noqa: E402
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.path_completion import (  # noqa: E402
    MASS_PATH_MODELS,
    mass_path_completion_profile,
    predict_mass_path_completion_frame,
)


def predict_bridge(frame: pd.DataFrame, model: str, parameters) -> tuple[np.ndarray, dict]:
    fields = predict_mass_path_completion_frame(frame, model, parameters)
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


def cross_validate(
    frame: pd.DataFrame, model: str, specification: dict, protocol: dict
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    folds = int(protocol["sample"]["bridge_folds"])
    seed = int(protocol["sample"]["bridge_seed"])
    names = specification["parameters"]
    bounds = specification["bounds"]
    tolerance = float(protocol["optimization"]["boundary_fraction_tolerance"])
    prediction = np.full(len(frame), np.nan)
    records = []
    for fold in range(folds):
        training = frame[frame["fold"] != fold]
        heldout = frame[frame["fold"] == fold]
        values = fit_bridge(training, model, bounds, protocol, seed + fold)
        prediction[heldout.index] = predict_bridge(heldout, model, values)[0]
        records.append(
            {
                "fold": fold,
                "parameters": dict(zip(names, map(float, values), strict=True)),
                "at_boundary": dict(
                    zip(names, boundary_flags(values, bounds, tolerance), strict=True)
                ),
            }
        )
    full = fit_bridge(frame, model, bounds, protocol, seed + 100)
    return prediction, records, full


def raw_profile(model: str, parameters: np.ndarray, protocol: dict) -> pd.DataFrame:
    source = pd.read_csv(ROOT / protocol["inputs"]["RXJ_profile"])
    cluster = source[
        source["model"].eq("sigma_refracted_AQUAL")
        & source["domain"].eq("RXJ2129")
    ].sort_values("radius_kpc")
    result = mass_path_completion_profile(
        cluster["radius_kpc"].to_numpy(float),
        cluster["gbar_m_s2"].to_numpy(float),
        model,
        parameters,
    )
    return pd.DataFrame(
        {
            "domain": "RXJ2129",
            "radius_kpc": cluster["radius_kpc"].to_numpy(float),
            "gSigma_m_s2": result["predicted_acceleration_m_s2"],
        }
    )


def solar_test(model: str, parameters: np.ndarray, protocol: dict) -> dict:
    settings = protocol["solar_diagnostics"]
    au = float(settings["astronomical_unit_m"])
    maximum = max(map(float, settings["probe_AU"])) * au
    radius_m = np.geomspace(float(settings["solar_radius_m"]), maximum, 4000)
    radius_kpc = radius_m / KPC_M
    gbar = float(settings["solar_GM_m3_s2"]) / radius_m**2
    result = mass_path_completion_profile(radius_kpc, gbar, model, parameters)
    output = {}
    for probe in settings["probe_AU"]:
        index = int(np.argmin(np.abs(radius_m / au - float(probe))))
        enhancement = float(result["enhancement_relative_to_local_G"][index])
        output[f"{float(probe):g}_AU"] = {
            "completion_fraction": float(result["completion_fraction"][index]),
            "enhancement_relative_to_local_G": enhancement,
            "fractional_change": enhancement - 1.0,
        }
    return output


def make_figure(report: dict, output: Path) -> None:
    models = list(report["models"])
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    bridge = [report["models"][name]["bridge_metrics"]["equal_domain_RMSE_dex"] for name in models]
    sparc = [report["models"][name]["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"] for name in models]
    raw = [report["models"][name]["raw_lensing"]["heldout"]["exact_radial_RMS_arcsec"] for name in models]
    axes[0].bar(models, bridge, color=colors)
    axes[0].axhline(report["references"]["prior_Sigma_bridge"], color="black", linestyle="--")
    axes[0].set(title="Held-out BCG + cluster", ylabel="RMSE (dex)")
    axes[1].bar(models, sparc, color=colors)
    axes[1].axhline(report["references"]["fixed_RAR_SPARC"], color="black", linestyle="--")
    axes[1].set(title="SPARC untouched outer radii", ylabel="RMSE (km/s)")
    axes[2].bar(models, raw, color=colors)
    axes[2].axhline(report["references"]["compact_halo_raw"], color="black", linestyle="--")
    axes[2].axhline(1.0, color="grey", linestyle=":")
    axes[2].set(title="RX J2129 raw images", ylabel="heldout RMS (arcsec)")
    for axis in axes:
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
        default=ROOT / "configs/mass_path_completion_full_test_protocol.json",
    )
    args = parser.parse_args()
    config_path = args.protocol.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "post_path_failure_frozen_before_mass_path_scores":
        raise RuntimeError("mass path protocol has wrong freeze status")
    if tuple(protocol["models"]) != MASS_PATH_MODELS:
        raise RuntimeError("mass path finite family changed")

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

    results = {}
    full_parameters = {}
    bridge_tables = []
    for model, specification in protocol["models"].items():
        print(f"cross-validating mass path model={model}", flush=True)
        prediction, folds, full = cross_validate(bridge, model, specification, protocol)
        full_parameters[model] = full
        _, fields = predict_bridge(bridge, model, full)
        table = bridge.copy()
        table["model"] = model
        table["predicted_log_gobs"] = prediction
        table["residual_dex"] = prediction - table["log_gobs"]
        for name, values in fields.items():
            table[f"full_fit_{name}"] = values
        bridge_tables.append(table)
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
            "fold_fits": folds,
            "full_fit_parameters": dict(
                zip(specification["parameters"], map(float, full), strict=True)
            ),
            "full_fit_parameter_vector": list(map(float, full)),
            "full_fit_at_boundary": flags,
            "G_max_over_G_measured": float(1.0 / full[0]),
            "bridge_completion_maximum": float(np.max(fields["completion_fraction"])),
            "solar": solar_test(model, full, protocol),
        }
    bridge_predictions = pd.concat(bridge_tables, ignore_index=True)

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
            model, full_parameters[model], sparc_protocol, sparc_source
        )
        sparc_tables.append(points)
        results[model]["SPARC_metrics"] = metrics
    sparc_predictions = pd.concat(sparc_tables, ignore_index=True)

    raw_tables = []
    for model in protocol["models"]:
        print(f"raw RXJ2129 lensing model={model}", flush=True)
        profile = raw_profile(model, full_parameters[model], protocol)
        predictions, summary = run_diagnostic_lensing(
            pd.Series(results[model]["full_fit_parameters"]), protocol, profile
        )
        predictions["mass_path_model"] = model
        raw_tables.append(predictions)
        results[model]["raw_lensing"] = summary
    raw_predictions = pd.concat(raw_tables, ignore_index=True)

    references = {
        "prior_Sigma_bridge": bridge_reference["metrics"][
            "RAR_sharp_coherence_gated_RG"
        ]["equal_domain_RMSE_dex"],
        "fixed_RAR_SPARC": sparc_reference["scores"]["fixed_RAR:invariant"][
            "outer_holdout"
        ]["RMSE_km_s"],
        "compact_halo_raw": raw_reference["raw_lensing"][
            "compact_halo_reference_heldout_RMS_arcsec"
        ],
    }
    gates = protocol["advance_gates"]
    for result in results.values():
        bridge_metrics = result["bridge_metrics"]
        audit = {
            "bridge_equal_domain_pass": bridge_metrics["equal_domain_RMSE_dex"]
            <= float(gates["bridge_equal_domain_RMSE_dex_max"]),
            "BCG_pass": bridge_metrics["BCG"]["equal_system_RMSE_dex"]
            <= float(gates["BCG_equal_system_RMSE_dex_max"]),
            "cluster_pass": bridge_metrics["cluster"]["equal_system_RMSE_dex"]
            <= float(gates["cluster_equal_system_RMSE_dex_max"]),
            "SPARC_transfer_pass": result["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"]
            / references["fixed_RAR_SPARC"]
            <= float(gates["SPARC_outer_RMSE_relative_to_fixed_RAR_max"]),
            "raw_lensing_pass": result["raw_lensing"]["heldout"]["exact_radial_RMS_arcsec"]
            <= float(gates["raw_heldout_RMS_arcsec_max"]),
            "solar_Earth_pass": abs(result["solar"]["1_AU"]["fractional_change"])
            <= float(gates["solar_Earth_fractional_change_max"]),
            "bounded_completion_pass": result["bridge_completion_maximum"]
            <= float(gates["completion_fraction_max"]),
            "full_fit_not_at_boundary_pass": not any(result["full_fit_at_boundary"].values()),
        }
        audit["all_frozen_gates_pass"] = all(audit.values())
        result["gate_audit"] = audit

    rank = sorted(
        results,
        key=lambda name: results[name]["bridge_metrics"]["equal_domain_RMSE_dex"],
    )
    survivors = [name for name in rank if results[name]["gate_audit"]["all_frozen_gates_pass"]]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed post-failure mass path family test",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "motivation": protocol["motivation"],
        "root_model": protocol["root_model"],
        "coverage": {
            "bridge_rows": len(bridge),
            "bridge_systems": int(bridge["system"].nunique()),
            "SPARC_galaxies": int(sparc_source["galaxy"].nunique()),
            "SPARC_inner_points": int(sparc_source["split"].eq("inner_train").sum()),
            "SPARC_outer_points": int(sparc_source["split"].eq("outer_holdout").sum()),
            "raw_images_per_model": int(raw_predictions.groupby("mass_path_model").size().iloc[0]),
        },
        "selection": {
            "bridge_rank": rank,
            "all_gate_survivors": survivors,
            "selected_survivor": survivors[0] if survivors else None,
            "parameters_fit_to_SPARC": 0,
            "parameters_fit_to_raw_images": 0,
        },
        "models": results,
        "references": references,
        "verdict": {
            "any_mass_path_survives": bool(survivors),
            "covariant_action_derived": False,
        },
        "claim_limits": protocol["claim_limits"],
        "outputs": protocol["outputs"],
    }

    output_dir = ROOT / Path(protocol["outputs"]["report"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_predictions.to_csv(ROOT / protocol["outputs"]["bridge_predictions"], index=False)
    sparc_predictions.to_csv(ROOT / protocol["outputs"]["SPARC_predictions"], index=False)
    raw_predictions.to_csv(ROOT / protocol["outputs"]["raw_lensing_predictions"], index=False)
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(report, ROOT / protocol["outputs"]["figure"])
    lines = [
        "# Post-failure mass-path completion test",
        "",
        f"Bridge ranking: `{', '.join(rank)}`.",
        f"All-gate survivors: `{', '.join(survivors) if survivors else 'none'}`.",
        "",
        "| model | Gmax/Gmeasured | bridge dex | SPARC km/s | raw arcsec | all gates |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for model in protocol["models"]:
        result = results[model]
        lines.append(
            f"| {model} | {result['G_max_over_G_measured']:.3f} | "
            f"{result['bridge_metrics']['equal_domain_RMSE_dex']:.4f} | "
            f"{result['SPARC_metrics']['outer_holdout']['RMSE_km_s']:.3f} | "
            f"{result['raw_lensing']['heldout']['exact_radial_RMS_arcsec']:.3f} | "
            f"{result['gate_audit']['all_frozen_gates_pass']} |"
        )
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
