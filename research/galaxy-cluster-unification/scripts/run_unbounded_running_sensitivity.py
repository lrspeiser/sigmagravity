#!/usr/bin/env python3
"""Fixed-shape sensitivity map for the strongest unbounded running laws."""

from __future__ import annotations

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
from run_sparc_independent_nuisance_refit import build_frame  # noqa: E402
from run_unbounded_running_full_test import (  # noqa: E402
    json_safe,
    predict_bridge,
    solar_penalty,
)
from run_vector_completion_full_test import run_sparc_transfer  # noqa: E402


def expand_parameters(free_values, family: dict, fixed_value: float) -> np.ndarray:
    values = np.empty(len(family["parameters"]), dtype=float)
    values[int(family["fixed_index"])] = float(fixed_value)
    values[np.asarray(family["free_indices"], dtype=int)] = np.asarray(free_values, dtype=float)
    return values


def fit_variant(frame, model, family, fixed_value, protocol, seed):
    bounds = list(map(tuple, family["free_bounds"]))
    penalty_coefficient = float(protocol["optimization"]["Cassini_violation_penalty"])

    def objective(free_values):
        values = expand_parameters(free_values, family, fixed_value)
        try:
            penalty, _ = solar_penalty(model, values, protocol)
            return equal_group_domain_mse(frame, predict_bridge(frame, model, values)) + penalty_coefficient * penalty
        except (FloatingPointError, OverflowError, ValueError):
            return 1.0e100

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=int(protocol["optimization"]["differential_evolution_maxiter"]),
        popsize=int(protocol["optimization"]["differential_evolution_popsize"]),
        polish=False,
        workers=1,
        tol=1.0e-8,
    )
    local = minimize(
        objective,
        result.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1.0e-14, "gtol": 1.0e-9},
    )
    return expand_parameters(local.x if local.success else result.x, family, fixed_value)


def main() -> None:
    config_path = ROOT / "configs/unbounded_running_sensitivity_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_sensitivity_scores":
        raise RuntimeError("sensitivity protocol must be frozen before scoring")

    sample = pd.read_csv(ROOT / protocol["inputs"]["bridge_sample"])
    sample = assign_group_folds(sample.drop(columns=["fold"], errors="ignore"), int(protocol["bridge_folds"]), int(protocol["bridge_seed"]))
    sparc_protocol = json.loads((ROOT / protocol["inputs"]["SPARC_protocol"]).read_text())
    sparc_frame = build_frame(
        sparc_protocol,
        ROOT / protocol["inputs"]["SPARC_raw"],
        ROOT / protocol["inputs"]["SPARC_morphology"],
    )
    bridge_reference = json.loads((ROOT / protocol["inputs"]["bridge_reference_report"]).read_text())
    sparc_reference = json.loads((ROOT / protocol["inputs"]["SPARC_reference_report"]).read_text())
    references = {
        "prior_Sigma_bridge_RMSE_dex": bridge_reference["metrics"]["RAR_sharp_coherence_gated_RG"]["equal_domain_RMSE_dex"],
        "RAR_SPARC_outer_RMSE_km_s": sparc_reference["scores"]["fixed_RAR:invariant"]["outer_holdout"]["RMSE_km_s"],
        "NFW_SPARC_outer_RMSE_km_s": sparc_reference["scores"]["NFW:invariant"]["outer_holdout"]["RMSE_km_s"],
    }

    records = []
    bridge_tables = []
    for family_number, (model, family) in enumerate(protocol["families"].items()):
        for value_number, fixed_value in enumerate(family["fixed_values"]):
            variant = f"{model}:{family['fixed_name']}={fixed_value:g}"
            print(f"cross-validation {variant}", flush=True)
            heldout = np.full(len(sample), np.nan)
            fold_parameters = []
            for fold in range(int(protocol["bridge_folds"])):
                parameters = fit_variant(
                    sample[sample.fold != fold], model, family, fixed_value, protocol,
                    int(protocol["bridge_seed"]) + 1000 * family_number + 20 * value_number + fold,
                )
                heldout[sample.fold == fold] = predict_bridge(sample[sample.fold == fold], model, parameters)
                fold_parameters.append(parameters.tolist())
            parameters = fit_variant(
                sample, model, family, fixed_value, protocol,
                int(protocol["bridge_seed"]) + 1000 * family_number + 20 * value_number + 10,
            )
            metrics = domain_metrics(sample, heldout)
            _, solar = solar_penalty(model, parameters, protocol)
            print(f"SPARC transfer {variant}", flush=True)
            _, sparc = run_sparc_transfer(model, parameters, sparc_protocol, sparc_frame)
            row = {
                "variant": variant,
                "model": model,
                "fixed_name": family["fixed_name"],
                "fixed_value": float(fixed_value),
                "bridge_RMSE_dex": metrics["equal_domain_RMSE_dex"],
                "BCG_RMSE_dex": metrics["BCG"]["equal_system_RMSE_dex"],
                "cluster_RMSE_dex": metrics["cluster"]["equal_system_RMSE_dex"],
                "SPARC_outer_RMSE_km_s": sparc["outer_holdout"]["RMSE_km_s"],
                "SPARC_outer_bias_sigma": sparc["outer_holdout"]["mean_standardized_residual"],
                "Cassini_maximum_change": solar["maximum_fractional_change_limb_to_Saturn"],
                "Earth_orbit_change": abs(solar["Earth_orbit_enhancement"] - 1.0),
                "parameters": dict(zip(family["parameters"], map(float, parameters), strict=True)),
                "fold_parameters": fold_parameters,
            }
            row["consistency_max_reference_ratio"] = max(
                row["bridge_RMSE_dex"] / references["prior_Sigma_bridge_RMSE_dex"],
                row["SPARC_outer_RMSE_km_s"] / references["RAR_SPARC_outer_RMSE_km_s"],
            )
            records.append(row)
            table = sample[["domain", "system", "radius_kpc", "log_gbar", "log_gobs", "fold"]].copy()
            table.insert(0, "variant", variant)
            table["predicted_log_gobs"] = heldout
            table["residual_dex"] = heldout - table.log_gobs
            bridge_tables.append(table)

    for candidate in records:
        candidate["Pareto_nondominated"] = not any(
            other["bridge_RMSE_dex"] <= candidate["bridge_RMSE_dex"]
            and other["SPARC_outer_RMSE_km_s"] <= candidate["SPARC_outer_RMSE_km_s"]
            and (other["bridge_RMSE_dex"] < candidate["bridge_RMSE_dex"] or other["SPARC_outer_RMSE_km_s"] < candidate["SPARC_outer_RMSE_km_s"])
            for other in records
        )
    ranking = sorted(records, key=lambda row: row["consistency_max_reference_ratio"])
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed fixed-shape sensitivity map",
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "coverage": {"variants": len(records), "bridge_rows": len(sample), "bridge_systems": int(sample.system.nunique()), "SPARC_galaxies": int(sparc_frame.galaxy.nunique()), "SPARC_outer_points": int(sparc_frame.split.eq("outer_holdout").sum())},
        "references": references,
        "ranking": [row["variant"] for row in ranking],
        "Pareto_front": [row["variant"] for row in records if row["Pareto_nondominated"]],
        "variants": records,
        "verdict": {"best_consistency_variant": ranking[0]["variant"], "any_RAR_level_and_prior_Sigma_level": any(row["bridge_RMSE_dex"] <= references["prior_Sigma_bridge_RMSE_dex"] and row["SPARC_outer_RMSE_km_s"] <= references["RAR_SPARC_outer_RMSE_km_s"] for row in records)},
    }
    output = (ROOT / protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{key: value for key, value in row.items() if key not in {"parameters", "fold_parameters"}} for row in records]).to_csv(ROOT / protocol["outputs"]["scores"], index=False)
    pd.concat(bridge_tables, ignore_index=True).to_csv(ROOT / protocol["outputs"]["bridge_predictions"], index=False)

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = {"curvature_power": "tab:red", "curvature_additive_power": "tab:blue"}
    for model in protocol["families"]:
        family_rows = sorted((row for row in records if row["model"] == model), key=lambda row: row["fixed_value"])
        x = [row["bridge_RMSE_dex"] for row in family_rows]
        y = [row["SPARC_outer_RMSE_km_s"] for row in family_rows]
        axis.plot(x, y, "o-", color=colors[model], label=model)
        for row in family_rows:
            axis.annotate(f"{row['fixed_value']:g}", (row["bridge_RMSE_dex"], row["SPARC_outer_RMSE_km_s"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axis.axvline(references["prior_Sigma_bridge_RMSE_dex"], color="black", linestyle="--", label="prior Sigma bridge")
    axis.axhline(references["RAR_SPARC_outer_RMSE_km_s"], color="black", linestyle=":", label="RAR galaxy")
    axis.axhline(references["NFW_SPARC_outer_RMSE_km_s"], color="gray", linestyle=":", label="NFW galaxy")
    axis.set(xlabel="held-out BCG+cluster RMSE (dex)", ylabel="SPARC outer RMSE (km/s)", title="Fixed-shape sensitivity: lower left is better")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(ROOT / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    lines = ["# Unbounded running sensitivity map", "", f"Tested {len(records)} fixed-shape values with five-fold bridge fits and locked SPARC transfer.", "", "| rank | variant | bridge (dex) | SPARC outer (km/s) | Cassini change | Pareto |", "|---:|---|---:|---:|---:|---|"]
    for rank, row in enumerate(ranking, 1):
        lines.append(f"| {rank} | {row['variant']} | {row['bridge_RMSE_dex']:.4f} | {row['SPARC_outer_RMSE_km_s']:.3f} | {row['Cassini_maximum_change']:.2e} | {row['Pareto_nondominated']} |")
    lines += ["", f"Best compromise: **{ranking[0]['variant']}**.", f"Reached both prior-Sigma bridge and RAR galaxy levels: **{report['verdict']['any_RAR_level_and_prior_Sigma_level']}**."]
    (ROOT / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(report["verdict"]), indent=2))


if __name__ == "__main__":
    main()
