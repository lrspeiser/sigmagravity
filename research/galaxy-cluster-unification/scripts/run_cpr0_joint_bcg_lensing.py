#!/usr/bin/env python3
"""Fit and cross-validate one CPR0 response across BCG dynamics and lensing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_cpr0_cluster_lensing_density import build_sample as build_cluster_sample
from scripts.run_cpr0_manga_bcg_coherence import load_sample as load_bcg_sample
from voidscreen.sigma_refracted import (
    coherence_partitioned_spherical_enhancement,
    refracted_permittivity,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_domain(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    output = frame.copy()
    output["domain"] = domain
    output["system"] = output["plateifu"] if domain == "BCG" else output["canonical_name"]
    if domain == "cluster":
        output["coherence"] = 0.0
    return output[
        ["domain", "system", "log_gbar", "log_gobs", "local_density_g_cm3", "coherence"]
    ].reset_index(drop=True)


def assign_folds(frame: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    output = frame.copy()
    assignments = {}
    for domain_index, (domain, block) in enumerate(output.groupby("domain", sort=True)):
        systems = block["system"].to_numpy(dtype=object)
        permutation = np.random.default_rng(seed + domain_index).permutation(systems)
        assignments.update(
            {(domain, str(system)): int(index % folds) for index, system in enumerate(permutation)}
        )
    output["fold"] = [
        assignments[(row.domain, str(row.system))] for row in output.itertuples()
    ]
    return output


def predict(frame: pd.DataFrame, model: str, vector, response_amplitude: float) -> np.ndarray:
    density = frame["local_density_g_cm3"].to_numpy(dtype=float)
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(dtype=float))
    if model == "constant_epsilon":
        enhancement = np.full(len(frame), 1.0 / float(vector[0]))
    elif model in ("rg", "cpr0"):
        epsilon_0, log10_rho_c, sharpness = np.asarray(vector, dtype=float)
        if model == "rg":
            enhancement = 1.0 / refracted_permittivity(
                density,
                minimum_permittivity=float(epsilon_0),
                critical_density=10.0**float(log10_rho_c),
                rg_sharpness=float(sharpness),
            )
        else:
            enhancement = coherence_partitioned_spherical_enhancement(
                gbar,
                density,
                frame["coherence"].to_numpy(dtype=float),
                response_amplitude=response_amplitude,
                minimum_permittivity=float(epsilon_0),
                critical_density=10.0**float(log10_rho_c),
                rg_sharpness=float(sharpness),
            )
    else:
        raise ValueError(f"unknown model {model}")
    return frame["log_gbar"].to_numpy(dtype=float) + np.log10(enhancement)


def equal_domain_mse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    residual = prediction - frame["log_gobs"].to_numpy(dtype=float)
    values = pd.DataFrame({"domain": frame["domain"], "squared": residual**2})
    return float(values.groupby("domain")["squared"].mean().mean())


def fit(frame: pd.DataFrame, model: str, bounds, seed: int, response_amplitude: float) -> np.ndarray:
    if model == "constant_epsilon":
        result = minimize_scalar(
            lambda value: equal_domain_mse(
                frame, predict(frame, model, [value], response_amplitude)
            ),
            bounds=tuple(bounds),
            method="bounded",
        )
        return np.asarray([result.x])
    objective = lambda values: equal_domain_mse(
        frame, predict(frame, model, values, response_amplitude)
    )
    global_result = differential_evolution(
        objective,
        list(map(tuple, bounds)),
        seed=seed,
        maxiter=350,
        popsize=18,
        tol=1.0e-11,
        polish=False,
    )
    local_result = minimize(
        objective,
        global_result.x,
        method="L-BFGS-B",
        bounds=list(map(tuple, bounds)),
        options={"maxiter": 8000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    return np.asarray(local_result.x if local_result.success else global_result.x)


def cross_validate(frame: pd.DataFrame, model: str, bounds, folds: int, seed: int, response_amplitude: float):
    prediction = np.full(len(frame), np.nan)
    fits = []
    names = ["epsilon"] if model == "constant_epsilon" else ["epsilon_0", "log10_rho_c_g_cm3", "Q"]
    for fold_id in range(folds):
        training = frame[frame["fold"] != fold_id]
        heldout = frame[frame["fold"] == fold_id]
        vector = fit(training, model, bounds, seed + fold_id, response_amplitude)
        prediction[heldout.index] = predict(heldout, model, vector, response_amplitude)
        fits.append(
            {
                "fold": fold_id,
                "heldout": {
                    domain: sorted(block["system"].tolist())
                    for domain, block in heldout.groupby("domain", sort=True)
                },
                "parameters": dict(zip(names, map(float, vector))),
            }
        )
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError(f"{model} left missing predictions")
    return prediction, fits


def domain_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    output = {}
    domain_mse = []
    for domain, block in frame.groupby("domain", sort=True):
        index = block.index.to_numpy()
        residual = prediction[index] - block["log_gobs"].to_numpy(dtype=float)
        mse = float(np.mean(np.square(residual)))
        domain_mse.append(mse)
        output[domain] = {
            "systems": len(block),
            "RMSE_dex": float(np.sqrt(mse)),
            "median_absolute_residual_dex": float(np.median(np.abs(residual))),
            "mean_residual_dex": float(np.mean(residual)),
            "median_predicted_to_observed": float(np.median(np.power(10.0, residual))),
            "residual_log_density_correlation": float(
                np.corrcoef(residual, np.log10(block["local_density_g_cm3"]))[0, 1]
            ),
        }
    output["equal_domain_RMSE_dex"] = float(np.sqrt(np.mean(domain_mse)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "cpr0_joint_bcg_lensing_protocol.json")
    parser.add_argument("--bcg-protocol", type=Path, default=ROOT / "configs" / "cpr0_manga_bcg_coherence_protocol.json")
    parser.add_argument("--tian", type=Path, default=ROOT / "data" / "derived" / "manga_bcg_tian2024.csv")
    parser.add_argument("--dynpop", type=Path, default=ROOT / "data" / "raw" / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits")
    parser.add_argument("--gas", type=Path, default=ROOT / "data" / "raw" / "cluster_gas_profiles" / "elkholy2015" / "clusters.csv")
    parser.add_argument("--lensing", type=Path, default=ROOT / "data" / "raw" / "cccp_meneacs_herbonnet2020" / "weak_lensing_masses.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cpr0_joint_bcg_lensing")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    bcg_protocol = json.loads(args.bcg_protocol.read_text(encoding="utf-8"))
    bcg = prepare_domain(load_bcg_sample(args.tian, args.dynpop, bcg_protocol), "BCG")
    cluster = prepare_domain(build_cluster_sample(args.gas, args.lensing, 0.10), "cluster")
    frame = pd.concat([bcg, cluster], ignore_index=True)
    cv = protocol["cross_validation"]
    frame = assign_folds(frame, cv["folds"], cv["seed"])
    response_amplitude = protocol["fixed_sigma_response_amplitude"]
    bounds = cv["bounds"]
    model_specs = {
        "constant_epsilon_CV": ("constant_epsilon", bounds["constant_epsilon"]),
        "universal_RG_CV": ("rg", [bounds["epsilon_0"], bounds["log10_rho_c_g_cm3"], bounds["Q"]]),
        "universal_CPR0_CV": ("cpr0", [bounds["epsilon_0"], bounds["log10_rho_c_g_cm3"], bounds["Q"]]),
    }
    predictions = {}
    fold_fits = {}
    for label, (model, model_bounds) in model_specs.items():
        predictions[label], fold_fits[label] = cross_validate(
            frame, model, model_bounds, cv["folds"], cv["seed"], response_amplitude
        )
    metric_table = {
        label: domain_metrics(frame, prediction)
        for label, prediction in predictions.items()
    }
    bcg_density = np.log10(bcg["local_density_g_cm3"].to_numpy(dtype=float))
    cluster_density = np.log10(cluster["local_density_g_cm3"].to_numpy(dtype=float))
    density_gap = float(np.min(bcg_density) - np.max(cluster_density))
    cpr0 = metric_table["universal_CPR0_CV"]
    rg = metric_table["universal_RG_CV"]
    gates = protocol["advance_gates"]
    rhoc = [row["parameters"]["log10_rho_c_g_cm3"] for row in fold_fits["universal_CPR0_CV"]]
    gate_audit = {
        "BCG_RMSE": cpr0["BCG"]["RMSE_dex"] <= gates["BCG_RMSE_dex_max"],
        "cluster_lensing_RMSE": cpr0["cluster"]["RMSE_dex"] <= gates["cluster_lensing_RMSE_dex_max"],
        "equal_domain_RMSE": cpr0["equal_domain_RMSE_dex"] <= gates["equal_domain_RMSE_dex_max"],
        "CPR0_improves_on_RG": rg["equal_domain_RMSE_dex"] - cpr0["equal_domain_RMSE_dex"] >= gates["CPR0_improvement_vs_density_only_RG_min_dex"],
        "domain_mean_residuals": all(abs(cpr0[domain]["mean_residual_dex"]) <= gates["absolute_domain_mean_residual_dex_max"] for domain in ("BCG", "cluster")),
        "domain_density_residual_correlations": all(abs(cpr0[domain]["residual_log_density_correlation"]) <= gates["absolute_domain_residual_density_correlation_max"] for domain in ("BCG", "cluster")),
        "rho_c_fold_range": max(rhoc) - min(rhoc) <= gates["log10_rho_c_fold_range_max_dex"],
        "density_gap": density_gap <= gates["maximum_unmeasured_density_gap_dex"],
    }
    gate_audit["passes_all"] = all(gate_audit.values())
    rows = []
    for label, prediction in predictions.items():
        block = frame.copy()
        block["model"] = label
        block["predicted_log_gobs"] = prediction
        block["residual_dex"] = prediction - block["log_gobs"]
        rows.append(block)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
    report = {
        "status": "completed shared-parameter held-out BCG dynamics plus cluster lensing test",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "bcg_protocol_sha256": sha256(args.bcg_protocol),
            "tian_sha256": sha256(args.tian),
            "dynpop_sha256": sha256(args.dynpop),
            "gas_sha256": sha256(args.gas),
            "lensing_sha256": sha256(args.lensing),
        },
        "sample": {
            "BCG_systems": len(bcg),
            "cluster_systems": len(cluster),
            "density_gap_dex": density_gap,
            "BCG_log10_density_range": [float(np.min(bcg_density)), float(np.max(bcg_density))],
            "cluster_log10_density_range": [float(np.min(cluster_density)), float(np.max(cluster_density))],
        },
        "metrics": metric_table,
        "fold_fits": fold_fits,
        "gate_audit": gate_audit,
        "interpretation_guardrails": protocol["claim_boundary"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
