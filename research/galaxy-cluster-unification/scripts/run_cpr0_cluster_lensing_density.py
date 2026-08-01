#!/usr/bin/env python3
"""Cross-match Chandra gas profiles to weak-lensing masses and test CPR0 C=0."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_refracted import refracted_permittivity
from voidscreen.unified import G_SI, M_SUN_KG

MPC_M = 3.085677581491367e22
MPC_CM = MPC_M * 100.0
M_SUN_G = M_SUN_KG * 1.0e3
PROTON_G = 1.67262192369e-24
MU_E = 1.17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_name(value: str) -> str:
    name = value.lower().replace("abell", "a")
    name = re.sub(r"[^a-z0-9+]", "", name)
    if name.startswith("zwcl"):
        match = re.match(r"zwcl(\d{4})", name)
        if match:
            return "zwcl" + match.group(1)
    return name


def electron_density_cm3(radius_mpc: float, row: pd.Series) -> float:
    radius = max(float(radius_mpc), 1.0e-12)
    squared = (radius / row["rc"]) ** (-row["alpha"])
    squared /= (1.0 + (radius / row["rc"]) ** 2) ** (
        3.0 * row["beta"] - row["alpha"] / 2.0
    )
    squared /= (1.0 + (radius / row["rs"]) ** 3) ** (row["eps"] / 3.0)
    return float(row["n0"] * np.sqrt(squared))


def raw_gas_mass_msun(radius_mpc: float, row: pd.Series) -> float:
    mass_g = quad(
        lambda radius: 4.0
        * np.pi
        * (radius * MPC_CM) ** 2
        * MPC_CM
        * MU_E
        * PROTON_G
        * electron_density_cm3(radius, row),
        0.0,
        float(radius_mpc),
        limit=400,
        epsabs=0.0,
        epsrel=2.0e-8,
    )[0]
    return float(mass_g / M_SUN_G)


def build_sample(gas_path: Path, lensing_path: Path, stellar_to_gas: float) -> pd.DataFrame:
    gas = pd.read_csv(gas_path)
    lensing = pd.read_csv(lensing_path)
    gas["canonical_name"] = gas["ID"].map(canonical_name)
    lensing["canonical_name"] = lensing["cluster"].map(canonical_name)
    frame = gas.merge(lensing, on="canonical_name", how="inner", validate="one_to_one")
    rows = []
    for _, row in frame.iterrows():
        published_gas_mass = float(row["Mgas"]) * 1.0e13
        raw_at_published_r500 = raw_gas_mass_msun(float(row["R500"]), row)
        normalization = published_gas_mass / raw_at_published_r500
        radius = float(row["r_ap500_mpc"])
        gas_mass = normalization * raw_gas_mass_msun(radius, row)
        baryonic_mass = gas_mass * (1.0 + stellar_to_gas)
        lensing_mass = float(row["m_ap500_1e14_msun"]) * 1.0e14
        radius_m = radius * MPC_M
        local_density = (
            normalization * MU_E * PROTON_G * electron_density_cm3(radius, row)
        )
        record = row.to_dict()
        record.update(
            {
                "profile_normalization": normalization,
                "gas_mass_at_lensing_r500_msun": gas_mass,
                "stellar_to_gas_mass": stellar_to_gas,
                "baryonic_mass_at_lensing_r500_msun": baryonic_mass,
                "local_density_g_cm3": local_density,
                "log_gbar": float(
                    np.log10(G_SI * baryonic_mass * M_SUN_KG / radius_m**2)
                ),
                "log_gobs": float(
                    np.log10(G_SI * lensing_mass * M_SUN_KG / radius_m**2)
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("canonical_name").reset_index(drop=True)


def predict(frame: pd.DataFrame, model: str, vector) -> np.ndarray:
    if model == "constant_epsilon":
        epsilon = np.full(len(frame), float(vector[0]))
    elif model == "rg":
        epsilon_0, log10_rho_c, sharpness = np.asarray(vector, dtype=float)
        epsilon = refracted_permittivity(
            frame["local_density_g_cm3"].to_numpy(dtype=float),
            minimum_permittivity=float(epsilon_0),
            critical_density=10.0**float(log10_rho_c),
            rg_sharpness=float(sharpness),
        )
    else:
        raise ValueError(f"unknown model: {model}")
    return frame["log_gbar"].to_numpy(dtype=float) - np.log10(epsilon)


def mse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - frame["log_gobs"].to_numpy(dtype=float))))


def fit(frame: pd.DataFrame, model: str, bounds, seed: int) -> np.ndarray:
    if model == "constant_epsilon":
        result = minimize_scalar(
            lambda value: mse(frame, predict(frame, model, [value])),
            bounds=tuple(bounds),
            method="bounded",
        )
        return np.asarray([result.x])
    objective = lambda values: mse(frame, predict(frame, model, values))
    global_result = differential_evolution(
        objective,
        list(map(tuple, bounds)),
        seed=seed,
        maxiter=250,
        popsize=15,
        tol=1.0e-10,
        polish=False,
    )
    local_result = minimize(
        objective,
        global_result.x,
        method="L-BFGS-B",
        bounds=list(map(tuple, bounds)),
        options={"maxiter": 5000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    return np.asarray(local_result.x if local_result.success else global_result.x)


def cross_validate(frame: pd.DataFrame, model: str, bounds, folds: int, seed: int):
    output = frame.copy()
    permutation = np.random.default_rng(seed).permutation(output["canonical_name"])
    assignment = {name: int(index % folds) for index, name in enumerate(permutation)}
    output["fold"] = output["canonical_name"].map(assignment)
    prediction = np.full(len(output), np.nan)
    fits = []
    names = ["epsilon"] if model == "constant_epsilon" else ["epsilon_0", "log10_rho_c_g_cm3", "Q"]
    for fold_id in range(folds):
        training = output[output["fold"] != fold_id]
        heldout = output[output["fold"] == fold_id]
        vector = fit(training, model, bounds, seed + fold_id)
        prediction[heldout.index] = predict(heldout, model, vector)
        fits.append(
            {
                "fold": fold_id,
                "heldout_clusters": sorted(heldout["ID"].tolist()),
                "parameters": dict(zip(names, map(float, vector))),
            }
        )
    return output, prediction, fits


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | int]:
    residual = prediction - frame["log_gobs"].to_numpy(dtype=float)
    return {
        "systems": len(frame),
        "RMSE_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "median_absolute_residual_dex": float(np.median(np.abs(residual))),
        "mean_residual_dex": float(np.mean(residual)),
        "median_predicted_to_observed": float(np.median(np.power(10.0, residual))),
        "residual_log_density_correlation": float(
            np.corrcoef(residual, np.log10(frame["local_density_g_cm3"]))[0, 1]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "cpr0_cluster_lensing_density_protocol.json")
    parser.add_argument("--gas", type=Path, default=ROOT / "data" / "raw" / "cluster_gas_profiles" / "elkholy2015" / "clusters.csv")
    parser.add_argument("--lensing", type=Path, default=ROOT / "data" / "raw" / "cccp_meneacs_herbonnet2020" / "weak_lensing_masses.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cpr0_cluster_lensing_density")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    nominal = build_sample(args.gas, args.lensing, 0.10)
    if len(nominal) < protocol["datasets"]["minimum_overlap"]:
        raise ValueError(f"only {len(nominal)} clusters overlap")
    heldout = protocol["heldout_test"]
    bounds = heldout["bounds"]
    scored, constant_prediction, constant_fits = cross_validate(
        nominal, "constant_epsilon", bounds["constant_epsilon"], heldout["folds"], heldout["fold_seed"]
    )
    _, rg_prediction, rg_fits = cross_validate(
        nominal,
        "rg",
        [bounds["epsilon_0"], bounds["log10_rho_c_g_cm3"], bounds["Q"]],
        heldout["folds"],
        heldout["fold_seed"],
    )
    predictions = {
        "constant_epsilon_CV": constant_prediction,
        "RG_local_density_CV": rg_prediction,
    }
    for label, values in protocol["fixed_parameters"]["published_rg_sets"].items():
        predictions[f"published_{label}_RG"] = predict(
            nominal,
            "rg",
            [values["epsilon_0"], values["log10_rho_c_g_cm3"], values["Q"]],
        )
    metric_table = {name: metrics(nominal, values) for name, values in predictions.items()}
    sensitivity = {}
    for stellar_to_gas in (0.0, 0.2):
        sample = build_sample(args.gas, args.lensing, stellar_to_gas)
        for label, values in protocol["fixed_parameters"]["published_rg_sets"].items():
            prediction = predict(sample, "rg", [values["epsilon_0"], values["log10_rho_c_g_cm3"], values["Q"]])
            sensitivity[f"stellar_to_gas_{stellar_to_gas:.1f}_published_{label}_RG"] = metrics(sample, prediction)

    transfer_names = [name for name in predictions if name.startswith("published_")]
    best_transfer_name = min(transfer_names, key=lambda name: metric_table[name]["RMSE_dex"])
    rhoc = [row["parameters"]["log10_rho_c_g_cm3"] for row in rg_fits]
    gates = protocol["advance_gates"]
    rg_metrics = metric_table["RG_local_density_CV"]
    constant_metrics = metric_table["constant_epsilon_CV"]
    gate_audit = {
        "overlap_systems": len(nominal) >= gates["overlap_systems_min"],
        "best_zero_fit_transfer_RMSE": metric_table[best_transfer_name]["RMSE_dex"] <= gates["best_zero_fit_transfer_RMSE_dex_max"],
        "RG_improves_on_constant_epsilon": constant_metrics["RMSE_dex"] - rg_metrics["RMSE_dex"] >= gates["RG_CV_RMSE_improvement_vs_constant_epsilon_min_dex"],
        "RG_mean_residual": abs(rg_metrics["mean_residual_dex"]) <= gates["RG_CV_absolute_mean_residual_dex_max"],
        "RG_density_residual_correlation": abs(rg_metrics["residual_log_density_correlation"]) <= gates["RG_CV_absolute_residual_density_correlation_max"],
        "RG_rhoc_fold_range": max(rhoc) - min(rhoc) <= gates["log10_rho_c_fold_range_max_dex"],
    }
    gate_audit["passes_all"] = all(gate_audit.values())

    rows = []
    for name, values in predictions.items():
        block = scored.copy()
        block["model"] = name
        block["predicted_log_gobs"] = values
        block["residual_dex"] = values - block["log_gobs"]
        rows.append(block)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
    report = {
        "status": "completed added-data local-density weak-lensing endpoint test",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "gas_sha256": sha256(args.gas),
            "lensing_sha256": sha256(args.lensing),
        },
        "sample": {
            "overlap_systems": len(nominal),
            "clusters": nominal["ID"].tolist(),
            "log10_local_density_g_cm3": {
                "minimum": float(np.log10(nominal["local_density_g_cm3"]).min()),
                "median": float(np.log10(nominal["local_density_g_cm3"]).median()),
                "maximum": float(np.log10(nominal["local_density_g_cm3"]).max()),
            },
            "profile_normalization": {
                "minimum": float(nominal["profile_normalization"].min()),
                "median": float(nominal["profile_normalization"].median()),
                "maximum": float(nominal["profile_normalization"].max()),
            },
        },
        "metrics": metric_table,
        "fold_fits": {"constant_epsilon_CV": constant_fits, "RG_local_density_CV": rg_fits},
        "stellar_mass_bracket": sensitivity,
        "best_zero_fit_transfer": {"model": best_transfer_name, "metrics": metric_table[best_transfer_name]},
        "gate_audit": gate_audit,
        "interpretation_guardrails": protocol["claim_boundary"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
