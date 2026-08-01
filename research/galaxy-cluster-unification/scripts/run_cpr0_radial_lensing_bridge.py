#!/usr/bin/env python3
"""Test a shared CPR0 response on grouped radial weak-lensing profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_cpr0_cluster_lensing_density import (
    MPC_M,
    MU_E,
    PROTON_G,
    build_sample as build_cluster_sample,
    electron_density_cm3,
    raw_gas_mass_msun,
)
from scripts.run_cpr0_joint_bcg_lensing import (
    assign_folds,
    predict,
    prepare_domain,
)
from scripts.run_cpr0_manga_bcg_coherence import load_sample as load_bcg_sample
from voidscreen.host_profiles import nfw_mass_function
from voidscreen.unified import G_SI, M_SUN_KG

H0_SI = 70.0 * 1000.0 / MPC_M
OMEGA_M = 0.3
OMEGA_LAMBDA = 0.7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def critical_density_kg_m3(redshift: float) -> float:
    hubble = H0_SI * np.sqrt(OMEGA_M * (1.0 + redshift) ** 3 + OMEGA_LAMBDA)
    return float(3.0 * hubble**2 / (8.0 * np.pi * G_SI))


def overdensity_radius_mpc(mass_msun: float, delta: float, redshift: float) -> float:
    density = critical_density_kg_m3(redshift)
    radius_m = (
        3.0 * mass_msun * M_SUN_KG / (4.0 * np.pi * delta * density)
    ) ** (1.0 / 3.0)
    return float(radius_m / MPC_M)


def nfw_concentration_from_two_masses(
    mass200_msun: float,
    mass500_msun: float,
    redshift: float,
) -> tuple[float, float, float]:
    r200 = overdensity_radius_mpc(mass200_msun, 200.0, redshift)
    r500 = overdensity_radius_mpc(mass500_msun, 500.0, redshift)
    radius_ratio = r500 / r200
    mass_ratio = mass500_msun / mass200_msun

    def equation(concentration: float) -> float:
        return float(
            nfw_mass_function(concentration * radius_ratio)
            / nfw_mass_function(concentration)
            - mass_ratio
        )

    concentration = brentq(equation, 0.1, 30.0)
    return float(concentration), r200, r500


def build_radial_cluster_sample(
    gas_path: Path,
    lensing_path: Path,
    *,
    stellar_to_gas: float = 0.10,
    radial_points: int = 5,
) -> pd.DataFrame:
    base = build_cluster_sample(gas_path, lensing_path, stellar_to_gas)
    rows = []
    for _, source in base.iterrows():
        mass200 = float(source["m_nfw200_1e14_msun"]) * 1.0e14
        mass500 = float(source["m_nfw500_1e14_msun"]) * 1.0e14
        concentration, r200, r500 = nfw_concentration_from_two_masses(
            mass200, mass500, float(source["redshift"])
        )
        minimum_fraction = max(0.35, 0.5 / r500)
        if minimum_fraction >= 1.0:
            continue
        for fraction in np.geomspace(minimum_fraction, 1.0, radial_points):
            radius = float(fraction * r500)
            gas_mass = float(
                source["profile_normalization"]
                * raw_gas_mass_msun(radius, source)
            )
            baryonic_mass = gas_mass * (1.0 + stellar_to_gas)
            total_mass = float(
                mass200
                * nfw_mass_function(concentration * radius / r200)
                / nfw_mass_function(concentration)
            )
            radius_m = radius * MPC_M
            rows.append(
                {
                    "domain": "cluster",
                    "system": source["canonical_name"],
                    "cluster": source["ID"],
                    "radius_mpc": radius,
                    "radius_over_nfw_r500": fraction,
                    "nfw_concentration_200": concentration,
                    "log_gbar": float(
                        np.log10(G_SI * baryonic_mass * M_SUN_KG / radius_m**2)
                    ),
                    "log_gobs": float(
                        np.log10(G_SI * total_mass * M_SUN_KG / radius_m**2)
                    ),
                    "local_density_g_cm3": float(
                        source["profile_normalization"]
                        * MU_E
                        * PROTON_G
                        * electron_density_cm3(radius, source)
                    ),
                    "coherence": 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["system", "radius_mpc"]).reset_index(drop=True)


def equal_group_domain_mse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    residual = prediction - frame["log_gobs"].to_numpy(dtype=float)
    scores = pd.DataFrame(
        {"domain": frame["domain"], "system": frame["system"], "squared": residual**2}
    )
    per_system = scores.groupby(["domain", "system"])["squared"].mean()
    return float(per_system.groupby("domain").mean().mean())


def fit(frame: pd.DataFrame, model: str, bounds, seed: int, response_amplitude: float) -> np.ndarray:
    objective = lambda values: equal_group_domain_mse(
        frame, predict(frame, model, values, response_amplitude)
    )
    global_result = differential_evolution(
        objective,
        list(map(tuple, bounds)),
        seed=seed,
        maxiter=400,
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
    for fold_id in range(folds):
        training = frame[frame["fold"] != fold_id]
        heldout = frame[frame["fold"] == fold_id]
        vector = fit(training, model, bounds, seed + fold_id, response_amplitude)
        prediction[heldout.index] = predict(heldout, model, vector, response_amplitude)
        fits.append(
            {
                "fold": fold_id,
                "heldout": {
                    domain: sorted(block["system"].unique().tolist())
                    for domain, block in heldout.groupby("domain", sort=True)
                },
                "parameters": dict(
                    zip(["epsilon_0", "log10_rho_c_g_cm3", "Q"], map(float, vector))
                ),
            }
        )
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError(f"{model} left missing predictions")
    return prediction, fits


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    result = {}
    domain_mse = []
    for domain, block in frame.groupby("domain", sort=True):
        index = block.index.to_numpy()
        residual = prediction[index] - block["log_gobs"].to_numpy(dtype=float)
        per_system_mse = (
            pd.DataFrame({"system": block["system"], "squared": residual**2})
            .groupby("system")["squared"]
            .mean()
        )
        mse = float(per_system_mse.mean())
        domain_mse.append(mse)
        record = {
            "systems": int(block["system"].nunique()),
            "points": len(block),
            "equal_system_RMSE_dex": float(np.sqrt(mse)),
            "mean_residual_dex": float(np.mean(residual)),
            "residual_log_density_correlation": float(
                np.corrcoef(residual, np.log10(block["local_density_g_cm3"]))[0, 1]
            ),
        }
        if domain == "cluster":
            record["radial_residual_slope_dex_per_dex"] = float(
                np.polyfit(np.log10(block["radius_mpc"]), residual, 1)[0]
            )
        result[domain] = record
    result["equal_domain_RMSE_dex"] = float(np.sqrt(np.mean(domain_mse)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "cpr0_radial_lensing_bridge_protocol.json")
    parser.add_argument("--bcg-protocol", type=Path, default=ROOT / "configs" / "cpr0_manga_bcg_coherence_protocol.json")
    parser.add_argument("--tian", type=Path, default=ROOT / "data" / "derived" / "manga_bcg_tian2024.csv")
    parser.add_argument("--dynpop", type=Path, default=ROOT / "data" / "raw" / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits")
    parser.add_argument("--gas", type=Path, default=ROOT / "data" / "raw" / "cluster_gas_profiles" / "elkholy2015" / "clusters.csv")
    parser.add_argument("--lensing", type=Path, default=ROOT / "data" / "raw" / "cccp_meneacs_herbonnet2020" / "weak_lensing_masses.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cpr0_radial_lensing_bridge")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    bcg_protocol = json.loads(args.bcg_protocol.read_text(encoding="utf-8"))
    bcg = prepare_domain(load_bcg_sample(args.tian, args.dynpop, bcg_protocol), "BCG")
    cluster = build_radial_cluster_sample(args.gas, args.lensing)
    frame = pd.concat([bcg, cluster], ignore_index=True, sort=False)
    cv = protocol["cross_validation"]
    frame = assign_folds(frame, cv["folds"], cv["seed"])
    response_amplitude = cv["fixed_sigma_response_amplitude"]
    bounds = [cv["bounds"]["epsilon_0"], cv["bounds"]["log10_rho_c_g_cm3"], cv["bounds"]["Q"]]
    predictions = {}
    fits = {}
    for label, model in (("universal_RG_CV", "rg"), ("universal_CPR0_CV", "cpr0")):
        predictions[label], fits[label] = cross_validate(
            frame, model, bounds, cv["folds"], cv["seed"], response_amplitude
        )
    metric_table = {label: metrics(frame, values) for label, values in predictions.items()}
    bcg_density = np.log10(bcg["local_density_g_cm3"].to_numpy(dtype=float))
    cluster_density = np.log10(cluster["local_density_g_cm3"].to_numpy(dtype=float))
    density_gap = float(max(0.0, np.min(bcg_density) - np.max(cluster_density)))
    cpr0 = metric_table["universal_CPR0_CV"]
    rg = metric_table["universal_RG_CV"]
    gates = protocol["advance_gates"]
    rhoc = [row["parameters"]["log10_rho_c_g_cm3"] for row in fits["universal_CPR0_CV"]]
    gate_audit = {
        "BCG_RMSE": cpr0["BCG"]["equal_system_RMSE_dex"] <= gates["BCG_RMSE_dex_max"],
        "cluster_RMSE": cpr0["cluster"]["equal_system_RMSE_dex"] <= gates["cluster_equal_system_RMSE_dex_max"],
        "equal_domain_RMSE": cpr0["equal_domain_RMSE_dex"] <= gates["equal_domain_RMSE_dex_max"],
        "CPR0_improves_on_RG": rg["equal_domain_RMSE_dex"] - cpr0["equal_domain_RMSE_dex"] >= gates["CPR0_improvement_vs_density_only_RG_min_dex"],
        "cluster_radial_slope": abs(cpr0["cluster"]["radial_residual_slope_dex_per_dex"]) <= gates["cluster_absolute_radial_residual_slope_dex_per_dex_max"],
        "cluster_density_correlation": abs(cpr0["cluster"]["residual_log_density_correlation"]) <= gates["cluster_absolute_residual_density_correlation_max"],
        "density_gap": density_gap <= gates["maximum_unmeasured_density_gap_dex"],
        "rho_c_fold_range": max(rhoc) - min(rhoc) <= gates["log10_rho_c_fold_range_max_dex"],
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
        "status": "completed grouped radial lensing density-bridge test",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "bcg_protocol_sha256": sha256(args.bcg_protocol),
            "gas_sha256": sha256(args.gas),
            "lensing_sha256": sha256(args.lensing),
        },
        "sample": {
            "BCG_systems": int(bcg["system"].nunique()),
            "cluster_systems": int(cluster["system"].nunique()),
            "cluster_radial_points": len(cluster),
            "density_gap_dex": density_gap,
            "BCG_log10_density_range": [float(np.min(bcg_density)), float(np.max(bcg_density))],
            "cluster_log10_density_range": [float(np.min(cluster_density)), float(np.max(cluster_density))],
            "cluster_radius_mpc_range": [float(cluster["radius_mpc"].min()), float(cluster["radius_mpc"].max())],
            "nfw_concentration_range": [float(cluster["nfw_concentration_200"].min()), float(cluster["nfw_concentration_200"].max())],
        },
        "metrics": metric_table,
        "fold_fits": fits,
        "gate_audit": gate_audit,
        "interpretation_guardrails": protocol["claim_boundary"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
