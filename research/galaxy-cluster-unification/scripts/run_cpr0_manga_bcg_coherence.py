#!/usr/bin/env python3
"""Run the frozen CPR0 local-density/measured-coherence MaNGA BCG test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import differential_evolution, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.data import KPC_M
from voidscreen.host_profiles import prugniel_simien_local_density_from_enclosed_mass
from voidscreen.sigma_refracted import (
    coherence_partitioned_spherical_enhancement,
    refracted_permittivity,
    sigma_enhancement,
)
from voidscreen.unified import G_SI, M_SUN_KG


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native(values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.byteorder not in ("=", "|"):
        return array.astype(array.dtype.newbyteorder("="), copy=False)
    return array


def strings(values) -> np.ndarray:
    return np.char.strip(native(values).astype(str))


def load_sample(tian_path: Path, dynpop_path: Path, protocol: dict) -> pd.DataFrame:
    tian = pd.read_csv(tian_path)
    with fits.open(dynpop_path, memmap=True) as hdus:
        table = hdus[1].data
        dynpop = pd.DataFrame(
            {
                "plateifu": strings(table["plateifu"]),
                "lambda_re": native(table["Lambda_Re"]),
                "sigma_re_km_s": native(table["Sigma_Re"]),
                "dynpop_quality": native(table["Qual"]),
                "drp3qual": native(table["drp3qual"]),
            }
        )
    frame = tian.merge(dynpop, on="plateifu", how="left", validate="one_to_one")
    if frame["lambda_re"].notna().mean() != protocol["sample"]["required_match_fraction"]:
        raise ValueError("DynPop match fraction does not meet the frozen requirement")
    finite = np.isfinite(
        frame[
            [
                "lambda_re",
                "radius_kpc",
                "effective_radius_kpc",
                "sersic_n",
                "log_gbar",
                "log_gobs",
            ]
        ]
    ).all(axis=1)
    selected = frame[
        finite
        & (frame["dynpop_quality"] >= 0)
        & (frame["drp3qual"] == 1)
        & frame["lambda_re"].between(0.0, 1.0)
        & (frame["radius_kpc"] > 0.0)
        & (frame["effective_radius_kpc"] > 0.0)
        & (frame["sersic_n"] > 0.0)
    ].copy()
    expected = protocol["sample"]["predictor_only_audit"]["expected_selected_rows"]
    if len(selected) != expected:
        raise ValueError(f"frozen selection expected {expected} rows, found {len(selected)}")
    radius_m = selected["radius_kpc"].to_numpy(dtype=float) * KPC_M
    enclosed_mass = (
        np.power(10.0, selected["log_gbar"].to_numpy(dtype=float))
        * np.square(radius_m)
        / G_SI
        / M_SUN_KG
    )
    selected["enclosed_baryonic_mass_msun"] = enclosed_mass
    selected["local_density_g_cm3"] = prugniel_simien_local_density_from_enclosed_mass(
        enclosed_mass,
        selected["radius_kpc"],
        selected["effective_radius_kpc"],
        selected["sersic_n"],
    )
    selected["coherence"] = selected["lambda_re"].clip(0.0, 1.0)
    return selected.reset_index(drop=True)


def predict(frame: pd.DataFrame, model: str, vector, *, response_amplitude: float) -> np.ndarray:
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(dtype=float))
    density = frame["local_density_g_cm3"].to_numpy(dtype=float)
    if model == "constant_epsilon":
        enhancement = np.full(len(frame), 1.0 / float(vector[0]))
    elif model == "sigma":
        enhancement = sigma_enhancement(gbar, float(vector[0]))
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


def mse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - frame["log_gobs"].to_numpy(dtype=float))))


def fit(frame: pd.DataFrame, model: str, bounds, *, seed: int, response_amplitude: float) -> np.ndarray:
    if model in ("constant_epsilon", "sigma"):
        result = minimize_scalar(
            lambda value: mse(
                frame, predict(frame, model, [value], response_amplitude=response_amplitude)
            ),
            bounds=tuple(bounds),
            method="bounded",
        )
        return np.asarray([result.x], dtype=float)
    objective = lambda values: mse(
        frame, predict(frame, model, values, response_amplitude=response_amplitude)
    )
    global_result = differential_evolution(
        objective,
        bounds=list(map(tuple, bounds)),
        seed=seed,
        maxiter=250,
        popsize=15,
        polish=False,
        tol=1.0e-10,
        workers=1,
    )
    local_result = minimize(
        objective,
        global_result.x,
        method="L-BFGS-B",
        bounds=list(map(tuple, bounds)),
        options={"maxiter": 5000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    values = local_result.x if local_result.success else global_result.x
    return np.asarray(values, dtype=float)


def assign_folds(frame: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    output = frame.copy()
    systems = output["plateifu"].to_numpy(dtype=object)
    permutation = np.random.default_rng(seed).permutation(systems)
    assignment = {str(system): int(index % folds) for index, system in enumerate(permutation)}
    output["fold"] = output["plateifu"].map(assignment).astype(int)
    return output


def cross_validate(frame: pd.DataFrame, model: str, bounds, *, seed: int, response_amplitude: float):
    prediction = np.full(len(frame), np.nan)
    fits = []
    parameter_names = {
        "constant_epsilon": ["epsilon"],
        "sigma": ["B"],
        "rg": ["epsilon_0", "log10_rho_c_g_cm3", "Q"],
        "cpr0": ["epsilon_0", "log10_rho_c_g_cm3", "Q"],
    }[model]
    for fold_id in sorted(frame["fold"].unique()):
        training = frame[frame["fold"] != fold_id]
        heldout = frame[frame["fold"] == fold_id]
        values = fit(
            training,
            model,
            bounds,
            seed=seed + int(fold_id),
            response_amplitude=response_amplitude,
        )
        prediction[heldout.index] = predict(
            heldout, model, values, response_amplitude=response_amplitude
        )
        fits.append(
            {
                "fold": int(fold_id),
                "heldout_plateifus": sorted(heldout["plateifu"].tolist()),
                "parameters": dict(zip(parameter_names, map(float, values))),
            }
        )
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError(f"{model} cross-validation left missing predictions")
    return prediction, fits


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | int]:
    residual = prediction - frame["log_gobs"].to_numpy(dtype=float)
    lambda_correlation = float(np.corrcoef(residual, frame["lambda_re"])[0, 1])
    gbar_slope = float(np.polyfit(frame["log_gbar"], residual, 1)[0])
    return {
        "systems": len(frame),
        "RMSE_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "median_absolute_residual_dex": float(np.median(np.abs(residual))),
        "mean_residual_dex": float(np.mean(residual)),
        "median_predicted_to_observed": float(np.median(np.power(10.0, residual))),
        "residual_lambda_re_correlation": lambda_correlation,
        "residual_log_gbar_slope": gbar_slope,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "cpr0_manga_bcg_coherence_protocol.json")
    parser.add_argument("--tian", type=Path, default=ROOT / "data" / "derived" / "manga_bcg_tian2024.csv")
    parser.add_argument("--dynpop", type=Path, default=ROOT / "data" / "raw" / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cpr0_manga_bcg_coherence")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frame = load_sample(args.tian, args.dynpop, protocol)
    heldout = protocol["heldout_test"]
    frame = assign_folds(frame, heldout["folds"], heldout["fold_seed"])
    response_amplitude = protocol["fixed_parameters"]["sigma_response_amplitude"]

    predictions = {}
    fold_fits = {}
    bounds = heldout["bounds"]
    for name, model, model_bounds in [
        ("constant_epsilon_CV", "constant_epsilon", bounds["constant_epsilon"]),
        ("Sigma_B_CV", "sigma", bounds["Sigma_B"]),
        ("RG_local_density_CV", "rg", [bounds["epsilon_0"], bounds["log10_rho_c_g_cm3"], bounds["Q"]]),
        ("CPR0_measured_coherence_CV", "cpr0", [bounds["epsilon_0"], bounds["log10_rho_c_g_cm3"], bounds["Q"]]),
    ]:
        predictions[name], fold_fits[name] = cross_validate(
            frame,
            model,
            model_bounds,
            seed=heldout["fold_seed"],
            response_amplitude=response_amplitude,
        )

    transfer = {}
    for label, values in protocol["fixed_parameters"]["published_rg_sets"].items():
        vector = [values["epsilon_0"], values["log10_rho_c_g_cm3"], values["Q"]]
        for model in ("rg", "cpr0"):
            key = f"published_{label}_{model.upper()}"
            transfer[key] = predict(
                frame, model, vector, response_amplitude=response_amplitude
            )
    predictions.update(transfer)
    predictions["fixed_galaxy_Sigma"] = predict(
        frame, "sigma", [response_amplitude], response_amplitude=response_amplitude
    )

    metric_table = {name: metrics(frame, values) for name, values in predictions.items()}
    output_rows = []
    for name, values in predictions.items():
        block = frame.copy()
        block["model"] = name
        block["predicted_log_gobs"] = values
        block["residual_dex"] = values - block["log_gobs"]
        output_rows.append(block)
    output = pd.concat(output_rows, ignore_index=True)

    sensitivity = {}
    for scale in (0.8, 1.2):
        shifted = frame.copy()
        shifted["local_density_g_cm3"] *= scale
        for label, values in protocol["fixed_parameters"]["published_rg_sets"].items():
            vector = [values["epsilon_0"], values["log10_rho_c_g_cm3"], values["Q"]]
            name = f"published_{label}_CPR0"
            sensitivity[f"density_x{scale:.1f}_{name}"] = metrics(
                shifted,
                predict(shifted, "cpr0", vector, response_amplitude=response_amplitude),
            )

    gates = protocol["advance_gates"]
    cv = metric_table["CPR0_measured_coherence_CV"]
    rg = metric_table["RG_local_density_CV"]
    constant = metric_table["constant_epsilon_CV"]
    zero_fit_names = list(transfer)
    best_transfer_name = min(zero_fit_names, key=lambda key: metric_table[key]["RMSE_dex"])
    rhoc_values = [
        fit_row["parameters"]["log10_rho_c_g_cm3"]
        for fit_row in fold_fits["CPR0_measured_coherence_CV"]
    ]
    gate_audit = {
        "selected_systems": len(frame) >= gates["selected_systems_min"],
        "best_zero_fit_transfer_RMSE": metric_table[best_transfer_name]["RMSE_dex"] <= gates["best_zero_fit_transfer_RMSE_dex_max"],
        "CPR0_improves_on_RG": rg["RMSE_dex"] - cv["RMSE_dex"] >= gates["CPR0_CV_RMSE_improvement_vs_RG_min_dex"],
        "CPR0_improves_on_constant_epsilon": constant["RMSE_dex"] - cv["RMSE_dex"] >= gates["CPR0_CV_RMSE_improvement_vs_constant_epsilon_min_dex"],
        "CPR0_mean_residual": abs(cv["mean_residual_dex"]) <= gates["CPR0_CV_absolute_mean_residual_dex_max"],
        "CPR0_lambda_residual_correlation": abs(cv["residual_lambda_re_correlation"]) <= gates["CPR0_CV_absolute_residual_lambda_correlation_max"],
        "CPR0_rhoc_fold_range": max(rhoc_values) - min(rhoc_values) <= gates["log10_rho_c_fold_range_max_dex"],
    }
    gate_audit["passes_all"] = all(gate_audit.values())

    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "predictions.csv", index=False)
    report = {
        "status": "completed frozen CPR0 MaNGA BCG local-density and measured-coherence test",
        "inputs": {
            "protocol": str(args.protocol.relative_to(ROOT)),
            "protocol_sha256": sha256(args.protocol),
            "tian_sha256": sha256(args.tian),
            "dynpop_sha256": sha256(args.dynpop),
        },
        "sample": {
            "systems": len(frame),
            "quality_counts": {str(k): int(v) for k, v in frame["dynpop_quality"].value_counts().sort_index().items()},
            "lambda_re": {"minimum": float(frame["lambda_re"].min()), "median": float(frame["lambda_re"].median()), "maximum": float(frame["lambda_re"].max())},
            "log10_local_density_g_cm3": {"minimum": float(np.log10(frame["local_density_g_cm3"]).min()), "median": float(np.log10(frame["local_density_g_cm3"]).median()), "maximum": float(np.log10(frame["local_density_g_cm3"]).max())},
        },
        "metrics": metric_table,
        "fold_fits": fold_fits,
        "density_normalization_sensitivity": sensitivity,
        "best_zero_fit_transfer": {"model": best_transfer_name, "metrics": metric_table[best_transfer_name]},
        "gate_audit": gate_audit,
        "interpretation_guardrails": protocol["claim_boundary"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
