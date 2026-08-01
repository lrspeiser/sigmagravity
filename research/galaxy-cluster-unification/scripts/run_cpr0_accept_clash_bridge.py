#!/usr/bin/env python3
"""Test RG/CPR0 with ACCEPT densities at the actual CLASH lensing radii."""

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

from scripts.run_cpr0_joint_bcg_lensing import predict, prepare_domain
from scripts.run_cpr0_manga_bcg_coherence import load_sample as load_bcg_sample
from voidscreen.accept_profiles import (
    interpolate_electron_density_cm3,
    load_accept_profiles,
)

MU_E = 1.17
PROTON_G = 1.67262192369e-24
CLASH_COLUMNS = (
    "cluster",
    "radius_kpc",
    "log_gbar",
    "log_gobs",
    "err_log_gbar",
    "err_log_gobs",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_clash_accelerations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=CLASH_COLUMNS)
    numeric = list(CLASH_COLUMNS[1:])
    if frame.empty or frame[numeric].isna().any().any():
        raise ValueError("CLASH acceleration table is empty or malformed")
    if np.any(~np.isfinite(frame[numeric].to_numpy(dtype=float))):
        raise ValueError("CLASH acceleration table contains non-finite values")
    return frame


def build_clash_sample(
    accept_path: Path,
    clash_path: Path,
    name_map: dict[str, str],
    *,
    minimum_radius_kpc: float,
    density_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join on declared names and interpolate only within measured coverage."""
    if minimum_radius_kpc < 0.0 or density_scale <= 0.0:
        raise ValueError("radius cut must be nonnegative and density scale positive")
    profiles = load_accept_profiles(accept_path)
    clash = load_clash_accelerations(clash_path)
    rows: list[dict] = []
    audit: list[dict] = []
    available_names = set(profiles["name"])

    for cluster, clash_block in clash.groupby("cluster", sort=True):
        accept_name = name_map.get(str(cluster))
        if accept_name is None or accept_name not in available_names:
            audit.append(
                {
                    "cluster": cluster,
                    "accept_name": accept_name or "",
                    "match": False,
                    "profile_min_kpc": np.nan,
                    "profile_max_kpc": np.nan,
                    "clash_rows": len(clash_block),
                    "selected_rows": 0,
                }
            )
            continue
        profile = profiles[profiles["name"] == accept_name].copy()
        measured_min = float(profile["radius_kpc"].min())
        measured_max = float(profile["radius_kpc"].max())
        selected = clash_block[
            (clash_block["radius_kpc"] >= minimum_radius_kpc)
            & (clash_block["radius_kpc"] >= measured_min)
            & (clash_block["radius_kpc"] <= measured_max)
        ].copy()
        if not selected.empty:
            electron_density = interpolate_electron_density_cm3(
                profile, selected["radius_kpc"].to_numpy(dtype=float)
            )
            for source, nelec in zip(selected.itertuples(), electron_density):
                rows.append(
                    {
                        "domain": "cluster",
                        "system": str(source.cluster),
                        "accept_name": accept_name,
                        "radius_kpc": float(source.radius_kpc),
                        "log_gbar": float(source.log_gbar),
                        "log_gobs": float(source.log_gobs),
                        "err_log_gbar": float(source.err_log_gbar),
                        "err_log_gobs": float(source.err_log_gobs),
                        "electron_density_cm3": float(nelec),
                        "local_density_g_cm3": float(
                            density_scale * MU_E * PROTON_G * nelec
                        ),
                        "coherence": 0.0,
                    }
                )
        audit.append(
            {
                "cluster": cluster,
                "accept_name": accept_name,
                "match": True,
                "profile_min_kpc": measured_min,
                "profile_max_kpc": measured_max,
                "clash_rows": len(clash_block),
                "selected_rows": len(selected),
            }
        )

    sample = pd.DataFrame(rows)
    if sample.empty:
        raise ValueError("ACCEPT x CLASH selection produced no rows")
    sample = sample.sort_values(["system", "radius_kpc"]).reset_index(drop=True)
    return sample, pd.DataFrame(audit).sort_values("cluster").reset_index(drop=True)


def assign_group_folds(frame: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    output = frame.reset_index(drop=True).copy()
    assignments: dict[tuple[str, str], int] = {}
    for domain_index, (domain, block) in enumerate(output.groupby("domain", sort=True)):
        systems = np.asarray(sorted(block["system"].astype(str).unique()), dtype=object)
        systems = np.random.default_rng(seed + domain_index).permutation(systems)
        assignments.update(
            {(str(domain), str(system)): int(index % folds) for index, system in enumerate(systems)}
        )
    output["fold"] = [
        assignments[(str(row.domain), str(row.system))] for row in output.itertuples()
    ]
    return output


def equal_group_domain_mse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    scored = frame[["domain", "system"]].copy()
    scored["squared"] = np.square(
        prediction - frame["log_gobs"].to_numpy(dtype=float)
    )
    per_system = scored.groupby(["domain", "system"])["squared"].mean()
    return float(per_system.groupby("domain").mean().mean())


def fit_model(
    frame: pd.DataFrame,
    model: str,
    bounds,
    seed: int,
    response_amplitude: float,
) -> np.ndarray:
    if model == "constant_epsilon":
        result = minimize_scalar(
            lambda value: equal_group_domain_mse(
                frame, predict(frame, model, [value], response_amplitude)
            ),
            bounds=tuple(bounds),
            method="bounded",
        )
        return np.asarray([result.x])
    objective = lambda values: equal_group_domain_mse(
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


def cross_validate(
    frame: pd.DataFrame,
    model: str,
    bounds,
    folds: int,
    seed: int,
    response_amplitude: float,
) -> tuple[np.ndarray, list[dict]]:
    folded = assign_group_folds(frame, folds, seed)
    prediction = np.full(len(folded), np.nan)
    fits: list[dict] = []
    names = (
        ["epsilon"]
        if model == "constant_epsilon"
        else ["epsilon_0", "log10_rho_c_g_cm3", "Q"]
    )
    for fold_id in range(folds):
        training = folded[folded["fold"] != fold_id]
        heldout = folded[folded["fold"] == fold_id]
        vector = fit_model(
            training, model, bounds, seed + fold_id, response_amplitude
        )
        prediction[heldout.index] = predict(
            heldout, model, vector, response_amplitude
        )
        fits.append(
            {
                "fold": fold_id,
                "heldout": {
                    domain: sorted(block["system"].astype(str).unique().tolist())
                    for domain, block in heldout.groupby("domain", sort=True)
                },
                "parameters": dict(zip(names, map(float, vector))),
            }
        )
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError(f"{model} left missing held-out predictions")
    return prediction, fits


def domain_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    output: dict[str, dict | float] = {}
    domain_mse = []
    for domain, block in frame.reset_index(drop=True).groupby("domain", sort=True):
        index = block.index.to_numpy(dtype=int)
        residual = prediction[index] - block["log_gobs"].to_numpy(dtype=float)
        squared = pd.DataFrame(
            {"system": block["system"].to_numpy(), "squared": residual**2}
        )
        system_mse = squared.groupby("system")["squared"].mean()
        mse = float(system_mse.mean())
        domain_mse.append(mse)
        density_log = np.log10(block["local_density_g_cm3"].to_numpy(dtype=float))
        record = {
            "systems": int(block["system"].nunique()),
            "points": len(block),
            "equal_system_RMSE_dex": float(np.sqrt(mse)),
            "point_RMSE_dex": float(np.sqrt(np.mean(residual**2))),
            "median_absolute_residual_dex": float(np.median(np.abs(residual))),
            "mean_residual_dex": float(np.mean(residual)),
            "median_predicted_to_observed": float(np.median(np.power(10.0, residual))),
            "residual_log_density_correlation": float(
                np.corrcoef(residual, density_log)[0, 1]
            ),
        }
        if domain == "cluster":
            log_radius = np.log10(block["radius_kpc"].to_numpy(dtype=float))
            record["radial_residual_slope_dex_per_dex"] = float(
                np.polyfit(log_radius, residual, 1)[0]
            )
            slopes = []
            scored = block[["system", "radius_kpc"]].copy()
            scored["residual"] = residual
            for _, system in scored.groupby("system"):
                if len(system) >= 2:
                    slopes.append(
                        float(
                            np.polyfit(
                                np.log10(system["radius_kpc"]),
                                system["residual"],
                                1,
                            )[0]
                        )
                    )
            record["median_system_radial_residual_slope_dex_per_dex"] = (
                float(np.median(slopes)) if slopes else None
            )
            sigma = np.hypot(
                block["err_log_gbar"].to_numpy(dtype=float),
                block["err_log_gobs"].to_numpy(dtype=float),
            )
            record["diagonal_error_normalized_RMS"] = float(
                np.sqrt(np.mean(np.square(residual / sigma)))
            )
        output[str(domain)] = record
    output["equal_domain_RMSE_dex"] = float(np.sqrt(np.mean(domain_mse)))
    return output


def parameter_vector(record: dict) -> list[float]:
    return [
        float(record["epsilon_0"]),
        float(record["log10_rho_c_g_cm3"]),
        float(record["Q"]),
    ]


def score_fixed_transfers(
    sample: pd.DataFrame, fixed: dict, response_amplitude: float
) -> tuple[dict, dict[str, np.ndarray]]:
    metrics = {}
    predictions = {}
    records = {
        "locked_prior_joint_RG": fixed["prior_joint_RG_median_fold_fit"],
        "locked_prior_joint_CPR0": fixed["prior_joint_CPR0_median_fold_fit"],
        **{f"published_{key}_RG": value for key, value in fixed["published_rg_sets"].items()},
    }
    for label, record in records.items():
        model = "cpr0" if label == "locked_prior_joint_CPR0" else "rg"
        prediction = predict(
            sample, model, parameter_vector(record), response_amplitude
        )
        predictions[label] = prediction
        metrics[label] = domain_metrics(sample, prediction)
    return metrics, predictions


def run_cv_suite(sample: pd.DataFrame, cv: dict) -> tuple[dict, dict, dict]:
    bounds = cv["bounds"]
    amplitude = float(cv["fixed_sigma_response_amplitude"])
    specs = {
        "constant_epsilon_CV": ("constant_epsilon", bounds["constant_epsilon"]),
        "RG_CV": (
            "rg",
            [bounds["epsilon_0"], bounds["log10_rho_c_g_cm3"], bounds["Q"]],
        ),
    }
    if sample["domain"].nunique() > 1:
        specs["CPR0_CV"] = (
            "cpr0",
            [bounds["epsilon_0"], bounds["log10_rho_c_g_cm3"], bounds["Q"]],
        )
        specs.pop("constant_epsilon_CV")
    predictions, fits, metrics = {}, {}, {}
    for label, (model, model_bounds) in specs.items():
        predictions[label], fits[label] = cross_validate(
            sample,
            model,
            model_bounds,
            int(cv["folds"]),
            int(cv["seed"]),
            amplitude,
        )
        metrics[label] = domain_metrics(sample, predictions[label])
    return metrics, fits, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "cpr0_accept_clash_bridge_protocol.json",
    )
    parser.add_argument(
        "--bcg-protocol",
        type=Path,
        default=ROOT / "configs" / "cpr0_manga_bcg_coherence_protocol.json",
    )
    parser.add_argument(
        "--accept",
        type=Path,
        default=ROOT
        / "data"
        / "raw"
        / "accept_cavagnolo2009"
        / "all_profiles.dat.txt",
    )
    parser.add_argument(
        "--clash",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
    )
    parser.add_argument(
        "--tian",
        type=Path,
        default=ROOT / "data" / "derived" / "manga_bcg_tian2024.csv",
    )
    parser.add_argument(
        "--dynpop",
        type=Path,
        default=ROOT
        / "data"
        / "raw"
        / "manga_dynpop"
        / "SDSSDR17_MaNGA_JAM.fits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "cpr0_accept_clash_bridge",
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    bcg_protocol = json.loads(args.bcg_protocol.read_text(encoding="utf-8"))
    construction = protocol["observable_construction"]
    name_map = protocol["cluster_name_map"]
    primary, audit = build_clash_sample(
        args.accept,
        args.clash,
        name_map,
        minimum_radius_kpc=float(construction["primary_radius_cut_kpc"]),
    )
    all_radii, _ = build_clash_sample(
        args.accept,
        args.clash,
        name_map,
        minimum_radius_kpc=float(construction["secondary_radius_cut_kpc"]),
    )
    bcg = prepare_domain(
        load_bcg_sample(args.tian, args.dynpop, bcg_protocol), "BCG"
    )
    joint = pd.concat([bcg, primary], ignore_index=True, sort=False)
    cv = protocol["cross_validation"]
    amplitude = float(cv["fixed_sigma_response_amplitude"])

    primary_metrics, primary_fits, primary_predictions = run_cv_suite(primary, cv)
    all_metrics, all_fits, all_predictions = run_cv_suite(all_radii, cv)
    joint_metrics, joint_fits, joint_predictions = run_cv_suite(joint, cv)
    fixed_metrics, fixed_predictions = score_fixed_transfers(
        primary, protocol["fixed_parameters"], amplitude
    )

    density_sensitivity = {}
    for scale in construction["density_scale_sensitivity"]:
        scaled, _ = build_clash_sample(
            args.accept,
            args.clash,
            name_map,
            minimum_radius_kpc=float(construction["primary_radius_cut_kpc"]),
            density_scale=float(scale),
        )
        sensitivity_metrics, _ = score_fixed_transfers(
            scaled, protocol["fixed_parameters"], amplitude
        )
        density_sensitivity[f"scale_{float(scale):.1f}"] = sensitivity_metrics

    cluster_density = np.log10(primary["local_density_g_cm3"].to_numpy(dtype=float))
    bcg_density = np.log10(bcg["local_density_g_cm3"].to_numpy(dtype=float))
    density_gap = float(max(0.0, np.min(bcg_density) - np.max(cluster_density)))
    gates = protocol["advance_gates"]
    cluster_rg = primary_metrics["RG_CV"]["cluster"]
    cluster_constant = primary_metrics["constant_epsilon_CV"]["cluster"]
    joint_rg = joint_metrics["RG_CV"]
    joint_cpr0 = joint_metrics["CPR0_CV"]
    rhoc = [
        row["parameters"]["log10_rho_c_g_cm3"]
        for row in joint_fits["RG_CV"]
    ]
    gate_audit = {
        "overlap_systems": primary["system"].nunique()
        >= gates["overlap_systems_min"],
        "primary_cluster_points": len(primary)
        >= gates["primary_cluster_points_min"],
        "density_gap": density_gap <= gates["density_gap_dex_max"],
        "locked_prior_RG_cluster_RMSE": fixed_metrics["locked_prior_joint_RG"][
            "cluster"
        ]["equal_system_RMSE_dex"]
        <= gates["locked_prior_RG_cluster_RMSE_dex_max"],
        "cluster_RG_CV_RMSE": cluster_rg["equal_system_RMSE_dex"]
        <= gates["cluster_RG_CV_RMSE_dex_max"],
        "cluster_RG_CV_improves_constant": cluster_constant[
            "equal_system_RMSE_dex"
        ]
        - cluster_rg["equal_system_RMSE_dex"]
        >= gates["cluster_RG_CV_improvement_vs_constant_min_dex"],
        "joint_RG_BCG_RMSE": joint_rg["BCG"]["equal_system_RMSE_dex"]
        <= gates["joint_RG_BCG_RMSE_dex_max"],
        "joint_RG_cluster_RMSE": joint_rg["cluster"]["equal_system_RMSE_dex"]
        <= gates["joint_RG_cluster_RMSE_dex_max"],
        "joint_RG_equal_domain_RMSE": joint_rg["equal_domain_RMSE_dex"]
        <= gates["joint_RG_equal_domain_RMSE_dex_max"],
        "CPR0_improves_RG": joint_rg["equal_domain_RMSE_dex"]
        - joint_cpr0["equal_domain_RMSE_dex"]
        >= gates["CPR0_improvement_vs_density_only_RG_min_dex"],
        "cluster_radial_residual_slope": abs(
            joint_cpr0["cluster"]["radial_residual_slope_dex_per_dex"]
        )
        <= gates["cluster_absolute_radial_residual_slope_dex_per_dex_max"],
        "cluster_residual_density_correlation": abs(
            joint_cpr0["cluster"]["residual_log_density_correlation"]
        )
        <= gates["cluster_absolute_residual_density_correlation_max"],
        "rho_c_fold_range": max(rhoc) - min(rhoc)
        <= gates["log10_rho_c_fold_range_max_dex"],
    }
    gate_audit["passes_all"] = all(gate_audit.values())

    args.output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output / "selection_audit.csv", index=False)
    prediction_rows = []
    for selection, sample, collections in (
        ("primary_r_ge_100_kpc", primary, {**primary_predictions, **fixed_predictions}),
        ("all_measured_radii", all_radii, all_predictions),
        ("joint_primary", joint, joint_predictions),
    ):
        for label, prediction in collections.items():
            block = sample.copy()
            block["selection"] = selection
            block["model"] = label
            block["predicted_log_gobs"] = prediction
            block["residual_dex"] = prediction - block["log_gobs"].to_numpy(dtype=float)
            prediction_rows.append(block)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output / "predictions.csv", index=False
    )

    report = {
        "status": "completed measured ACCEPT-density plus CLASH-lensing bridge",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "bcg_protocol_sha256": sha256(args.bcg_protocol),
            "accept_sha256": sha256(args.accept),
            "clash_sha256": sha256(args.clash),
            "tian_bcg_sha256": sha256(args.tian),
            "dynpop_sha256": sha256(args.dynpop),
        },
        "sample": {
            "declared_name_matches": int(audit["match"].sum()),
            "primary_systems": int(primary["system"].nunique()),
            "primary_points": len(primary),
            "all_radii_systems": int(all_radii["system"].nunique()),
            "all_radii_points": len(all_radii),
            "primary_radius_range_kpc": [
                float(primary["radius_kpc"].min()),
                float(primary["radius_kpc"].max()),
            ],
            "primary_log10_density_range": [
                float(np.min(cluster_density)),
                float(np.max(cluster_density)),
            ],
            "BCG_log10_density_range": [
                float(np.min(bcg_density)),
                float(np.max(bcg_density)),
            ],
            "remaining_density_gap_dex": density_gap,
            "primary_points_by_radius_kpc": {
                str(float(radius)): int(count)
                for radius, count in primary["radius_kpc"].value_counts().sort_index().items()
            },
        },
        "primary_cluster_only_metrics": primary_metrics,
        "primary_cluster_only_fold_fits": primary_fits,
        "fixed_transfer_metrics": fixed_metrics,
        "all_measured_radii_sensitivity": {
            "metrics": all_metrics,
            "fold_fits": all_fits,
        },
        "joint_BCG_plus_primary_CLUSTER": {
            "metrics": joint_metrics,
            "fold_fits": joint_fits,
        },
        "accept_density_scale_fixed_transfer_sensitivity": density_sensitivity,
        "gate_audit": gate_audit,
        "interpretation_guardrails": protocol["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
