#!/usr/bin/env python3
"""Cross-validate a finite Sigma/RG formula family against MOND/RAR controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_cpr0_accept_clash_bcg_stellar import build_stellar_augmented_sample
from scripts.run_cpr0_accept_clash_bridge import (
    assign_group_folds,
    domain_metrics,
    equal_group_domain_mse,
)
from scripts.run_cpr0_manga_bcg_coherence import load_sample as load_bcg_sample
from voidscreen.phenomenology import (
    dimensionless_baryonic_potential,
    fixed_rar_enhancement,
    response_enhancement,
    simple_mond_enhancement,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(value):
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def build_joint_sample(
    protocol: dict,
    base_protocol: dict,
    bcg_protocol: dict,
    *,
    accept_path: Path,
    clash_path: Path,
    table1_path: Path,
    tian_path: Path,
    dynpop_path: Path,
) -> pd.DataFrame:
    cluster = build_stellar_augmented_sample(
        accept_path, clash_path, table1_path, base_protocol["cluster_name_map"]
    ).copy()
    bcg = load_bcg_sample(tian_path, dynpop_path, bcg_protocol).copy()
    bcg["domain"] = "BCG"
    bcg["system"] = bcg["plateifu"].astype(str)
    bcg["err_log_gbar"] = np.nan
    bcg["err_log_gobs"] = np.nan
    keep = [
        "domain",
        "system",
        "radius_kpc",
        "log_gbar",
        "log_gobs",
        "err_log_gbar",
        "err_log_gobs",
        "local_density_g_cm3",
        "coherence",
    ]
    joint = pd.concat([bcg[keep], cluster[keep]], ignore_index=True, sort=False)
    sample = protocol["sample"]
    return assign_group_folds(joint, int(sample["folds"]), int(sample["seed"]))


def predict_formula(frame: pd.DataFrame, model: str, parameters, constants: dict) -> np.ndarray:
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(dtype=float))
    enhancement = response_enhancement(
        model,
        gbar,
        frame["local_density_g_cm3"].to_numpy(dtype=float),
        frame["radius_kpc"].to_numpy(dtype=float),
        parameters,
        g_reference_m_s2=float(constants["g_reference_m_s2"]),
        potential_reference=float(constants["potential_reference"]),
        sigma_g_dagger_m_s2=float(constants["sigma_g_dagger_m_s2"]),
        rar_acceleration_m_s2=float(constants.get("rar_acceleration_m_s2", 1.2e-10)),
        fixed_gate_log10_phi_c=float(constants.get("fixed_gate_log10_phi_c", -6.3)),
        fixed_gate_sharpness=float(constants.get("fixed_gate_sharpness", 4.0)),
        coherence=frame["coherence"].to_numpy(dtype=float),
        coherence_gate_power=float(constants.get("coherence_gate_power", 2.0)),
    )
    return frame["log_gbar"].to_numpy(dtype=float) + np.log10(enhancement)


def fit_formula(
    frame: pd.DataFrame, model: str, bounds, constants: dict, optimization: dict, seed: int
) -> np.ndarray:
    objective = lambda values: equal_group_domain_mse(
        frame, predict_formula(frame, model, values, constants)
    )
    global_result = differential_evolution(
        objective,
        list(map(tuple, bounds)),
        seed=seed,
        maxiter=int(optimization["differential_evolution_maxiter"]),
        popsize=int(optimization["differential_evolution_popsize"]),
        tol=1.0e-10,
        polish=False,
        workers=1,
    )
    local = minimize(
        objective,
        global_result.x,
        method="L-BFGS-B",
        bounds=list(map(tuple, bounds)),
        options={"maxiter": 8000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    return np.asarray(local.x if local.success else global_result.x, dtype=float)


def boundary_flags(values: np.ndarray, bounds, tolerance: float) -> list[bool]:
    flags = []
    for value, (lower, upper) in zip(values, bounds):
        width = float(upper) - float(lower)
        flags.append(
            bool(
                value <= float(lower) + tolerance * width
                or value >= float(upper) - tolerance * width
            )
        )
    return flags


def cross_validate_formula(
    frame: pd.DataFrame, model: str, spec: dict, protocol: dict
) -> tuple[np.ndarray, list[dict], dict]:
    prediction = np.full(len(frame), np.nan)
    fits = []
    bounds = spec["bounds"]
    tolerance = float(protocol["advance_gates"]["boundary_fraction_tolerance"])
    for fold in range(int(protocol["sample"]["folds"])):
        training = frame[frame["fold"] != fold]
        heldout = frame[frame["fold"] == fold]
        values = fit_formula(
            training,
            model,
            bounds,
            protocol["shared_constants"],
            protocol["optimization"],
            int(protocol["sample"]["seed"]) + fold,
        )
        prediction[heldout.index] = predict_formula(
            heldout, model, values, protocol["shared_constants"]
        )
        fits.append(
            {
                "fold": fold,
                "heldout": {
                    domain: sorted(block["system"].astype(str).unique().tolist())
                    for domain, block in heldout.groupby("domain", sort=True)
                },
                "parameters": dict(zip(spec["parameters"], map(float, values))),
                "at_boundary": dict(
                    zip(spec["parameters"], boundary_flags(values, bounds, tolerance))
                ),
            }
        )
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError(f"{model} left non-finite held-out predictions")
    full_values = fit_formula(
        frame,
        model,
        bounds,
        protocol["shared_constants"],
        protocol["optimization"],
        int(protocol["sample"]["seed"]) + 100,
    )
    full = {
        "parameters": dict(zip(spec["parameters"], map(float, full_values))),
        "at_boundary": dict(
            zip(spec["parameters"], boundary_flags(full_values, bounds, tolerance))
        ),
        "training_equal_domain_RMSE_dex": float(
            np.sqrt(
                equal_group_domain_mse(
                    frame,
                    predict_formula(frame, model, full_values, protocol["shared_constants"]),
                )
            )
        ),
    }
    return prediction, fits, full


def fixed_prediction(frame: pd.DataFrame, kind: str, scale: float | None = None) -> np.ndarray:
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(dtype=float))
    if kind == "newtonian":
        enhancement = np.ones_like(gbar)
    elif kind == "rar":
        enhancement = fixed_rar_enhancement(gbar, float(scale))
    elif kind == "simple_mond":
        enhancement = simple_mond_enhancement(gbar, float(scale))
    else:
        raise ValueError(kind)
    return frame["log_gbar"].to_numpy(dtype=float) + np.log10(enhancement)


def surface_matrix(frame: pd.DataFrame, name: str, center=None, spread=None):
    logg = frame["log_gbar"].to_numpy(dtype=float)
    logrho = np.log10(frame["local_density_g_cm3"].to_numpy(dtype=float))
    potential = dimensionless_baryonic_potential(
        np.power(10.0, logg), frame["radius_kpc"].to_numpy(dtype=float)
    )
    raw = np.column_stack([logg, logrho, np.log10(potential)])
    if center is None:
        center = raw.mean(axis=0)
        spread = raw.std(axis=0)
    standardized = (raw - center) / np.where(np.asarray(spread) > 0.0, spread, 1.0)
    x, y, z = standardized.T
    if name == "linear_g_rho":
        matrix = np.column_stack([np.ones(len(frame)), x, y])
    elif name == "quadratic_g_rho":
        matrix = np.column_stack([np.ones(len(frame)), x, y, x * x, x * y, y * y])
    elif name == "quadratic_g_rho_potential":
        matrix = np.column_stack(
            [np.ones(len(frame)), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z]
        )
    else:
        raise ValueError(name)
    return matrix, np.asarray(center), np.asarray(spread)


def row_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(["domain", "system"])["system"].transform("count").to_numpy()
    systems = frame.groupby("domain")["system"].transform("nunique").to_numpy()
    domains = frame["domain"].nunique()
    return 1.0 / (domains * systems * counts)


def cross_validate_surface(frame: pd.DataFrame, name: str, ridge: float) -> np.ndarray:
    prediction = np.full(len(frame), np.nan)
    for fold in sorted(frame["fold"].unique()):
        training = frame[frame["fold"] != fold]
        heldout = frame[frame["fold"] == fold]
        train_x, center, spread = surface_matrix(training, name)
        heldout_x, _, _ = surface_matrix(heldout, name, center, spread)
        target = (
            training["log_gobs"].to_numpy(dtype=float)
            - training["log_gbar"].to_numpy(dtype=float)
        )
        weights = row_weights(training)
        normal = train_x.T @ (weights[:, None] * train_x)
        penalty = np.eye(train_x.shape[1]) * ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            normal + penalty, train_x.T @ (weights * target)
        )
        prediction[heldout.index] = (
            heldout["log_gbar"].to_numpy(dtype=float) + heldout_x @ coefficients
        )
    return prediction


def paired_system_bootstrap(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict:
    records = []
    for domain, block in frame.groupby("domain", sort=True):
        indices = [group.index.to_numpy(dtype=int) for _, group in block.groupby("system")]
        records.append((str(domain), indices))
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    observed = np.sqrt(equal_group_domain_mse(frame, candidate)) - np.sqrt(
        equal_group_domain_mse(frame, reference)
    )
    observed_values = frame["log_gobs"].to_numpy(dtype=float)
    for draw in range(draws):
        candidate_domain_mse = []
        reference_domain_mse = []
        for _, groups in records:
            picked = rng.integers(0, len(groups), size=len(groups))
            candidate_domain_mse.append(
                np.mean(
                    [np.mean((candidate[groups[i]] - observed_values[groups[i]]) ** 2) for i in picked]
                )
            )
            reference_domain_mse.append(
                np.mean(
                    [np.mean((reference[groups[i]] - observed_values[groups[i]]) ** 2) for i in picked]
                )
            )
        deltas[draw] = np.sqrt(np.mean(candidate_domain_mse)) - np.sqrt(
            np.mean(reference_domain_mse)
        )
    return {
        "definition": "candidate minus reference equal-domain RMSE; negative favors candidate",
        "observed_delta_dex": float(observed),
        "percentile_95_interval_dex": list(map(float, np.percentile(deltas, [2.5, 97.5]))),
        "probability_candidate_better": float(np.mean(deltas < 0.0)),
        "draws": draws,
    }


def make_summary_plot(metrics: dict, output: Path) -> None:
    names = list(metrics)
    bcg = [metrics[name]["BCG"]["equal_system_RMSE_dex"] for name in names]
    cluster = [metrics[name]["cluster"]["equal_system_RMSE_dex"] for name in names]
    figure, axis = plt.subplots(figsize=(12, 6.5))
    positions = np.arange(len(names))
    width = 0.38
    axis.bar(positions - width / 2, bcg, width, label="BCG dynamics")
    axis.bar(positions + width / 2, cluster, width, label="CLASH NFW-lensing reconstruction")
    axis.axhline(0.12, color="black", linestyle="--", linewidth=1, label="0.12 dex target")
    axis.set_ylabel("held-out equal-system RMSE (dex)")
    axis.set_xticks(positions, [name.replace("RG_", "") for name in names], rotation=50, ha="right")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs" / "phenomenology_formula_sweep_protocol.json"
    )
    parser.add_argument(
        "--base-protocol", type=Path, default=ROOT / "configs" / "cpr0_accept_clash_bridge_protocol.json"
    )
    parser.add_argument(
        "--bcg-protocol", type=Path, default=ROOT / "configs" / "cpr0_manga_bcg_coherence_protocol.json"
    )
    parser.add_argument(
        "--accept", type=Path, default=ROOT / "data" / "raw" / "accept_cavagnolo2009" / "all_profiles.dat.txt"
    )
    parser.add_argument(
        "--clash", type=Path, default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat"
    )
    parser.add_argument(
        "--table1", type=Path, default=ROOT / "data" / "raw" / "clash_tian2020" / "table1.dat"
    )
    parser.add_argument(
        "--tian", type=Path, default=ROOT / "data" / "derived" / "manga_bcg_tian2024.csv"
    )
    parser.add_argument(
        "--dynpop", type=Path, default=ROOT / "data" / "raw" / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "phenomenology_formula_sweep"
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    base = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    bcg_protocol = json.loads(args.bcg_protocol.read_text(encoding="utf-8"))
    frame = build_joint_sample(
        protocol,
        base,
        bcg_protocol,
        accept_path=args.accept,
        clash_path=args.clash,
        table1_path=args.table1,
        tian_path=args.tian,
        dynpop_path=args.dynpop,
    )

    references = protocol["fixed_references"]
    predictions = {
        "Newtonian": fixed_prediction(frame, "newtonian"),
        "fixed_galaxy_RAR": fixed_prediction(
            frame, "rar", float(references["galaxy_RAR_g_dagger_m_s2"])
        ),
        "simple_MOND": fixed_prediction(
            frame, "simple_mond", float(references["simple_MOND_a0_m_s2"])
        ),
        "cluster_scale_RAR_diagnostic": fixed_prediction(
            frame, "rar", float(references["cluster_RAR_diagnostic_g_dagger_m_s2"])
        ),
    }
    fold_fits = {}
    full_fits = {}
    for model, spec in protocol["models"].items():
        print(f"cross-validating {model}", flush=True)
        predictions[model], fold_fits[model], full_fits[model] = cross_validate_formula(
            frame, model, spec, protocol
        )

    ridge = float(protocol["diagnostic_surfaces"]["ridge_penalty"])
    for surface in ("linear_g_rho", "quadratic_g_rho", "quadratic_g_rho_potential"):
        predictions[surface] = cross_validate_surface(frame, surface, ridge)

    metrics = {name: domain_metrics(frame, prediction) for name, prediction in predictions.items()}
    physical_names = list(protocol["models"])
    best_physical = min(
        physical_names, key=lambda name: metrics[name]["equal_domain_RMSE_dex"]
    )
    best_surface = min(
        ("linear_g_rho", "quadratic_g_rho", "quadratic_g_rho_potential"),
        key=lambda name: metrics[name]["equal_domain_RMSE_dex"],
    )
    rg = metrics["RG"]
    winner = metrics[best_physical]
    fixed_rar_bcg = metrics["fixed_galaxy_RAR"]["BCG"]["equal_system_RMSE_dex"]
    gates = protocol["advance_gates"]
    boundary_count = sum(
        sum(record["at_boundary"].values()) for record in fold_fits[best_physical]
    )
    gate_audit = {
        "primary_equal_domain_RMSE": winner["equal_domain_RMSE_dex"]
        <= gates["primary_equal_domain_RMSE_dex_max"],
        "BCG_competitive_with_fixed_galaxy_RAR": winner["BCG"]["equal_system_RMSE_dex"]
        <= gates["BCG_RMSE_relative_to_fixed_galaxy_RAR_max"] * fixed_rar_bcg,
        "cluster_within_diagonal_error_scale": winner["cluster"][
            "diagonal_error_normalized_RMS"
        ]
        <= gates["cluster_diagonal_error_normalized_RMS_max"],
        "cluster_radial_shape": abs(
            winner["cluster"]["radial_residual_slope_dex_per_dex"]
        )
        <= gates["absolute_cluster_radial_residual_slope_dex_per_dex_max"],
        "improves_RG": rg["equal_domain_RMSE_dex"] - winner["equal_domain_RMSE_dex"]
        >= gates["improvement_over_RG_equal_domain_RMSE_dex_min"],
        "neither_domain_regresses_vs_RG": all(
            winner[domain]["equal_system_RMSE_dex"]
            <= rg[domain]["equal_system_RMSE_dex"]
            + gates["maximum_domain_regression_vs_RG_dex"]
            for domain in ("BCG", "cluster")
        ),
        "no_fold_parameter_at_boundary": boundary_count == 0,
    }
    gate_audit["passes_all"] = all(gate_audit.values())

    bootstrap = paired_system_bootstrap(
        frame,
        predictions[best_physical],
        predictions["RG"],
        draws=int(protocol["optimization"]["paired_system_bootstrap_draws"]),
        seed=int(protocol["sample"]["seed"]) + 500,
    )
    cluster = frame[frame["domain"] == "cluster"]
    cluster_sigma = np.hypot(
        cluster["err_log_gbar"].to_numpy(dtype=float),
        cluster["err_log_gobs"].to_numpy(dtype=float),
    )
    report = {
        "status": "completed finite exploratory formula sweep",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "base_protocol_sha256": sha256(args.base_protocol),
            "bcg_protocol_sha256": sha256(args.bcg_protocol),
            "accept_sha256": sha256(args.accept),
            "clash_sha256": sha256(args.clash),
            "table1_sha256": sha256(args.table1),
            "tian_sha256": sha256(args.tian),
            "dynpop_sha256": sha256(args.dynpop),
        },
        "sample": {
            "points": len(frame),
            "BCG_systems": int(frame[frame["domain"] == "BCG"]["system"].nunique()),
            "cluster_systems": int(cluster["system"].nunique()),
            "cluster_points": len(cluster),
            "cluster_median_diagonal_error_dex": float(np.median(cluster_sigma)),
            "log_gbar_range": list(map(float, [frame["log_gbar"].min(), frame["log_gbar"].max()])),
            "log_density_range": list(
                map(
                    float,
                    [
                        np.log10(frame["local_density_g_cm3"]).min(),
                        np.log10(frame["local_density_g_cm3"]).max(),
                    ],
                )
            ),
        },
        "metrics": metrics,
        "fold_fits": fold_fits,
        "full_sample_descriptive_fits": full_fits,
        "selection": {
            "best_physical_model": best_physical,
            "best_diagnostic_surface": best_surface,
            "best_physical_vs_RG_paired_bootstrap": bootstrap,
            "gate_audit": gate_audit,
        },
        "dark_matter_comparison": {
            "NFW_construction_residual_dex": 0.0,
            "why_not_an_independent_score": references["cluster_NFW_reference"],
            "usable_comparison": "candidate residual divided by the public diagonal CLASH acceleration error",
            "best_physical_cluster_error_normalized_RMS": winner["cluster"][
                "diagonal_error_normalized_RMS"
            ],
        },
        "claim_boundary": protocol["claim_boundary"],
        "failure_policy": protocol["failure_policy"],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "sample.csv", index=False)
    rows = []
    for name, prediction in predictions.items():
        block = frame.copy()
        block["model"] = name
        block["predicted_log_gobs"] = prediction
        block["residual_dex"] = prediction - block["log_gobs"].to_numpy(dtype=float)
        rows.append(block)
    pd.concat(rows, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
    (args.output / "report.json").write_text(
        json.dumps(strict_json(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    make_summary_plot(metrics, args.output / "formula_sweep_summary.png")
    print(json.dumps(strict_json(report["selection"]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
