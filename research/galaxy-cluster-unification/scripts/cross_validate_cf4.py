from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from voidscreen.data import PackedDataset, pack_dataset
from voidscreen.experiment import (
    ExperimentSettings,
    fit_model,
    metrics,
    prediction_frame,
    resolve_device,
    set_reproducible_seed,
)
from voidscreen.models import TensorDataset, build_model

ROOT = Path(__file__).resolve().parents[1]
SCORE_COLUMNS = (
    "void_score_grouped_64",
    "void_score_ungrouped_64",
    "void_score_ungrouped_128",
)


def _settings(config_path: Path, *, steps: int) -> ExperimentSettings:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return ExperimentSettings(
        seed=int(config["seed"]),
        disk_mass_to_light_prior=float(config["disk_mass_to_light_prior"]),
        bulge_mass_to_light_prior=float(config["bulge_mass_to_light_prior"]),
        log_mass_to_light_prior_sigma=float(config["log_mass_to_light_prior_sigma"]),
        velocity_error_floor_kms=float(config["velocity_error_floor_kms"]),
        rar_acceleration_m_s2=float(config["rar_acceleration_m_s2"]),
        hubble_km_s_mpc=float(config["hubble_km_s_mpc"]),
        learning_rate=float(config["learning_rate"]),
        steps=steps,
    )


def _score_lookup(environment_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(environment_csv).set_index("galaxy")
    missing = set(SCORE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CF4 sensitivity columns: {sorted(missing)}")
    return frame


def _environment_values(
    packed: PackedDataset, lookup: pd.DataFrame, score_column: str
) -> np.ndarray:
    return lookup.loc[list(packed.galaxy_names), score_column].to_numpy(dtype=np.float64)


def _balanced_folds(primary_score: np.ndarray, n_folds: int) -> np.ndarray:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    order = np.argsort(primary_score, kind="stable")
    assignments = np.empty(primary_score.size, dtype=np.int64)
    # A serpentine assignment balances every fold over the full environment range.
    for block_start in range(0, primary_score.size, n_folds):
        block = order[block_start : block_start + n_folds]
        fold_order = np.arange(block.size)
        if (block_start // n_folds) % 2:
            fold_order = fold_order[::-1]
        assignments[block] = fold_order
    return assignments


def _fold_dataset(
    packed: PackedDataset,
    raw_environment: np.ndarray,
    fold_assignment: np.ndarray,
    heldout_fold: int,
    score_column: str,
) -> PackedDataset:
    training_galaxies = fold_assignment != heldout_fold
    mean = float(raw_environment[training_galaxies].mean())
    standard_deviation = float(raw_environment[training_galaxies].std(ddof=0))
    if standard_deviation <= 0.0:
        raise ValueError(f"Zero training environment variance for fold {heldout_fold}")
    standardized = (raw_environment - mean) / standard_deviation
    train_mask = training_galaxies[packed.galaxy_index]
    return replace(
        packed,
        train_mask=train_mask,
        environment_raw=raw_environment,
        environment_standardized=standardized,
        environment_score_column=score_column,
    )


def _per_galaxy_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame["split"] == "outer_holdout"].copy()
    selected["residual"] = (
        selected["velocity_predicted_kms"] - selected["velocity_observed_adjusted_kms"]
    )
    selected["chi2_term"] = (selected["residual"] / selected["velocity_error_total_kms"]) ** 2
    selected["squared_error"] = selected["residual"] ** 2
    return (
        selected.groupby("galaxy", sort=True)
        .agg(
            n=("residual", "size"),
            chi2=("chi2_term", "sum"),
            squared_error=("squared_error", "sum"),
            mean_residual_kms=("residual", "mean"),
        )
        .reset_index()
    )


def _variant_specs(*, include_potential: bool, include_extensions: bool) -> list[dict[str, Any]]:
    variants = [
        {
            "name": "newtonian_reference",
            "model": "newtonian",
            "environment_enabled": False,
            "score_column": SCORE_COLUMNS[0],
            "fit": False,
        },
        {
            "name": "rar_reference",
            "model": "rar",
            "environment_enabled": False,
            "score_column": SCORE_COLUMNS[0],
            "fit": False,
        },
        {
            "name": "void_no_environment",
            "model": "void",
            "environment_enabled": False,
            "score_column": SCORE_COLUMNS[0],
            "fit": True,
        },
        *[
            {
                "name": f"void_env_{column.removeprefix('void_score_')}",
                "model": "void",
                "environment_enabled": True,
                "score_column": column,
                "fit": True,
            }
            for column in SCORE_COLUMNS
        ],
    ]
    if include_potential:
        variants.append(
            {
                "name": "potential_P0",
                "model": "potential",
                "environment_enabled": False,
                "boundary_layer_enabled": False,
                "score_column": SCORE_COLUMNS[0],
                "fit": True,
            }
        )
    if include_extensions:
        if not include_potential:
            raise ValueError("Potential extensions require the P0 baseline")
        variants.extend(
            [
                {
                    "name": "potential_P1_environment",
                    "model": "potential",
                    "environment_enabled": True,
                    "boundary_layer_enabled": False,
                    "score_column": SCORE_COLUMNS[0],
                    "fit": True,
                },
                {
                    "name": "potential_B1_boundary",
                    "model": "potential",
                    "environment_enabled": False,
                    "boundary_layer_enabled": True,
                    "score_column": SCORE_COLUMNS[0],
                    "fit": True,
                },
            ]
        )
    return variants


def _bootstrap_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    merged = baseline.merge(candidate, on="galaxy", suffixes=("_baseline", "_candidate"))
    if not np.array_equal(merged["n_baseline"], merged["n_candidate"]):
        raise ValueError("Paired models do not have identical galaxy point counts")
    n = merged["n_baseline"].to_numpy(dtype=float)
    delta_chi2 = (merged["chi2_candidate"] - merged["chi2_baseline"]).to_numpy(dtype=float)
    sse_baseline = merged["squared_error_baseline"].to_numpy(dtype=float)
    sse_candidate = merged["squared_error_candidate"].to_numpy(dtype=float)
    observed_delta = float(delta_chi2.sum() / n.sum())
    observed_rmse_delta = float(
        math.sqrt(sse_candidate.sum() / n.sum()) - math.sqrt(sse_baseline.sum() / n.sum())
    )

    rng = np.random.default_rng(seed)
    chi2_draws: list[np.ndarray] = []
    rmse_draws: list[np.ndarray] = []
    chunk_size = 10_000
    for start in range(0, draws, chunk_size):
        chunk = min(chunk_size, draws - start)
        indices = rng.integers(0, len(merged), size=(chunk, len(merged)))
        point_count = n[indices].sum(axis=1)
        chi2_draws.append(delta_chi2[indices].sum(axis=1) / point_count)
        base_rmse = np.sqrt(sse_baseline[indices].sum(axis=1) / point_count)
        candidate_rmse = np.sqrt(sse_candidate[indices].sum(axis=1) / point_count)
        rmse_draws.append(candidate_rmse - base_rmse)
    boot_chi2 = np.concatenate(chi2_draws)
    boot_rmse = np.concatenate(rmse_draws)
    return {
        "galaxies": len(merged),
        "points": int(n.sum()),
        "candidate_minus_baseline_chi2_per_point": observed_delta,
        "chi2_per_point_bootstrap_ci95_low": float(np.quantile(boot_chi2, 0.025)),
        "chi2_per_point_bootstrap_ci95_high": float(np.quantile(boot_chi2, 0.975)),
        "bootstrap_probability_candidate_improves_chi2": float(np.mean(boot_chi2 < 0.0)),
        "candidate_minus_baseline_rmse_kms": observed_rmse_delta,
        "rmse_bootstrap_ci95_low_kms": float(np.quantile(boot_rmse, 0.025)),
        "rmse_bootstrap_ci95_high_kms": float(np.quantile(boot_rmse, 0.975)),
        "galaxy_chi2_win_fraction": float(
            np.mean(merged["chi2_candidate"] < merged["chi2_baseline"])
        ),
    }


def _run_or_load(
    *,
    variant: dict[str, Any],
    heldout_fold: int,
    fold_data: PackedDataset,
    settings: ExperimentSettings,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    run_dir = output_dir / f"fold_{heldout_fold}" / variant["name"]
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "heldout_predictions.csv"
    galaxy_path = run_dir / "heldout_galaxy_metrics.csv"
    if summary_path.exists() and predictions_path.exists() and galaxy_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["optimization"]["steps"] == settings.steps:
            return summary, pd.read_csv(predictions_path), pd.read_csv(galaxy_path)

    set_reproducible_seed(settings.seed + heldout_fold)
    tensor_data = TensorDataset.from_packed(fold_data, device=device, dtype=torch.float64)
    model = build_model(
        variant["model"],
        fold_data,
        disk_ml_prior=settings.disk_mass_to_light_prior,
        bulge_ml_prior=settings.bulge_mass_to_light_prior,
        log_ml_prior_sigma=settings.log_mass_to_light_prior_sigma,
        rar_acceleration_m_s2=settings.rar_acceleration_m_s2,
        hubble_km_s_mpc=settings.hubble_km_s_mpc,
        environment_enabled=variant["environment_enabled"],
        boundary_layer_enabled=variant.get("boundary_layer_enabled", False),
    ).to(device=device, dtype=torch.float64)
    history: list[dict[str, float]] = []
    best_objective: float | None = None
    if variant["fit"]:
        history, best_objective = fit_model(
            model,
            tensor_data,
            steps=settings.steps,
            learning_rate=settings.learning_rate,
            error_floor_kms=settings.velocity_error_floor_kms,
            progress=False,
        )
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_data)
    frame = prediction_frame(fold_data, tensor_data, prediction, settings.velocity_error_floor_kms)
    heldout = frame.loc[frame["split"] == "outer_holdout"].copy()
    galaxy_metrics = _per_galaxy_metrics(frame)
    summary = {
        "fold": heldout_fold,
        "variant": variant["name"],
        "model": variant["model"],
        "environment_enabled": variant["environment_enabled"],
        "boundary_layer_enabled": variant.get("boundary_layer_enabled", False),
        "environment_score_column": variant["score_column"],
        "strict_galaxy_holdout": True,
        "heldout_nuisance_parameters": "prior centers; no heldout velocities used",
        "global_parameters": model.physical_parameters(),
        "best_objective": best_objective,
        "train": metrics(frame, "train"),
        "heldout_galaxies": metrics(frame, "outer_holdout"),
        "optimization": asdict(settings),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(predictions_path, index=False)
    galaxy_metrics.to_csv(galaxy_path, index=False)
    pd.DataFrame(history).to_csv(run_dir / "optimization_history.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if variant["fit"]:
        torch.save(model.state_dict(), run_dir / "model_state.pt")
    return summary, heldout, galaxy_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict galaxy-level cross-validation of CF4 environment coupling."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--environment-csv",
        type=Path,
        default=ROOT / "data" / "derived" / "void_scores_cf4.csv",
    )
    parser.add_argument(
        "--include-potential-extensions",
        action="store_true",
        help="Add the conditional P1 environment-threshold and B1 boundary tests.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cf4_galaxy_cv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument(
        "--include-potential",
        action="store_true",
        help="Add the preregistered P0 self-potential-screened model.",
    )
    args = parser.parse_args()

    settings = _settings(args.config, steps=args.steps)
    lookup = _score_lookup(args.environment_csv)
    packed = pack_dataset(args.data, environment_csv=args.environment_csv)
    raw_scores = {column: _environment_values(packed, lookup, column) for column in SCORE_COLUMNS}
    fold_assignment = _balanced_folds(raw_scores[SCORE_COLUMNS[0]], args.folds)
    assignments = pd.DataFrame(
        {
            "galaxy": packed.galaxy_names,
            "fold": fold_assignment,
            **{column: raw_scores[column] for column in SCORE_COLUMNS},
        }
    ).sort_values("galaxy")
    args.output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.output / "fold_assignments.csv", index=False)

    device = resolve_device(args.device)
    summaries: list[dict[str, Any]] = []
    heldout_frames: list[pd.DataFrame] = []
    galaxy_frames: list[pd.DataFrame] = []
    variants = _variant_specs(
        include_potential=args.include_potential,
        include_extensions=args.include_potential_extensions,
    )
    for heldout_fold in range(args.folds):
        for variant in variants:
            score_column = variant["score_column"]
            fold_data = _fold_dataset(
                packed,
                raw_scores[score_column],
                fold_assignment,
                heldout_fold,
                score_column,
            )
            print(f"fold={heldout_fold} variant={variant['name']}", flush=True)
            summary, heldout, per_galaxy = _run_or_load(
                variant=variant,
                heldout_fold=heldout_fold,
                fold_data=fold_data,
                settings=settings,
                device=device,
                output_dir=args.output,
            )
            summaries.append(summary)
            heldout.insert(0, "variant", variant["name"])
            heldout.insert(0, "fold", heldout_fold)
            heldout_frames.append(heldout)
            per_galaxy.insert(0, "variant", variant["name"])
            per_galaxy.insert(0, "fold", heldout_fold)
            galaxy_frames.append(per_galaxy)

    predictions = pd.concat(heldout_frames, ignore_index=True)
    galaxy_metrics = pd.concat(galaxy_frames, ignore_index=True)
    predictions.to_csv(args.output / "all_heldout_predictions.csv", index=False)
    galaxy_metrics.to_csv(args.output / "all_heldout_galaxy_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": summary["fold"],
                "variant": summary["variant"],
                "environment_beta": summary["global_parameters"].get("environment_beta"),
                "power_p": summary["global_parameters"].get("power_p"),
                "transition_width_dex": summary["global_parameters"].get("transition_width_dex"),
                "transition_potential_chi": summary["global_parameters"].get(
                    "transition_potential_chi"
                ),
                "environment_shift_zeta": summary["global_parameters"].get(
                    "environment_shift_zeta_dex_per_sigma"
                ),
                "boundary_kappa": summary["global_parameters"].get("boundary_kappa"),
                "heldout_chi2_per_point": summary["heldout_galaxies"]["chi2_per_point"],
                "heldout_rmse_kms": summary["heldout_galaxies"]["rmse_kms"],
            }
            for summary in summaries
        ]
    ).to_csv(args.output / "fold_summary.csv", index=False)

    by_variant = {}
    for variant, selected in galaxy_metrics.groupby("variant"):
        by_variant[variant] = {
            "galaxies": len(selected),
            "points": int(selected["n"].sum()),
            "chi2_per_point": float(selected["chi2"].sum() / selected["n"].sum()),
            "rmse_kms": float(math.sqrt(selected["squared_error"].sum() / selected["n"].sum())),
        }

    baseline = galaxy_metrics.loc[galaxy_metrics["variant"] == "void_no_environment"]
    comparisons = {}
    for column in SCORE_COLUMNS:
        variant = f"void_env_{column.removeprefix('void_score_')}"
        candidate = galaxy_metrics.loc[galaxy_metrics["variant"] == variant]
        comparison = _bootstrap_comparison(
            baseline,
            candidate,
            draws=args.bootstrap_draws,
            seed=settings.seed + len(comparisons),
        )
        beta = [
            float(summary["global_parameters"]["environment_beta"])
            for summary in summaries
            if summary["variant"] == variant
        ]
        comparison["fold_beta_values"] = beta
        comparison["all_fold_betas_positive"] = all(value > 0.0 for value in beta)
        comparison["mean_fold_beta"] = float(np.mean(beta))
        comparisons[variant] = comparison

    if args.include_potential:
        candidate = galaxy_metrics.loc[galaxy_metrics["variant"] == "potential_P0"]
        for baseline_name in ("void_no_environment", "rar_reference"):
            baseline_candidate = galaxy_metrics.loc[galaxy_metrics["variant"] == baseline_name]
            comparison = _bootstrap_comparison(
                baseline_candidate,
                candidate,
                draws=args.bootstrap_draws,
                seed=settings.seed + len(comparisons),
            )
            p0_parameters = [
                summary["global_parameters"]
                for summary in summaries
                if summary["variant"] == "potential_P0"
            ]
            comparison["fold_power_p"] = [
                float(parameters["power_p"]) for parameters in p0_parameters
            ]
            comparison["fold_transition_potential_chi"] = [
                float(parameters["transition_potential_chi"]) for parameters in p0_parameters
            ]
            comparisons[f"potential_P0_vs_{baseline_name}"] = comparison

    extension_holm: dict[str, Any] | None = None
    if args.include_potential_extensions:
        p0 = galaxy_metrics.loc[galaxy_metrics["variant"] == "potential_P0"]
        extension_specs = {
            "potential_P1_environment_vs_P0": (
                "potential_P1_environment",
                "environment_shift_zeta_dex_per_sigma",
            ),
            "potential_B1_boundary_vs_P0": ("potential_B1_boundary", "boundary_kappa"),
        }
        raw_p_values: dict[str, float] = {}
        for comparison_name, (variant_name, parameter_name) in extension_specs.items():
            candidate = galaxy_metrics.loc[galaxy_metrics["variant"] == variant_name]
            comparison = _bootstrap_comparison(
                p0,
                candidate,
                draws=args.bootstrap_draws,
                seed=settings.seed + len(comparisons),
            )
            parameter_values = [
                float(summary["global_parameters"][parameter_name])
                for summary in summaries
                if summary["variant"] == variant_name
            ]
            comparison["fold_parameter_name"] = parameter_name
            comparison["fold_parameter_values"] = parameter_values
            comparison["all_fold_parameters_positive"] = all(
                value > 0.0 for value in parameter_values
            )
            one_sided_p = 1.0 - float(comparison["bootstrap_probability_candidate_improves_chi2"])
            comparison["one_sided_bootstrap_p_improvement"] = one_sided_p
            raw_p_values[comparison_name] = one_sided_p
            comparisons[comparison_name] = comparison

        ordered = sorted(raw_p_values, key=raw_p_values.get)
        adjusted: dict[str, float] = {}
        running = 0.0
        family_size = len(ordered)
        for rank, name in enumerate(ordered):
            candidate_adjusted = min(1.0, (family_size - rank) * raw_p_values[name])
            running = max(running, candidate_adjusted)
            adjusted[name] = running
            comparisons[name]["holm_adjusted_one_sided_p"] = running
        extension_holm = {
            "family": list(extension_specs),
            "raw_one_sided_p": raw_p_values,
            "holm_adjusted_p": adjusted,
        }

    report = {
        "status": "completed strict galaxy-level cross-validation",
        "folds": args.folds,
        "steps_per_fitted_model": args.steps,
        "bootstrap_draws": args.bootstrap_draws,
        "include_potential_P0": args.include_potential,
        "include_potential_extensions": args.include_potential_extensions,
        "device": str(device),
        "design": (
            "Global parameters train on all radii of the non-held-out galaxy folds. "
            "Held-out galaxies use "
            "catalog baryonic inputs, CF4 score, and nuisance-prior centers; none of their "
            "rotation velocities enter optimization. Folds are balanced over the primary score."
        ),
        "variant_metrics": by_variant,
        "paired_environment_vs_no_environment": comparisons,
        "potential_extension_holm_family": extension_holm,
    }
    (args.output / "cross_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
