from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from voidscreen.data import PackedDataset, pack_dataset
from voidscreen.void_cage import (
    balanced_rank_folds,
    baryonic_velocity_squared,
    fixed_rar_velocity,
    harmonic_cage_velocity,
    screened_cage_velocity,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    score_column: str | None = None
    shuffled: bool = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_bounds(family: str, environment: bool) -> list[tuple[float, float]]:
    if family == "harmonic":
        bounds = [(-38.0, -27.0)]
    elif family == "screened":
        bounds = [(0.0, math.log10(500.0)), (-1.0, math.log10(20.0))]
    else:
        return []
    if environment:
        bounds.append((0.0, 4.0))
    return bounds


def predict(
    variant: Variant,
    parameters: np.ndarray,
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    environment: np.ndarray | None,
) -> np.ndarray:
    if variant.family == "newtonian":
        return np.sqrt(baryonic_v2)
    if variant.family == "rar":
        return fixed_rar_velocity(packed, baryonic_v2)
    exponent = float(parameters[-1]) if environment is not None else 0.0
    if variant.family == "harmonic":
        return harmonic_cage_velocity(
            packed,
            baryonic_v2,
            log10_kappa_s2=float(parameters[0]),
            environment_by_galaxy=environment,
            environment_exponent=exponent,
        )
    if variant.family == "screened":
        return screened_cage_velocity(
            packed,
            baryonic_v2,
            log10_velocity_scale_km_s=float(parameters[0]),
            log10_transition_scale_lengths=float(parameters[1]),
            environment_by_galaxy=environment,
            environment_exponent=exponent,
        )
    raise ValueError(f"Unknown family {variant.family}")


def fit_parameters(
    variant: Variant,
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    environment: np.ndarray | None,
    train_points: np.ndarray,
    sigma: np.ndarray,
    *,
    starts: int,
    seed: int,
) -> tuple[np.ndarray, float, bool]:
    bounds = parameter_bounds(variant.family, environment is not None)
    if not bounds:
        prediction = predict(variant, np.empty(0), packed, baryonic_v2, environment)
        objective = float(
            np.sum(((prediction[train_points] - packed.velocity_observed_kms[train_points]) / sigma[train_points]) ** 2)
        )
        return np.empty(0), objective, True

    def objective(values: np.ndarray) -> float:
        prediction = predict(variant, values, packed, baryonic_v2, environment)
        residual = (
            prediction[train_points] - packed.velocity_observed_kms[train_points]
        ) / sigma[train_points]
        if not np.isfinite(residual).all():
            return 1e300
        return float(np.sum(residual**2))

    rng = np.random.default_rng(seed)
    starts_list = [np.asarray([(low + high) / 2.0 for low, high in bounds])]
    if environment is not None:
        nested = starts_list[0].copy()
        nested[-1] = 0.0
        starts_list.append(nested)
    while len(starts_list) < starts:
        starts_list.append(np.asarray([rng.uniform(low, high) for low, high in bounds]))
    best = None
    for initial in starts_list:
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None
    return np.asarray(best.x, dtype=np.float64), float(best.fun), bool(best.success)


def normalize_environment(raw: np.ndarray, training_galaxies: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(raw, dtype=np.float64)
    normalization = float(np.median(values[training_galaxies]))
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Training-fold cage score must have a positive finite median")
    return values / normalization, normalization


def variant_parameters(variant: Variant, values: np.ndarray) -> dict[str, float]:
    if variant.family == "harmonic":
        output = {"log10_kappa_s2": float(values[0])}
    elif variant.family == "screened":
        output = {
            "velocity_scale_km_s": float(10.0 ** values[0]),
            "transition_scale_lengths": float(10.0 ** values[1]),
        }
    else:
        return {}
    if variant.score_column is not None:
        output["environment_exponent"] = float(values[-1])
    return output


def per_galaxy_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, galaxy), selected in predictions.groupby(["variant", "galaxy"], sort=False):
        residual = selected["velocity_predicted_kms"] - selected["velocity_observed_kms"]
        rows.append(
            {
                "variant": variant,
                "galaxy": galaxy,
                "fold": int(selected["fold"].iloc[0]),
                "n": len(selected),
                "chi2": float(np.sum((residual / selected["velocity_error_total_kms"]) ** 2)),
                "squared_error": float(np.sum(residual**2)),
                "mean_residual_kms": float(np.mean(residual)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(per_galaxy: pd.DataFrame) -> dict[str, float | int]:
    points = int(per_galaxy["n"].sum())
    return {
        "galaxies": len(per_galaxy),
        "points": points,
        "chi2_per_point": float(per_galaxy["chi2"].sum() / points),
        "rmse_kms": float(math.sqrt(per_galaxy["squared_error"].sum() / points)),
        "macro_mean_abs_residual_kms": float(per_galaxy["mean_residual_kms"].abs().mean()),
    }


def bootstrap_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    merged = baseline.merge(candidate, on="galaxy", suffixes=("_baseline", "_candidate"))
    if len(merged) != len(baseline) or not np.array_equal(
        merged["n_baseline"], merged["n_candidate"]
    ):
        raise ValueError("Paired comparison requires identical heldout galaxies and point counts")
    count = merged["n_baseline"].to_numpy(dtype=float)
    delta_chi2 = (merged["chi2_candidate"] - merged["chi2_baseline"]).to_numpy()
    base_sse = merged["squared_error_baseline"].to_numpy(dtype=float)
    candidate_sse = merged["squared_error_candidate"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    chi_draws = []
    rmse_draws = []
    for start in range(0, draws, 10_000):
        chunk = min(10_000, draws - start)
        indices = rng.integers(0, len(merged), size=(chunk, len(merged)))
        total = count[indices].sum(axis=1)
        chi_draws.append(delta_chi2[indices].sum(axis=1) / total)
        rmse_draws.append(
            np.sqrt(candidate_sse[indices].sum(axis=1) / total)
            - np.sqrt(base_sse[indices].sum(axis=1) / total)
        )
    chi = np.concatenate(chi_draws)
    rmse = np.concatenate(rmse_draws)
    observed_chi = float(np.sum(delta_chi2) / np.sum(count))
    observed_rmse = float(
        math.sqrt(float(np.sum(candidate_sse) / np.sum(count)))
        - math.sqrt(float(np.sum(base_sse) / np.sum(count)))
    )
    return {
        "galaxies": len(merged),
        "points": int(np.sum(count)),
        "candidate_minus_baseline_chi2_per_point": observed_chi,
        "chi2_ci95_low": float(np.quantile(chi, 0.025)),
        "chi2_ci95_high": float(np.quantile(chi, 0.975)),
        "bootstrap_probability_candidate_improves_chi2": float(np.mean(chi < 0.0)),
        "candidate_minus_baseline_rmse_kms": observed_rmse,
        "rmse_ci95_low_kms": float(np.quantile(rmse, 0.025)),
        "rmse_ci95_high_kms": float(np.quantile(rmse, 0.975)),
        "galaxy_chi2_win_fraction": float(np.mean(delta_chi2 < 0.0)),
    }


def run_cross_validation(
    variants: list[Variant],
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    scores: dict[str, np.ndarray],
    fold_assignment: np.ndarray,
    sigma: np.ndarray,
    *,
    optimizer_starts: int,
    seed: int,
    shuffled_primary: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    prediction_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    for heldout_fold in range(int(fold_assignment.max()) + 1):
        heldout_galaxies = fold_assignment == heldout_fold
        training_galaxies = ~heldout_galaxies
        train_points = training_galaxies[packed.galaxy_index]
        heldout_points = heldout_galaxies[packed.galaxy_index]
        for variant_index, variant in enumerate(variants):
            raw_environment = None
            normalization = None
            if variant.score_column is not None:
                raw = shuffled_primary if variant.shuffled else scores[variant.score_column]
                raw_environment, normalization = normalize_environment(raw, training_galaxies)
            parameters, train_chi2, converged = fit_parameters(
                variant,
                packed,
                baryonic_v2,
                raw_environment,
                train_points,
                sigma,
                starts=optimizer_starts,
                seed=seed + 1000 * heldout_fold + variant_index,
            )
            predicted = predict(variant, parameters, packed, baryonic_v2, raw_environment)
            selected = np.flatnonzero(heldout_points)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "variant": variant.name,
                        "fold": heldout_fold,
                        "galaxy": np.asarray(packed.galaxy_names, dtype=object)[
                            packed.galaxy_index[selected]
                        ],
                        "radius_kpc": packed.radius_kpc[selected],
                        "velocity_observed_kms": packed.velocity_observed_kms[selected],
                        "velocity_error_total_kms": sigma[selected],
                        "velocity_baryonic_kms": np.sqrt(baryonic_v2[selected]),
                        "velocity_predicted_kms": predicted[selected],
                    }
                )
            )
            fold_summaries.append(
                {
                    "fold": heldout_fold,
                    "variant": variant.name,
                    "family": variant.family,
                    "score_column": variant.score_column,
                    "shuffled": variant.shuffled,
                    "environment_normalization": normalization,
                    "parameters": variant_parameters(variant, parameters),
                    "train_chi2": train_chi2,
                    "optimizer_converged": converged,
                }
            )
            print(f"fold={heldout_fold} variant={variant.name}", flush=True)
    return pd.concat(prediction_frames, ignore_index=True), fold_summaries


def permutation_distribution(
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    primary_score: np.ndarray,
    fold_assignment: np.ndarray,
    sigma: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> list[dict[str, float | int]]:
    variant = Variant("screened_environment_permutation", "screened", "permuted")
    rng = np.random.default_rng(seed)
    output = []
    for permutation in range(permutations):
        shuffled = rng.permutation(primary_score)
        total_chi2 = 0.0
        total_squared_error = 0.0
        total_points = 0
        exponents = []
        for heldout_fold in range(int(fold_assignment.max()) + 1):
            heldout_galaxies = fold_assignment == heldout_fold
            training_galaxies = ~heldout_galaxies
            train_points = training_galaxies[packed.galaxy_index]
            heldout_points = heldout_galaxies[packed.galaxy_index]
            environment, _ = normalize_environment(shuffled, training_galaxies)
            parameters, _, _ = fit_parameters(
                variant,
                packed,
                baryonic_v2,
                environment,
                train_points,
                sigma,
                starts=4,
                seed=seed + permutation * 101 + heldout_fold,
            )
            predicted = predict(variant, parameters, packed, baryonic_v2, environment)
            residual = predicted[heldout_points] - packed.velocity_observed_kms[heldout_points]
            total_chi2 += float(np.sum((residual / sigma[heldout_points]) ** 2))
            total_squared_error += float(np.sum(residual**2))
            total_points += int(np.sum(heldout_points))
            exponents.append(float(parameters[-1]))
        output.append(
            {
                "permutation": permutation,
                "chi2_per_point": total_chi2 / total_points,
                "rmse_kms": math.sqrt(total_squared_error / total_points),
                "mean_environment_exponent": float(np.mean(exponents)),
            }
        )
    return output


def plot_metrics(metrics: dict[str, dict[str, float | int]], output: Path) -> None:
    ordered = sorted(metrics, key=lambda name: float(metrics[name]["rmse_kms"]))
    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    values = [float(metrics[name]["rmse_kms"]) for name in ordered]
    axis.barh(np.arange(len(ordered)), values, color="#35618f")
    axis.set_yticks(np.arange(len(ordered)), labels=ordered)
    axis.invert_yaxis()
    axis.set_xlabel("Whole-galaxy heldout RMSE (km/s; lower is better)")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered SPARC void-cage test.")
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs" / "void_cage_test_protocol.json"
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--geometry", type=Path, default=ROOT / "data" / "derived" / "void_cage_geometry.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "void_cage_test"
    )
    parser.add_argument("--permutations", type=int, default=None)
    args = parser.parse_args()
    for name in ("protocol", "data", "geometry", "output"):
        path = getattr(args, name)
        setattr(args, name, path if path.is_absolute() else ROOT / path)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    sample = protocol["rotation_sample"]
    validation = protocol["validation"]
    primary_range = protocol["kernels"]["yukawa_primary"]["range_h100_inverse_mpc"]
    primary_column = f"grouped_64_yukawa_l{str(primary_range).replace('.', 'p')}_kappa_unit"
    geometry = pd.read_csv(args.geometry).set_index("galaxy")
    packed = pack_dataset(
        args.data,
        quality_max=int(sample["quality_max"]),
        minimum_inclination_deg=float(sample["minimum_inclination_deg"]),
        minimum_points=int(sample["minimum_points"]),
        environment_csv=args.geometry,
        environment_score_column=primary_column,
    )
    if packed.n_galaxies != int(sample["expected_galaxies"]) or packed.n_points != int(
        sample["expected_points"]
    ):
        raise RuntimeError(
            f"Frozen sample expected {sample['expected_galaxies']}/{sample['expected_points']}, "
            f"found {packed.n_galaxies}/{packed.n_points}"
        )
    baryonic_v2 = baryonic_velocity_squared(
        packed,
        disk_mass_to_light=float(sample["disk_mass_to_light"]),
        bulge_mass_to_light=float(sample["bulge_mass_to_light"]),
    )
    sigma = np.sqrt(
        packed.velocity_error_kms**2 + float(sample["velocity_error_floor_km_s"]) ** 2
    )

    ranges = [
        primary_range,
        *protocol["kernels"]["yukawa_range_robustness_h100_inverse_mpc"],
    ]
    ranges = list(dict.fromkeys(ranges))
    score_columns = [
        f"{grid}_yukawa_l{str(value).replace('.', 'p')}_kappa_unit"
        for grid in ("grouped_64", "ungrouped_64", "ungrouped_128")
        for value in ranges
        if grid == "grouped_64" or value == primary_range
    ]
    score_columns.append("grouped_64_power_p3_kappa_unit")
    scores = {
        column: geometry.loc[list(packed.galaxy_names), column].to_numpy(dtype=float)
        for column in score_columns
    }
    fold_assignment = balanced_rank_folds(
        np.log(scores[primary_column]), int(validation["folds"])
    )
    rng = np.random.default_rng(int(validation["seed"]))
    shuffled_primary = rng.permutation(scores[primary_column])

    variants = [
        Variant("newtonian_baryons", "newtonian"),
        Variant("fixed_rar", "rar"),
        Variant("direct_harmonic_blind", "harmonic"),
        Variant("direct_harmonic_primary", "harmonic", primary_column),
        Variant("screened_radial_blind", "screened"),
        Variant("screened_primary", "screened", primary_column),
        Variant("screened_primary_shuffled", "screened", primary_column, shuffled=True),
    ]
    for column in score_columns:
        if column != primary_column:
            variants.append(Variant(f"screened_{column.removesuffix('_kappa_unit')}", "screened", column))

    predictions, folds = run_cross_validation(
        variants,
        packed,
        baryonic_v2,
        scores,
        fold_assignment,
        sigma,
        optimizer_starts=int(validation["optimizer_starts"]),
        seed=int(validation["seed"]),
        shuffled_primary=shuffled_primary,
    )
    galaxy_metrics = per_galaxy_metrics(predictions)
    metrics = {
        variant: aggregate_metrics(selected)
        for variant, selected in galaxy_metrics.groupby("variant")
    }
    comparisons = {}
    for baseline in ("screened_radial_blind", "fixed_rar"):
        comparisons[f"screened_primary_vs_{baseline}"] = bootstrap_comparison(
            galaxy_metrics.loc[galaxy_metrics["variant"] == baseline],
            galaxy_metrics.loc[galaxy_metrics["variant"] == "screened_primary"],
            draws=int(validation["paired_bootstrap_draws"]),
            seed=int(validation["seed"]) + len(comparisons),
        )
    for baseline in ("direct_harmonic_blind", "fixed_rar"):
        comparisons[f"direct_harmonic_primary_vs_{baseline}"] = bootstrap_comparison(
            galaxy_metrics.loc[galaxy_metrics["variant"] == baseline],
            galaxy_metrics.loc[galaxy_metrics["variant"] == "direct_harmonic_primary"],
            draws=int(validation["paired_bootstrap_draws"]),
            seed=int(validation["seed"]) + len(comparisons),
        )

    permutations = (
        int(validation["environment_permutations"])
        if args.permutations is None
        else int(args.permutations)
    )
    permutation_rows = permutation_distribution(
        packed,
        baryonic_v2,
        scores[primary_column],
        fold_assignment,
        sigma,
        permutations=permutations,
        seed=int(validation["seed"]) + 20_000,
    )
    permutation_frame = pd.DataFrame(permutation_rows)
    real_chi2 = float(metrics["screened_primary"]["chi2_per_point"])
    permutation_percentile = float(np.mean(real_chi2 < permutation_frame["chi2_per_point"]))

    primary_fold_parameters = [
        entry["parameters"]
        for entry in folds
        if entry["variant"] == "screened_primary"
    ]
    exponents = [float(entry["environment_exponent"]) for entry in primary_fold_parameters]
    reconstruction_variants = {
        "ungrouped_64": "screened_ungrouped_64_yukawa_l15p625",
        "ungrouped_128": "screened_ungrouped_128_yukawa_l15p625",
    }
    reconstruction_exponents = {
        key: [
            float(entry["parameters"]["environment_exponent"])
            for entry in folds
            if entry["variant"] == name
        ]
        for key, name in reconstruction_variants.items()
    }
    gates_config = protocol["primary_success_gates"]
    primary_vs_blind = comparisons["screened_primary_vs_screened_radial_blind"]
    primary_vs_rar = comparisons["screened_primary_vs_fixed_rar"]
    gates = {
        "rmse_5pct_better_than_screened_blind": float(metrics["screened_primary"]["rmse_kms"])
        <= float(gates_config["screened_environment_vs_screened_blind_rmse_fraction_max"])
        * float(metrics["screened_radial_blind"]["rmse_kms"]),
        "rmse_5pct_better_than_fixed_rar": float(metrics["screened_primary"]["rmse_kms"])
        <= float(gates_config["screened_environment_vs_fixed_rar_rmse_fraction_max"])
        * float(metrics["fixed_rar"]["rmse_kms"]),
        "bootstrap_vs_screened_blind": float(
            primary_vs_blind["bootstrap_probability_candidate_improves_chi2"]
        )
        >= float(gates_config["paired_bootstrap_probability_improves_min"]),
        "bootstrap_vs_fixed_rar": float(
            primary_vs_rar["bootstrap_probability_candidate_improves_chi2"]
        )
        >= float(gates_config["paired_bootstrap_probability_improves_min"]),
        "environment_exponent_positive_all_folds": all(value > 0.0 for value in exponents),
        "environment_exponent_not_at_bound_all_folds": all(
            1e-3 < value < 4.0 - 1e-3 for value in exponents
        ),
        "real_environment_beats_95pct_permutations": permutation_percentile
        >= float(gates_config["real_score_permutation_percentile_min"]),
        "reconstruction_sign_stable": all(
            values and all(value > 0.0 for value in values)
            for values in reconstruction_exponents.values()
        ),
        "no_lensing_only_parameter": True,
    }
    primary_pass = all(gates.values())
    direct_comparison = comparisons["direct_harmonic_primary_vs_direct_harmonic_blind"]
    direct_pass = (
        float(metrics["direct_harmonic_primary"]["rmse_kms"])
        <= 0.95 * float(metrics["direct_harmonic_blind"]["rmse_kms"])
        and float(direct_comparison["bootstrap_probability_candidate_improves_chi2"]) >= 0.95
    )

    fold_table = pd.DataFrame(
        [
            {
                "fold": entry["fold"],
                "variant": entry["variant"],
                "score_column": entry["score_column"],
                "optimizer_converged": entry["optimizer_converged"],
                **entry["parameters"],
            }
            for entry in folds
        ]
    )
    assignment_table = pd.DataFrame(
        {
            "galaxy": packed.galaxy_names,
            "fold": fold_assignment,
            "primary_cage_score": scores[primary_column],
        }
    ).sort_values("galaxy")

    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "heldout_predictions.csv", index=False)
    galaxy_metrics.to_csv(args.output / "heldout_galaxy_metrics.csv", index=False)
    fold_table.to_csv(args.output / "fold_parameters.csv", index=False)
    assignment_table.to_csv(args.output / "fold_assignments.csv", index=False)
    permutation_frame.to_csv(args.output / "environment_permutations.csv", index=False)
    plot_metrics(metrics, args.output / "heldout_rmse_comparison.png")

    report = {
        "status": "completed preregistered whole-galaxy void-cage test",
        "report_version": "void-cage-test-0.1",
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": sha256(args.protocol),
            "version": protocol["protocol_version"],
        },
        "inputs": {
            "geometry": {"path": str(args.geometry.relative_to(ROOT)), "sha256": sha256(args.geometry)},
            "SPARC_data_fingerprint": packed.data_fingerprint,
        },
        "design": {
            "galaxies": packed.n_galaxies,
            "points": packed.n_points,
            "folds": int(validation["folds"]),
            "holdout_unit": "whole galaxy",
            "baryonic_nuisance_rule": "fixed M/L and catalog distance/inclination for every model",
            "primary_score": primary_column,
            "optimizer_starts": int(validation["optimizer_starts"]),
            "permutations": permutations,
        },
        "variant_metrics": metrics,
        "paired_comparisons": comparisons,
        "primary_fold_environment_exponents": exponents,
        "reconstruction_fold_environment_exponents": reconstruction_exponents,
        "permutation_test": {
            "real_primary_chi2_per_point": real_chi2,
            "fraction_permutations_worse_than_real": permutation_percentile,
            "permutation_chi2_minimum": float(permutation_frame["chi2_per_point"].min()),
            "permutation_chi2_median": float(permutation_frame["chi2_per_point"].median()),
            "permutation_chi2_maximum": float(permutation_frame["chi2_per_point"].max()),
        },
        "gates": gates,
        "primary_screened_cage_pass": primary_pass,
        "literal_external_harmonic_cage_pass": direct_pass,
        "decision": {
            "literal_external_cage": "retain" if direct_pass else "reject_tested_literal_external_cage",
            "screened_void_cage": "retain_for_lensing" if primary_pass else "reject_tested_screened_conversion",
            "void_origin": "supported" if primary_pass else "not_supported",
            "lensing_unification": "not_claimed_pending_theory_neutral_same_system_data",
        },
        "interpretation_guardrails": protocol["known_identifiability_limits"],
        "artifacts": {
            name: {
                "path": str((args.output / name).relative_to(ROOT)),
                "sha256": sha256(args.output / name),
            }
            for name in (
                "heldout_predictions.csv",
                "heldout_galaxy_metrics.csv",
                "fold_parameters.csv",
                "fold_assignments.csv",
                "environment_permutations.csv",
                "heldout_rmse_comparison.png",
            )
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
