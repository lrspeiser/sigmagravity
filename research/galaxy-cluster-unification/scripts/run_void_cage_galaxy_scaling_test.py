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
from voidscreen.galaxy_scaling import (
    GalaxyPredictors,
    catalog_scaled_screened_velocity,
    load_sparc_structural_predictors,
    local_acceleration_screened_velocity,
    normalize_positive_by_training_median,
)
from voidscreen.void_cage import (
    baryonic_velocity_squared,
    fixed_rar_velocity,
    screened_cage_velocity,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    transition_driver: str | None = None
    score_column: str | None = None
    shuffled: bool = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_bounds(variant: Variant) -> list[tuple[float, float]]:
    if variant.family == "legacy":
        bounds = [(0.0, math.log10(500.0)), (-1.0, math.log10(20.0))]
    elif variant.family == "local":
        bounds = [(0.0, math.log10(500.0)), (-13.0, -8.0), (0.1, 4.0)]
    elif variant.family == "catalog":
        bounds = [
            (0.0, math.log10(500.0)),
            (-1.0, math.log10(20.0)),
            (-2.0, 2.0),
            (-2.0, 2.0),
        ]
    else:
        return []
    if variant.score_column is not None:
        bounds.append((0.0, 4.0))
    return bounds


def predict(
    variant: Variant,
    parameters: np.ndarray,
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    normalized_predictors: dict[str, np.ndarray],
    environment: np.ndarray | None,
) -> np.ndarray:
    if variant.family == "newtonian":
        return np.sqrt(baryonic_v2)
    if variant.family == "rar":
        return fixed_rar_velocity(packed, baryonic_v2)
    if variant.family == "legacy":
        return screened_cage_velocity(
            packed,
            baryonic_v2,
            log10_velocity_scale_km_s=float(parameters[0]),
            log10_transition_scale_lengths=float(parameters[1]),
        )
    environment_exponent = (
        float(parameters[-1]) if variant.score_column is not None else 0.0
    )
    if variant.family == "local":
        return local_acceleration_screened_velocity(
            packed,
            baryonic_v2,
            log10_velocity_scale_km_s=float(parameters[0]),
            log10_gstar_m_s2=float(parameters[1]),
            screening_power=float(parameters[2]),
            environment_by_galaxy=environment,
            environment_exponent=environment_exponent,
        )
    if variant.family == "catalog":
        assert variant.transition_driver in {"surface", "concentration"}
        return catalog_scaled_screened_velocity(
            packed,
            baryonic_v2,
            mass_by_galaxy=normalized_predictors["mass"],
            transition_driver_by_galaxy=normalized_predictors[
                variant.transition_driver
            ],
            log10_velocity_scale_km_s=float(parameters[0]),
            log10_transition_scale_lengths=float(parameters[1]),
            mass_amplitude_exponent=float(parameters[2]),
            transition_exponent=float(parameters[3]),
            environment_by_galaxy=environment,
            environment_exponent=environment_exponent,
        )
    raise ValueError(f"Unknown family {variant.family}")


def fit_parameters(
    variant: Variant,
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    normalized_predictors: dict[str, np.ndarray],
    environment: np.ndarray | None,
    train_points: np.ndarray,
    sigma: np.ndarray,
    *,
    starts: int,
    seed: int,
) -> tuple[np.ndarray, float, bool]:
    bounds = parameter_bounds(variant)
    if not bounds:
        prediction = predict(
            variant,
            np.empty(0),
            packed,
            baryonic_v2,
            normalized_predictors,
            environment,
        )
        residual = (
            prediction[train_points] - packed.velocity_observed_kms[train_points]
        ) / sigma[train_points]
        return np.empty(0), float(np.sum(residual**2)), True

    def objective(values: np.ndarray) -> float:
        prediction = predict(
            variant,
            values,
            packed,
            baryonic_v2,
            normalized_predictors,
            environment,
        )
        residual = (
            prediction[train_points] - packed.velocity_observed_kms[train_points]
        ) / sigma[train_points]
        if not np.isfinite(residual).all():
            return 1e300
        return float(np.sum(residual**2))

    midpoint = np.asarray([(low + high) / 2.0 for low, high in bounds])
    starts_list = [midpoint]
    if variant.family == "catalog":
        nested = midpoint.copy()
        nested[2:4] = 0.0
        if variant.score_column is not None:
            nested[-1] = 0.0
        starts_list.append(nested)
    elif variant.score_column is not None:
        nested = midpoint.copy()
        nested[-1] = 0.0
        starts_list.append(nested)
    rng = np.random.default_rng(seed)
    while len(starts_list) < starts:
        starts_list.append(
            np.asarray([rng.uniform(low, high) for low, high in bounds])
        )
    best = None
    for initial in starts_list:
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None
    return np.asarray(best.x), float(best.fun), bool(best.success)


def parameter_record(variant: Variant, values: np.ndarray) -> dict[str, float]:
    if variant.family == "legacy":
        output = {
            "velocity_scale_km_s": float(10.0 ** values[0]),
            "transition_scale_lengths": float(10.0 ** values[1]),
        }
    elif variant.family == "local":
        output = {
            "velocity_scale_km_s": float(10.0 ** values[0]),
            "log10_gstar_m_s2": float(values[1]),
            "screening_power_n": float(values[2]),
        }
    elif variant.family == "catalog":
        output = {
            "velocity_scale_km_s": float(10.0 ** values[0]),
            "transition_scale_lengths": float(10.0 ** values[1]),
            "mass_amplitude_exponent_eta": float(values[2]),
            "transition_exponent": float(values[3]),
        }
    else:
        return {}
    if variant.score_column is not None:
        output["void_exponent_m"] = float(values[-1])
    return output


def normalized_predictors_for_fold(
    predictors: GalaxyPredictors, training_galaxies: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    source = {
        "mass": predictors.mass_proxy_1e9_msun,
        "surface": predictors.central_stellar_surface_density_msun_pc2,
        "concentration": predictors.concentration_rdisk_over_reff,
    }
    normalized: dict[str, np.ndarray] = {}
    medians: dict[str, float] = {}
    for name, values in source.items():
        normalized[name], medians[name] = normalize_positive_by_training_median(
            values, training_galaxies
        )
    return normalized, medians


def run_cross_validation(
    variants: list[Variant],
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    predictors: GalaxyPredictors,
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
        normalized_predictors, predictor_medians = normalized_predictors_for_fold(
            predictors, training_galaxies
        )
        for variant_index, variant in enumerate(variants):
            environment = None
            environment_median = None
            if variant.score_column is not None:
                raw = (
                    shuffled_primary
                    if variant.shuffled
                    else scores[variant.score_column]
                )
                environment, environment_median = (
                    normalize_positive_by_training_median(raw, training_galaxies)
                )
            parameters, train_chi2, converged = fit_parameters(
                variant,
                packed,
                baryonic_v2,
                normalized_predictors,
                environment,
                train_points,
                sigma,
                starts=optimizer_starts,
                seed=seed + 1000 * heldout_fold + variant_index,
            )
            predicted = predict(
                variant,
                parameters,
                packed,
                baryonic_v2,
                normalized_predictors,
                environment,
            )
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
                        "velocity_observed_kms": packed.velocity_observed_kms[
                            selected
                        ],
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
                    "transition_driver": variant.transition_driver,
                    "score_column": variant.score_column,
                    "shuffled": variant.shuffled,
                    "predictor_training_medians": predictor_medians,
                    "environment_training_median": environment_median,
                    "parameters": parameter_record(variant, parameters),
                    "train_chi2": train_chi2,
                    "optimizer_converged": converged,
                }
            )
            print(f"fold={heldout_fold} variant={variant.name}", flush=True)
    return pd.concat(prediction_frames, ignore_index=True), fold_summaries


def per_galaxy_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, galaxy), selected in predictions.groupby(
        ["variant", "galaxy"], sort=False
    ):
        residual = (
            selected["velocity_predicted_kms"] - selected["velocity_observed_kms"]
        )
        rows.append(
            {
                "variant": variant,
                "galaxy": galaxy,
                "fold": int(selected["fold"].iloc[0]),
                "n": len(selected),
                "chi2": float(
                    np.sum((residual / selected["velocity_error_total_kms"]) ** 2)
                ),
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
        "rmse_kms": float(
            math.sqrt(per_galaxy["squared_error"].sum() / points)
        ),
        "macro_mean_abs_residual_kms": float(
            per_galaxy["mean_residual_kms"].abs().mean()
        ),
    }


def bootstrap_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    merged = baseline.merge(candidate, on="galaxy", suffixes=("_baseline", "_candidate"))
    count = merged["n_baseline"].to_numpy(dtype=float)
    delta_chi2 = (
        merged["chi2_candidate"] - merged["chi2_baseline"]
    ).to_numpy(dtype=float)
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
    return {
        "galaxies": len(merged),
        "points": int(np.sum(count)),
        "candidate_minus_baseline_chi2_per_point": float(
            np.sum(delta_chi2) / np.sum(count)
        ),
        "chi2_ci95_low": float(np.quantile(chi, 0.025)),
        "chi2_ci95_high": float(np.quantile(chi, 0.975)),
        "bootstrap_probability_candidate_improves_chi2": float(
            np.mean(chi < 0.0)
        ),
        "candidate_minus_baseline_rmse_kms": float(
            math.sqrt(np.sum(candidate_sse) / np.sum(count))
            - math.sqrt(np.sum(base_sse) / np.sum(count))
        ),
        "rmse_ci95_low_kms": float(np.quantile(rmse, 0.025)),
        "rmse_ci95_high_kms": float(np.quantile(rmse, 0.975)),
        "galaxy_chi2_win_fraction": float(np.mean(delta_chi2 < 0.0)),
    }


def permutation_distribution(
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    predictors: GalaxyPredictors,
    primary_score: np.ndarray,
    fold_assignment: np.ndarray,
    sigma: np.ndarray,
    *,
    permutations: int,
    starts: int,
    seed: int,
) -> list[dict[str, float | int]]:
    variant = Variant("local_void_permutation", "local", score_column="permuted")
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
            normalized_predictors, _ = normalized_predictors_for_fold(
                predictors, training_galaxies
            )
            environment, _ = normalize_positive_by_training_median(
                shuffled, training_galaxies
            )
            parameters, _, _ = fit_parameters(
                variant,
                packed,
                baryonic_v2,
                normalized_predictors,
                environment,
                train_points,
                sigma,
                starts=starts,
                seed=seed + 101 * permutation + heldout_fold,
            )
            predicted = predict(
                variant,
                parameters,
                packed,
                baryonic_v2,
                normalized_predictors,
                environment,
            )
            residual = (
                predicted[heldout_points]
                - packed.velocity_observed_kms[heldout_points]
            )
            total_chi2 += float(
                np.sum((residual / sigma[heldout_points]) ** 2)
            )
            total_squared_error += float(np.sum(residual**2))
            total_points += int(np.sum(heldout_points))
            exponents.append(float(parameters[-1]))
        output.append(
            {
                "permutation": permutation,
                "chi2_per_point": total_chi2 / total_points,
                "rmse_kms": math.sqrt(total_squared_error / total_points),
                "mean_void_exponent_m": float(np.mean(exponents)),
            }
        )
        print(f"permutation={permutation + 1}/{permutations}", flush=True)
    return output


def outermost_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    selected = predictions.loc[
        predictions.groupby(["variant", "galaxy"])["radius_kpc"].idxmax()
    ].copy()
    selected["residual"] = (
        selected["velocity_predicted_kms"] - selected["velocity_observed_kms"]
    )
    output = {}
    for variant, frame in selected.groupby("variant"):
        output[str(variant)] = {
            "galaxies": len(frame),
            "rmse_kms": float(np.sqrt(np.mean(frame["residual"] ** 2))),
            "mean_residual_kms": float(frame["residual"].mean()),
            "mean_absolute_residual_kms": float(frame["residual"].abs().mean()),
        }
    return output


def plot_metrics(metrics: dict[str, dict[str, float | int]], output: Path) -> None:
    ordered = sorted(metrics, key=lambda name: float(metrics[name]["rmse_kms"]))
    figure, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    axis.barh(
        np.arange(len(ordered)),
        [float(metrics[name]["rmse_kms"]) for name in ordered],
        color="#35618f",
    )
    axis.set_yticks(np.arange(len(ordered)), labels=ordered)
    axis.invert_yaxis()
    axis.set_xlabel("Whole-galaxy heldout RMSE (km/s; lower is better)")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def pair_gate(
    metrics: dict[str, dict[str, float | int]],
    comparisons: dict[str, dict[str, float | int]],
    candidate: str,
    baseline: str,
    *,
    rmse_fraction_max: float,
    bootstrap_min: float,
) -> dict[str, bool | float]:
    comparison = comparisons[f"{candidate}_vs_{baseline}"]
    ratio = float(metrics[candidate]["rmse_kms"]) / float(
        metrics[baseline]["rmse_kms"]
    )
    probability = float(
        comparison["bootstrap_probability_candidate_improves_chi2"]
    )
    return {
        "rmse_ratio": ratio,
        "rmse_gate": ratio <= rmse_fraction_max,
        "bootstrap_probability": probability,
        "bootstrap_gate": probability >= bootstrap_min,
        "pair_pass": ratio <= rmse_fraction_max and probability >= bootstrap_min,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "void_cage_galaxy_scaling_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "void_cage_galaxy_scaling_test",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256(args.protocol)
    inputs = protocol["inputs"]
    sample = protocol["sample"]
    validation = protocol["validation"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sparc_dir = ROOT / inputs["SPARC_directory"]
    metadata_path = ROOT / inputs["SPARC_metadata"]
    geometry_path = ROOT / inputs["void_geometry"]
    fold_path = ROOT / inputs["fold_assignments"]
    if sha256(fold_path) != inputs["expected_fold_assignment_sha256"]:
        raise ValueError("Frozen fold-assignment hash does not match the protocol")

    packed = pack_dataset(
        sparc_dir,
        quality_max=int(sample["quality_max"]),
        minimum_inclination_deg=float(sample["minimum_inclination_deg"]),
        minimum_points=int(sample["minimum_points"]),
    )
    if packed.n_galaxies != int(sample["expected_galaxies"]) or packed.n_points != int(
        sample["expected_points"]
    ):
        raise ValueError("SPARC sample does not match the frozen expected size")
    baryonic_v2 = baryonic_velocity_squared(
        packed,
        disk_mass_to_light=float(sample["disk_mass_to_light"]),
        bulge_mass_to_light=float(sample["bulge_mass_to_light"]),
    )
    sigma = np.sqrt(
        packed.velocity_error_kms**2
        + float(sample["velocity_error_floor_km_s"]) ** 2
    )
    predictors = load_sparc_structural_predictors(
        metadata_path, packed.galaxy_names
    )

    folds = pd.read_csv(fold_path).set_index("galaxy")
    if set(folds.index) != set(packed.galaxy_names):
        raise ValueError("Frozen folds do not match retained SPARC names")
    fold_assignment = folds.loc[list(packed.galaxy_names), "fold"].to_numpy(
        dtype=int
    )
    geometry = pd.read_csv(geometry_path).set_index("galaxy")
    score_columns = [
        inputs["primary_void_score"],
        *inputs["robustness_void_scores"],
    ]
    scores = {
        column: geometry.loc[list(packed.galaxy_names), column].to_numpy(dtype=float)
        for column in score_columns
    }
    primary = inputs["primary_void_score"]
    ungrouped_64, ungrouped_128 = inputs["robustness_void_scores"]
    rng = np.random.default_rng(int(validation["seed"]))
    shuffled_primary = rng.permutation(scores[primary])

    variants = [
        Variant("newtonian_baryons", "newtonian"),
        Variant("fixed_rar", "rar"),
        Variant("legacy_size_only", "legacy"),
        Variant("local_acceleration_internal", "local"),
        Variant("local_acceleration_void", "local", score_column=primary),
        Variant(
            "local_acceleration_void_shuffled",
            "local",
            score_column=primary,
            shuffled=True,
        ),
        Variant(
            "local_acceleration_void_ungrouped_64",
            "local",
            score_column=ungrouped_64,
        ),
        Variant(
            "local_acceleration_void_ungrouped_128",
            "local",
            score_column=ungrouped_128,
        ),
        Variant(
            "catalog_mass_surface_internal",
            "catalog",
            transition_driver="surface",
        ),
        Variant(
            "catalog_mass_surface_void",
            "catalog",
            transition_driver="surface",
            score_column=primary,
        ),
        Variant(
            "catalog_mass_surface_void_shuffled",
            "catalog",
            transition_driver="surface",
            score_column=primary,
            shuffled=True,
        ),
        Variant(
            "catalog_mass_concentration_internal",
            "catalog",
            transition_driver="concentration",
        ),
        Variant(
            "catalog_mass_concentration_void",
            "catalog",
            transition_driver="concentration",
            score_column=primary,
        ),
    ]
    predictions, fold_summaries = run_cross_validation(
        variants,
        packed,
        baryonic_v2,
        predictors,
        scores,
        fold_assignment,
        sigma,
        optimizer_starts=int(validation["optimizer_starts"]),
        seed=int(validation["seed"]),
        shuffled_primary=shuffled_primary,
    )
    galaxy_metrics = per_galaxy_metrics(predictions)
    metrics = {
        variant: aggregate_metrics(frame)
        for variant, frame in galaxy_metrics.groupby("variant")
    }

    pairs = [
        ("local_acceleration_internal", "legacy_size_only"),
        ("catalog_mass_surface_internal", "legacy_size_only"),
        ("catalog_mass_concentration_internal", "legacy_size_only"),
        ("local_acceleration_void", "local_acceleration_internal"),
        ("catalog_mass_surface_void", "catalog_mass_surface_internal"),
        (
            "catalog_mass_concentration_void",
            "catalog_mass_concentration_internal",
        ),
        ("local_acceleration_void", "fixed_rar"),
    ]
    comparisons = {}
    for pair_index, (candidate, baseline) in enumerate(pairs):
        comparisons[f"{candidate}_vs_{baseline}"] = bootstrap_comparison(
            galaxy_metrics.loc[galaxy_metrics["variant"] == baseline],
            galaxy_metrics.loc[galaxy_metrics["variant"] == candidate],
            draws=int(validation["paired_bootstrap_draws"]),
            seed=int(validation["seed"]) + 10_000 + pair_index,
        )

    permutation_rows = permutation_distribution(
        packed,
        baryonic_v2,
        predictors,
        scores[primary],
        fold_assignment,
        sigma,
        permutations=int(validation["void_score_permutations"]),
        starts=int(validation["permutation_optimizer_starts"]),
        seed=int(validation["seed"]) + 50_000,
    )
    permutation_frame = pd.DataFrame(permutation_rows)

    scaling_config = protocol["success_gates"]["galaxy_scaling"]
    void_config = protocol["success_gates"]["incremental_void_information"]
    internal_pairs = {
        "local_acceleration": "local_acceleration_internal",
        "mass_surface": "catalog_mass_surface_internal",
        "mass_concentration": "catalog_mass_concentration_internal",
    }
    internal_gates = {
        label: pair_gate(
            metrics,
            comparisons,
            candidate,
            "legacy_size_only",
            rmse_fraction_max=float(
                scaling_config["rmse_fraction_vs_legacy_size_only_max"]
            ),
            bootstrap_min=float(
                scaling_config["paired_bootstrap_probability_improves_min"]
            ),
        )
        for label, candidate in internal_pairs.items()
    }
    void_pairs = {
        "local_acceleration": (
            "local_acceleration_void",
            "local_acceleration_internal",
        ),
        "mass_surface": (
            "catalog_mass_surface_void",
            "catalog_mass_surface_internal",
        ),
        "mass_concentration": (
            "catalog_mass_concentration_void",
            "catalog_mass_concentration_internal",
        ),
    }
    incremental_pair_gates = {
        label: pair_gate(
            metrics,
            comparisons,
            candidate,
            baseline,
            rmse_fraction_max=float(
                void_config["rmse_fraction_vs_matching_internal_control_max"]
            ),
            bootstrap_min=float(
                void_config["paired_bootstrap_probability_improves_min"]
            ),
        )
        for label, (candidate, baseline) in void_pairs.items()
    }

    def exponents(variant_name: str) -> list[float]:
        return [
            float(row["parameters"]["void_exponent_m"])
            for row in fold_summaries
            if row["variant"] == variant_name
        ]

    primary_exponents = exponents("local_acceleration_void")
    reconstruction_exponents = {
        "ungrouped_64": exponents("local_acceleration_void_ungrouped_64"),
        "ungrouped_128": exponents("local_acceleration_void_ungrouped_128"),
    }
    real_chi2 = float(metrics["local_acceleration_void"]["chi2_per_point"])
    fraction_permutations_worse = float(
        np.mean(permutation_frame["chi2_per_point"] > real_chi2)
    )
    exponent_gate = all(1e-6 < value < 4.0 - 1e-6 for value in primary_exponents)
    reconstruction_sign_gate = all(
        value > 0.0
        for values in reconstruction_exponents.values()
        for value in values
    )
    primary_void_gates = {
        **incremental_pair_gates["local_acceleration"],
        "void_exponent_positive_and_not_at_bounds_all_folds": exponent_gate,
        "fraction_permutations_worse_than_real": fraction_permutations_worse,
        "permutation_gate": fraction_permutations_worse
        >= float(void_config["real_score_permutation_percentile_min"]),
        "reconstruction_sign_stable": reconstruction_sign_gate,
    }
    primary_void_pass = bool(
        primary_void_gates["pair_pass"]
        and primary_void_gates[
            "void_exponent_positive_and_not_at_bounds_all_folds"
        ]
        and primary_void_gates["permutation_gate"]
        and primary_void_gates["reconstruction_sign_stable"]
    )
    any_internal_pass = any(
        bool(gate["pair_pass"]) for gate in internal_gates.values()
    )

    predictors_frame = pd.DataFrame(
        {
            "galaxy": packed.galaxy_names,
            "fold": fold_assignment,
            "mass_proxy_1e9_msun": predictors.mass_proxy_1e9_msun,
            "central_stellar_surface_density_msun_pc2": predictors.central_stellar_surface_density_msun_pc2,
            "concentration_rdisk_over_reff": predictors.concentration_rdisk_over_reff,
            "primary_void_score": scores[primary],
        }
    )
    fold_parameter_rows = []
    for row in fold_summaries:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"parameters", "predictor_training_medians"}
        }
        flat.update(row["parameters"])
        flat.update(
            {
                f"training_median_{key}": value
                for key, value in row["predictor_training_medians"].items()
            }
        )
        fold_parameter_rows.append(flat)
    fold_parameters = pd.DataFrame(fold_parameter_rows)

    artifact_paths = {
        "heldout_predictions.csv": args.output_dir / "heldout_predictions.csv",
        "heldout_galaxy_metrics.csv": args.output_dir
        / "heldout_galaxy_metrics.csv",
        "fold_parameters.csv": args.output_dir / "fold_parameters.csv",
        "galaxy_predictors.csv": args.output_dir / "galaxy_predictors.csv",
        "void_score_permutations.csv": args.output_dir
        / "void_score_permutations.csv",
        "heldout_rmse_comparison.png": args.output_dir
        / "heldout_rmse_comparison.png",
    }
    predictions.to_csv(artifact_paths["heldout_predictions.csv"], index=False)
    galaxy_metrics.to_csv(
        artifact_paths["heldout_galaxy_metrics.csv"], index=False
    )
    fold_parameters.to_csv(artifact_paths["fold_parameters.csv"], index=False)
    predictors_frame.to_csv(artifact_paths["galaxy_predictors.csv"], index=False)
    permutation_frame.to_csv(
        artifact_paths["void_score_permutations.csv"], index=False
    )
    plot_metrics(metrics, artifact_paths["heldout_rmse_comparison.png"])

    report = {
        "status": "completed preregistered galaxy-dependent void-screening test",
        "report_version": "void-cage-galaxy-scaling-test-0.1",
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": protocol_hash,
            "version": protocol["protocol_version"],
            "frozen_utc": protocol["frozen_utc"],
        },
        "input_hashes": {
            "SPARC_metadata": sha256(metadata_path),
            "SPARC_data_fingerprint": packed.data_fingerprint,
            "void_geometry": sha256(geometry_path),
            "fold_assignments": sha256(fold_path),
        },
        "design": {
            "galaxies": packed.n_galaxies,
            "points": packed.n_points,
            "folds": int(fold_assignment.max()) + 1,
            "holdout_unit": "whole galaxy",
            "optimizer_starts": int(validation["optimizer_starts"]),
            "bootstrap_draws": int(validation["paired_bootstrap_draws"]),
            "void_permutations": int(validation["void_score_permutations"]),
            "forbidden_velocity_predictors_used": False,
        },
        "variant_metrics": metrics,
        "outermost_point_metrics": outermost_metrics(predictions),
        "paired_comparisons": comparisons,
        "internal_galaxy_scaling_gates": internal_gates,
        "any_internal_galaxy_scaling_pass": any_internal_pass,
        "incremental_void_pair_gates": incremental_pair_gates,
        "primary_void_exponents": primary_exponents,
        "reconstruction_void_exponents": reconstruction_exponents,
        "permutation_test": {
            "real_primary_chi2_per_point": real_chi2,
            "fraction_permutations_worse_than_real": fraction_permutations_worse,
            "permutation_chi2_minimum": float(
                permutation_frame["chi2_per_point"].min()
            ),
            "permutation_chi2_median": float(
                permutation_frame["chi2_per_point"].median()
            ),
            "permutation_chi2_maximum": float(
                permutation_frame["chi2_per_point"].max()
            ),
        },
        "primary_incremental_void_gates": primary_void_gates,
        "primary_incremental_void_pass": primary_void_pass,
        "decision": {
            "galaxy_dependent_transition": "retain"
            if any_internal_pass
            else "reject_tested_internal_scalings",
            "incremental_void_information": "retain"
            if primary_void_pass
            else "not_supported",
            "negative_gravity_origin": "candidate"
            if primary_void_pass
            else "not_established",
        },
        "interpretation_guardrails": [
            "An internal-only improvement is not evidence for a void origin.",
            "Vflat, observed velocities, and rotation residuals were excluded from galaxy predictors.",
            "The void candidate must beat the matching internal-only family, not merely Newtonian baryons.",
            "The fixed RAR comparison is reported but was not a frozen pass condition.",
        ],
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in artifact_paths.items()
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
