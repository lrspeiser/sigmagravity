from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from run_void_cage_galaxy_scaling_test import (
    aggregate_metrics,
    bootstrap_comparison,
    per_galaxy_metrics,
    plot_metrics,
    sha256,
)
from voidscreen.data import PackedDataset, pack_dataset
from voidscreen.galaxy_scaling import (
    GalaxyPredictors,
    catalog_scaled_screened_velocity,
    load_sparc_structural_predictors,
    normalize_positive_by_training_median,
)
from voidscreen.void_cage import baryonic_velocity_squared


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Variant:
    name: str
    transition_driver: str | None


def normalized_for_fold(
    predictors: GalaxyPredictors, training: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    raw = {
        "mass": predictors.mass_proxy_1e9_msun,
        "surface": predictors.central_stellar_surface_density_msun_pc2,
        "concentration": predictors.concentration_rdisk_over_reff,
    }
    values = {}
    medians = {}
    for name, source in raw.items():
        values[name], medians[name] = normalize_positive_by_training_median(
            source, training
        )
    return values, medians


def predict(
    variant: Variant,
    parameters: np.ndarray,
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    normalized: dict[str, np.ndarray],
) -> np.ndarray:
    if variant.transition_driver is None:
        driver = np.ones(packed.n_galaxies)
        transition_exponent = 0.0
    else:
        driver = normalized[variant.transition_driver]
        transition_exponent = float(parameters[3])
    return catalog_scaled_screened_velocity(
        packed,
        baryonic_v2,
        mass_by_galaxy=normalized["mass"],
        transition_driver_by_galaxy=driver,
        log10_velocity_scale_km_s=float(parameters[0]),
        log10_transition_scale_lengths=float(parameters[1]),
        mass_amplitude_exponent=float(parameters[2]),
        transition_exponent=transition_exponent,
    )


def fit(
    variant: Variant,
    packed: PackedDataset,
    baryonic_v2: np.ndarray,
    normalized: dict[str, np.ndarray],
    training_points: np.ndarray,
    sigma: np.ndarray,
    *,
    starts: int,
    seed: int,
) -> tuple[np.ndarray, float, bool]:
    bounds = [
        (0.0, math.log10(500.0)),
        (-1.0, math.log10(20.0)),
        (-2.0, 2.0),
    ]
    if variant.transition_driver is not None:
        bounds.append((-2.0, 2.0))

    def objective(parameters: np.ndarray) -> float:
        prediction = predict(variant, parameters, packed, baryonic_v2, normalized)
        residual = (
            prediction[training_points]
            - packed.velocity_observed_kms[training_points]
        ) / sigma[training_points]
        return float(np.sum(residual**2))

    midpoint = np.asarray([(low + high) / 2.0 for low, high in bounds])
    starts_list = [midpoint]
    if variant.transition_driver is not None:
        nested = midpoint.copy()
        nested[3] = 0.0
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "void_cage_transition_isolation_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "void_cage_transition_isolation",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    parent_path = ROOT / protocol["inputs"]["parent_protocol"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    sample = parent["sample"]
    validation = protocol["validation"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_path = ROOT / protocol["inputs"]["fold_assignments"]
    if sha256(fold_path) != protocol["inputs"]["expected_fold_assignment_sha256"]:
        raise ValueError("Frozen folds do not match the isolation protocol")
    packed = pack_dataset(
        ROOT / protocol["inputs"]["SPARC_directory"],
        quality_max=int(sample["quality_max"]),
        minimum_inclination_deg=float(sample["minimum_inclination_deg"]),
        minimum_points=int(sample["minimum_points"]),
    )
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
        ROOT / protocol["inputs"]["SPARC_metadata"], packed.galaxy_names
    )
    fold_frame = pd.read_csv(fold_path).set_index("galaxy")
    folds = fold_frame.loc[list(packed.galaxy_names), "fold"].to_numpy(dtype=int)

    variants = [
        Variant("mass_amplitude_only", None),
        Variant("mass_transition", "mass"),
        Variant("surface_transition", "surface"),
        Variant("concentration_transition", "concentration"),
    ]
    prediction_frames = []
    parameter_rows = []
    for heldout_fold in range(int(folds.max()) + 1):
        heldout_galaxies = folds == heldout_fold
        training_galaxies = ~heldout_galaxies
        training_points = training_galaxies[packed.galaxy_index]
        heldout_points = heldout_galaxies[packed.galaxy_index]
        normalized, medians = normalized_for_fold(predictors, training_galaxies)
        for variant_index, variant in enumerate(variants):
            parameters, train_chi2, converged = fit(
                variant,
                packed,
                baryonic_v2,
                normalized,
                training_points,
                sigma,
                starts=int(validation["optimizer_starts"]),
                seed=int(validation["seed"]) + 1000 * heldout_fold + variant_index,
            )
            prediction = predict(
                variant, parameters, packed, baryonic_v2, normalized
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
                        "velocity_predicted_kms": prediction[selected],
                    }
                )
            )
            row = {
                "fold": heldout_fold,
                "variant": variant.name,
                "transition_driver": variant.transition_driver,
                "optimizer_converged": converged,
                "train_chi2": train_chi2,
                "velocity_scale_km_s": float(10.0 ** parameters[0]),
                "transition_scale_lengths": float(10.0 ** parameters[1]),
                "mass_amplitude_exponent_eta": float(parameters[2]),
                "transition_exponent": (
                    float(parameters[3])
                    if variant.transition_driver is not None
                    else 0.0
                ),
            }
            row.update({f"training_median_{k}": v for k, v in medians.items()})
            parameter_rows.append(row)
            print(f"fold={heldout_fold} variant={variant.name}", flush=True)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    galaxy_metrics = per_galaxy_metrics(predictions)
    metrics = {
        variant: aggregate_metrics(frame)
        for variant, frame in galaxy_metrics.groupby("variant")
    }
    comparisons = {}
    for index, candidate in enumerate(
        ["mass_transition", "surface_transition", "concentration_transition"]
    ):
        comparisons[f"{candidate}_vs_mass_amplitude_only"] = bootstrap_comparison(
            galaxy_metrics.loc[
                galaxy_metrics["variant"] == "mass_amplitude_only"
            ],
            galaxy_metrics.loc[galaxy_metrics["variant"] == candidate],
            draws=int(validation["paired_bootstrap_draws"]),
            seed=int(validation["seed"]) + index,
        )

    parameter_frame = pd.DataFrame(parameter_rows)
    gates = {}
    for candidate in [
        "mass_transition",
        "surface_transition",
        "concentration_transition",
    ]:
        ratio = float(metrics[candidate]["rmse_kms"]) / float(
            metrics["mass_amplitude_only"]["rmse_kms"]
        )
        comparison = comparisons[f"{candidate}_vs_mass_amplitude_only"]
        exponents = parameter_frame.loc[
            parameter_frame["variant"] == candidate, "transition_exponent"
        ].to_numpy(dtype=float)
        same_nonzero_sign = bool(
            np.all(exponents > 1e-6) or np.all(exponents < -1e-6)
        )
        away_from_bounds = bool(np.all(np.abs(exponents) < 2.0 - 1e-6))
        probability = float(
            comparison["bootstrap_probability_candidate_improves_chi2"]
        )
        gate = {
            "rmse_ratio": ratio,
            "rmse_gate": ratio
            <= float(
                protocol["success_gates"][
                    "rmse_fraction_vs_mass_amplitude_only_max"
                ]
            ),
            "bootstrap_probability": probability,
            "bootstrap_gate": probability
            >= float(
                protocol["success_gates"][
                    "paired_bootstrap_probability_improves_min"
                ]
            ),
            "transition_exponents": exponents.tolist(),
            "same_nonzero_sign": same_nonzero_sign,
            "away_from_bounds": away_from_bounds,
        }
        gate["pass"] = bool(
            gate["rmse_gate"]
            and gate["bootstrap_gate"]
            and same_nonzero_sign
            and away_from_bounds
        )
        gates[candidate] = gate

    artifact_paths = {
        "heldout_predictions.csv": args.output_dir / "heldout_predictions.csv",
        "heldout_galaxy_metrics.csv": args.output_dir
        / "heldout_galaxy_metrics.csv",
        "fold_parameters.csv": args.output_dir / "fold_parameters.csv",
        "heldout_rmse_comparison.png": args.output_dir
        / "heldout_rmse_comparison.png",
    }
    predictions.to_csv(artifact_paths["heldout_predictions.csv"], index=False)
    galaxy_metrics.to_csv(
        artifact_paths["heldout_galaxy_metrics.csv"], index=False
    )
    parameter_frame.to_csv(artifact_paths["fold_parameters.csv"], index=False)
    plot_metrics(metrics, artifact_paths["heldout_rmse_comparison.png"])

    report = {
        "status": "completed transition-radius isolation check",
        "report_version": "void-cage-transition-isolation-0.1",
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": sha256(args.protocol),
            "version": protocol["protocol_version"],
            "frozen_utc": protocol["frozen_utc"],
        },
        "parent_protocol_sha256": sha256(parent_path),
        "design": {
            "galaxies": packed.n_galaxies,
            "points": packed.n_points,
            "folds": int(folds.max()) + 1,
            "holdout_unit": "whole galaxy",
        },
        "variant_metrics": metrics,
        "paired_comparisons": comparisons,
        "transition_gates": gates,
        "any_transition_driver_pass": any(bool(gate["pass"]) for gate in gates.values()),
        "decision": "retain_galaxy_dependent_transition"
        if any(bool(gate["pass"]) for gate in gates.values())
        else "mass_amplitude_explains_scaling_gain_without_supported_transition_shift",
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in artifact_paths.items()
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
