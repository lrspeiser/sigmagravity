#!/usr/bin/env python3
"""Fit each SPARC model's nuisances on inner radii and score outer radii."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_sparc_density_transfer import attach_surface_brightness
from voidscreen.data import pack_dataset
from voidscreen.sparc_refit import effective_prediction, fit_galaxy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(value):
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_json(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def build_frame(protocol: dict, sparc: Path, morphology_path: Path) -> pd.DataFrame:
    sample = protocol["sample"]
    packed = pack_dataset(
        sparc,
        quality_max=int(sample["quality_max"]),
        minimum_inclination_deg=float(sample["minimum_inclination_deg"]),
        minimum_points=int(sample["minimum_points"]),
        train_fraction=float(sample["train_fraction"]),
        minimum_train_points=int(sample["minimum_train_points"]),
        minimum_holdout_points=int(sample["minimum_holdout_points"]),
    )
    index = packed.galaxy_index
    frame = pd.DataFrame(
        {
            "galaxy": np.asarray(packed.galaxy_names, dtype=object)[index],
            "galaxy_index": index,
            "split": np.where(packed.train_mask, "inner_train", "outer_holdout"),
            "radius_catalog_kpc": packed.radius_kpc,
            "velocity_observed_catalog_kms": packed.velocity_observed_kms,
            "velocity_error_catalog_kms": packed.velocity_error_kms,
            "distance_fractional_error": packed.distance_fractional_error[index],
            "inclination_catalog_deg": packed.inclination_deg[index],
            "inclination_error_deg": packed.inclination_error_deg[index],
            "disk_scale_kpc": packed.disk_scale_kpc[index],
            "quality": packed.quality[index],
        }
    )
    morphology = pd.read_csv(morphology_path)
    retained = set(packed.galaxy_names)
    morphology = morphology[morphology["galaxy"].isin(retained)].copy()
    selected = [
        "galaxy",
        "HI_mass_billion_solar",
        "HI_radius_kpc",
        "disk_luminosity_fit_solar",
        "bulge_luminosity_fit_solar",
        "bulge_scale_fit_kpc",
    ]
    frame = frame.merge(morphology[selected], on="galaxy", how="left", validate="many_to_one")
    frame = attach_surface_brightness(frame, sparc)
    expected = sample
    counts = frame["split"].value_counts()
    if (
        frame["galaxy"].nunique() != int(expected["expected_galaxies"])
        or int(counts.get("inner_train", 0)) != int(expected["expected_train_points"])
        or int(counts.get("outer_holdout", 0)) != int(expected["expected_outer_points"])
    ):
        raise ValueError("independent refit did not preserve the frozen SPARC sample")
    return frame


def optimizer_settings(protocol: dict) -> dict[str, float]:
    nuisance = protocol["nuisance_fit"]
    nfw = nuisance["NFW_weak_priors"]
    return {
        "disk_mass_to_light_prior": float(nuisance["disk_mass_to_light_prior"]),
        "bulge_mass_to_light_prior": float(nuisance["bulge_mass_to_light_prior"]),
        "log_mass_to_light_prior_sigma": float(nuisance["log_mass_to_light_prior_sigma"]),
        "velocity_error_floor_km_s": float(nuisance["velocity_error_floor_km_s"]),
        "rar_acceleration_m_s2": 1.2e-10,
        "mond_acceleration_m_s2": 1.2e-10,
        "coherence_gate_power": 2.0,
        "hubble_km_s_mpc": 70.0,
        "nfw_v200_prior_km_s": float(nfw["log_V200_center_km_s"]),
        "nfw_log_v200_sigma": float(nfw["log_V200_sigma"]),
        "nfw_concentration_prior": float(nfw["log_concentration_center"]),
        "nfw_log_concentration_sigma": float(nfw["log_concentration_sigma"]),
    }


def bounds_for(model: str, protocol: dict) -> list[tuple[float, float]]:
    raw = protocol["nuisance_fit"]["bounds"]
    bounds = [
        tuple(map(float, raw["disk_log_shift"])),
        tuple(map(float, raw["bulge_log_shift"])),
        tuple(map(float, raw["distance_z"])),
        tuple(map(float, raw["inclination_z"])),
    ]
    if model == "nfw":
        bounds.extend(
            [
                tuple(map(math.log, raw["NFW_V200_km_s"])),
                tuple(map(math.log, raw["NFW_concentration"])),
            ]
        )
    return bounds


def starts_for(model: str, protocol: dict, galaxy_index: int) -> list[np.ndarray]:
    nuisance = protocol["nuisance_fit"]
    count = int(nuisance["starts_per_galaxy"])
    seed = int(nuisance["seed"]) + 104729 * int(galaxy_index)
    rng = np.random.default_rng(seed)
    center = np.zeros(6 if model == "nfw" else 4)
    if model == "nfw":
        center[4:] = [math.log(100.0), math.log(10.0)]
    starts = [center.copy()]
    scale = float(nuisance["random_start_scale"])
    bounds = bounds_for(model, protocol)
    for _ in range(count - 1):
        trial = center.copy()
        trial[:4] += rng.normal(0.0, scale, size=4)
        if model == "nfw":
            trial[4] += rng.normal(0.0, scale)
            trial[5] += rng.normal(0.0, 0.6 * scale)
        trial = np.asarray(
            [np.clip(value, low, high) for value, (low, high) in zip(trial, bounds, strict=True)]
        )
        starts.append(trial)
    return starts


def fit_one_variant(
    frame: pd.DataFrame,
    *,
    output_model: str,
    model: str,
    scenario: str,
    protocol: dict,
    settings: dict,
    candidate_parameters: np.ndarray,
    density_geometry: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_blocks = []
    fit_rows = []
    bounds = bounds_for(model, protocol)
    tolerance = 1.0e-4
    for galaxy_index, block in frame.groupby("galaxy_index", sort=True):
        inner = block[block["split"] == "inner_train"]
        fitted = fit_galaxy(
            inner,
            model=model,
            settings=settings,
            starts=starts_for(model, protocol, int(galaxy_index)),
            bounds=bounds,
            candidate_parameters=candidate_parameters if model == "candidate" else None,
            density_geometry=density_geometry,
            max_iterations=int(protocol["nuisance_fit"]["max_iterations"]),
        )
        predicted = effective_prediction(
            block,
            fitted.theta,
            model=model,
            settings=settings,
            candidate_parameters=candidate_parameters if model == "candidate" else None,
            density_geometry=density_geometry,
        )
        points = block[
            [
                "galaxy",
                "galaxy_index",
                "split",
                "radius_catalog_kpc",
                "velocity_observed_catalog_kms",
                "velocity_error_catalog_kms",
            ]
        ].copy()
        for name, values in predicted.items():
            points[name] = values
        points.insert(0, "scenario", scenario)
        points.insert(0, "model", output_model)
        point_blocks.append(points)

        at_boundary = [
            abs(value - low) <= tolerance * max(1.0, abs(low))
            or abs(value - high) <= tolerance * max(1.0, abs(high))
            for value, (low, high) in zip(fitted.theta, bounds, strict=True)
        ]
        row = {
            "model": output_model,
            "scenario": scenario,
            "galaxy": str(block["galaxy"].iloc[0]),
            "galaxy_index": int(galaxy_index),
            "objective_inner": fitted.objective,
            "optimizer_success": fitted.success,
            "finite_fit": fitted.finite,
            "evaluations": fitted.evaluations,
            "any_parameter_at_boundary": any(at_boundary),
            "disk_log_shift": fitted.theta[0],
            "bulge_log_shift": fitted.theta[1],
            "distance_z": fitted.theta[2],
            "inclination_z": fitted.theta[3],
            "disk_mass_to_light": predicted["disk_mass_to_light"][0],
            "bulge_mass_to_light": predicted["bulge_mass_to_light"][0],
            "distance_scale": predicted["distance_scale"][0],
            "inclination_adjusted_deg": predicted["inclination_adjusted_deg"][0],
            "nfw_V200_km_s": math.exp(fitted.theta[4]) if model == "nfw" else math.nan,
            "nfw_concentration": math.exp(fitted.theta[5]) if model == "nfw" else math.nan,
            "message": fitted.message,
        }
        fit_rows.append(row)
    return pd.concat(point_blocks, ignore_index=True), pd.DataFrame(fit_rows)


def metrics(points: pd.DataFrame, split: str) -> dict:
    selected = points[points["split"] == split].copy()
    residual = (
        selected["velocity_predicted_km_s"]
        - selected["velocity_observed_adjusted_km_s"]
    ).to_numpy(dtype=float)
    catalog_residual = (
        selected["velocity_predicted_catalog_km_s"]
        - selected["velocity_observed_catalog_kms"]
    ).to_numpy(dtype=float)
    sigma = selected["velocity_error_total_km_s"].to_numpy(dtype=float)
    per_galaxy_mse = selected.assign(residual_squared=np.square(residual)).groupby(
        "galaxy"
    )["residual_squared"].mean()
    extra_vs_rar = (
        selected["velocity_predicted_km_s"]
        - selected["velocity_RAR_same_nuisance_km_s"]
    ).to_numpy(dtype=float)
    active = np.isfinite(selected["coherence"].to_numpy(dtype=float)) & (
        selected["coherence"].to_numpy(dtype=float) < 0.999
    )
    return {
        "points": len(selected),
        "galaxies": int(selected["galaxy"].nunique()),
        "chi2_per_point": float(np.mean(np.square(residual / sigma))),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "equal_galaxy_RMSE_km_s": float(np.sqrt(per_galaxy_mse.mean())),
        "catalog_space_RMSE_km_s": float(np.sqrt(np.mean(np.square(catalog_residual)))),
        "MAE_km_s": float(np.mean(np.abs(residual))),
        "mean_standardized_residual": float(np.mean(residual / sigma)),
        "median_formula_velocity_delta_vs_RAR_same_nuisance_km_s": float(
            np.median(extra_vs_rar)
        ),
        "p95_formula_velocity_delta_vs_RAR_same_nuisance_km_s": float(
            np.percentile(extra_vs_rar, 95.0)
        ),
        "maximum_formula_velocity_delta_vs_RAR_same_nuisance_km_s": float(
            np.max(extra_vs_rar)
        ),
        "coherence_gate_active_point_fraction": float(np.mean(active)),
    }


def paired_bootstrap(
    candidate: pd.DataFrame, rar: pd.DataFrame, *, draws: int, seed: int
) -> dict:
    def galaxy_mse(points: pd.DataFrame) -> pd.Series:
        selected = points[points["split"] == "outer_holdout"].copy()
        selected["residual_squared"] = np.square(
            selected["velocity_predicted_km_s"]
            - selected["velocity_observed_adjusted_km_s"]
        )
        return selected.groupby("galaxy")["residual_squared"].mean().sort_index()

    cand = galaxy_mse(candidate)
    base = galaxy_mse(rar)
    if not cand.index.equals(base.index):
        raise ValueError("candidate and RAR galaxy sets differ")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cand), size=(draws, len(cand)))
    cand_draw = np.sqrt(np.mean(cand.to_numpy()[indices], axis=1))
    base_draw = np.sqrt(np.mean(base.to_numpy()[indices], axis=1))
    delta = cand_draw - base_draw
    return {
        "draws": draws,
        "delta_candidate_minus_RAR_equal_galaxy_RMSE_km_s": float(
            math.sqrt(cand.mean()) - math.sqrt(base.mean())
        ),
        "interval_95_km_s": list(map(float, np.percentile(delta, [2.5, 97.5]))),
        "probability_candidate_better_than_RAR": float(np.mean(delta < 0.0)),
    }


def fold_assignment(frame: pd.DataFrame, folds: int, seed: int) -> dict[str, int]:
    names = np.asarray(sorted(frame["galaxy"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    return {str(name): int(index % folds) for index, name in enumerate(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "sparc_independent_nuisance_refit_protocol.json",
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=ROOT / "results" / "rar_sharp_coherence_rg_sweep" / "report.json",
    )
    parser.add_argument(
        "--morphology",
        type=Path,
        default=ROOT / "data" / "derived" / "nbp0_sparc_morphology.csv",
    )
    parser.add_argument("--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sparc_independent_nuisance_refit",
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    parameter_record = candidate_report["full_sample_descriptive_fits"][
        "RAR_sharp_coherence_gated_RG"
    ]["parameters"]
    candidate_parameters = np.asarray(
        [
            parameter_record["epsilon_0"],
            parameter_record["log10_rho_c_g_cm3"],
            parameter_record["Q"],
        ]
    )
    frame = build_frame(protocol, args.sparc, args.morphology)
    settings = optimizer_settings(protocol)

    variants = [
        ("fixed_RAR", "rar", "invariant", None),
        ("simple_MOND", "simple_mond", "invariant", None),
        ("NFW", "nfw", "invariant", None),
    ]
    variants.extend(
        (
            "RAR_sharp_coherence_gated_RG",
            "candidate",
            scenario["name"],
            scenario,
        )
        for scenario in protocol["candidate_density_scenarios"]
    )

    all_points = []
    all_fits = []
    for output_model, model, scenario, geometry in variants:
        print(f"model={output_model} scenario={scenario}", flush=True)
        points, fits = fit_one_variant(
            frame,
            output_model=output_model,
            model=model,
            scenario=scenario,
            protocol=protocol,
            settings=settings,
            candidate_parameters=candidate_parameters,
            density_geometry=geometry,
        )
        all_points.append(points)
        all_fits.append(fits)
    points = pd.concat(all_points, ignore_index=True)
    fits = pd.concat(all_fits, ignore_index=True)

    scores = {}
    fit_diagnostics = {}
    for (model, scenario), block in points.groupby(["model", "scenario"], sort=False):
        key = f"{model}:{scenario}"
        scores[key] = {
            "inner_train": metrics(block, "inner_train"),
            "outer_holdout": metrics(block, "outer_holdout"),
        }
        fit_block = fits[(fits["model"] == model) & (fits["scenario"] == scenario)]
        fit_diagnostics[key] = {
            "finite_fit_fraction": float(fit_block["finite_fit"].mean()),
            "optimizer_success_fraction": float(fit_block["optimizer_success"].mean()),
            "nuisance_any_boundary_fraction": float(
                fit_block["any_parameter_at_boundary"].mean()
            ),
            "median_evaluations": float(fit_block["evaluations"].median()),
        }

    rar = points[(points["model"] == "fixed_RAR") & (points["scenario"] == "invariant")]
    primary = points[
        (points["model"] == "RAR_sharp_coherence_gated_RG")
        & (points["scenario"] == "primary")
    ]
    rar_outer = scores["fixed_RAR:invariant"]["outer_holdout"]
    primary_outer = scores[
        "RAR_sharp_coherence_gated_RG:primary"
    ]["outer_holdout"]
    bootstrap = paired_bootstrap(
        primary,
        rar,
        draws=int(protocol["uncertainty"]["paired_galaxy_bootstrap_draws"]),
        seed=int(protocol["nuisance_fit"]["seed"]) + 1,
    )
    assignments = fold_assignment(
        frame,
        int(protocol["uncertainty"]["galaxy_folds"]),
        int(protocol["nuisance_fit"]["seed"]),
    )
    fold_rows = []
    for fold in range(int(protocol["uncertainty"]["galaxy_folds"])):
        names = [name for name, value in assignments.items() if value == fold]
        candidate_fold = primary[primary["galaxy"].isin(names)]
        rar_fold = rar[rar["galaxy"].isin(names)]
        candidate_metric = metrics(candidate_fold, "outer_holdout")
        rar_metric = metrics(rar_fold, "outer_holdout")
        fold_rows.append(
            {
                "fold": fold,
                "galaxies": len(names),
                "candidate_RMSE_km_s": candidate_metric["RMSE_km_s"],
                "RAR_RMSE_km_s": rar_metric["RMSE_km_s"],
                "candidate_to_RAR_RMSE_ratio": (
                    candidate_metric["RMSE_km_s"] / rar_metric["RMSE_km_s"]
                ),
            }
        )

    candidate_sensitivity = {
        scenario["name"]: scores[
            f"RAR_sharp_coherence_gated_RG:{scenario['name']}"
        ]["outer_holdout"]
        for scenario in protocol["candidate_density_scenarios"]
    }
    gates = protocol["advance_gates"]
    primary_fit = fit_diagnostics["RAR_sharp_coherence_gated_RG:primary"]
    gate_audit = {
        "primary_outer_RMSE": (
            primary_outer["RMSE_km_s"]
            <= float(gates["primary_outer_RMSE_relative_to_RAR_max"])
            * rar_outer["RMSE_km_s"]
        ),
        "primary_outer_chi2": (
            primary_outer["chi2_per_point"]
            <= float(gates["primary_outer_chi2_per_point_relative_to_RAR_max"])
            * rar_outer["chi2_per_point"]
        ),
        "bootstrap_upper": (
            bootstrap["interval_95_km_s"][1]
            <= float(gates["bootstrap_delta_equal_galaxy_RMSE_95pct_upper_km_s_max"])
        ),
        "worst_fold": (
            max(row["candidate_to_RAR_RMSE_ratio"] for row in fold_rows)
            <= float(gates["worst_fold_outer_RMSE_relative_to_RAR_max"])
        ),
        "density_sensitivity": (
            max(value["RMSE_km_s"] for value in candidate_sensitivity.values())
            <= float(gates["worst_density_sensitivity_outer_RMSE_relative_to_RAR_max"])
            * rar_outer["RMSE_km_s"]
        ),
        "finite_fits": (
            primary_fit["finite_fit_fraction"]
            >= float(gates["finite_fit_fraction_min"])
        ),
        "optimizer_success": (
            primary_fit["optimizer_success_fraction"]
            >= float(gates["optimizer_success_fraction_min"])
        ),
        "nuisance_boundaries": (
            primary_fit["nuisance_any_boundary_fraction"]
            <= float(gates["nuisance_any_boundary_fraction_max"])
        ),
    }
    gate_audit["passes_all"] = all(gate_audit.values())

    report = {
        "status": "completed independent inner-nuisance outer-radius prediction",
        "inputs": {
            "protocol_sha256": sha256(args.protocol),
            "candidate_report_sha256": sha256(args.candidate_report),
            "morphology_sha256": sha256(args.morphology),
        },
        "candidate_parameters_fixed_before_SPARC_refit": parameter_record,
        "sample": {
            "galaxies": int(frame["galaxy"].nunique()),
            "inner_train_points": int((frame["split"] == "inner_train").sum()),
            "outer_holdout_points": int((frame["split"] == "outer_holdout").sum()),
        },
        "scores": scores,
        "fit_diagnostics": fit_diagnostics,
        "candidate_sensitivity": candidate_sensitivity,
        "paired_galaxy_bootstrap": bootstrap,
        "fold_stability": fold_rows,
        "gate_audit": gate_audit,
        "interpretation": {
            "galaxy_result": (
                "competitive with fixed RAR under the preregistered gates"
                if gate_audit["passes_all"]
                else "failed one or more preregistered galaxy gates"
            ),
            "distinct_from_RAR": (
                "not established; candidate inherits the RAR term and is tested for harmful leakage"
            ),
            "lensing_status": (
                "not tested here; local CLASH accelerations are NFW-deprojected and the raw "
                "same-system lensing+baryon likelihood is not yet complete"
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.output / "point_predictions.csv", index=False)
    fits.to_csv(args.output / "galaxy_fits.csv", index=False)
    pd.DataFrame(
        [{"galaxy": name, "fold": fold} for name, fold in assignments.items()]
    ).sort_values("galaxy").to_csv(args.output / "fold_assignments.csv", index=False)
    (args.output / "report.json").write_text(
        json.dumps(strict_json(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(strict_json({"gate_audit": gate_audit, "outer_scores": {
        key: value["outer_holdout"] for key, value in scores.items()
    }}), indent=2))


if __name__ == "__main__":
    main()
