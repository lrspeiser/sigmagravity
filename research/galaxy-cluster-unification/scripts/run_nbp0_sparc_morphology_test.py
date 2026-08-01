from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.data import KPC_M
from voidscreen.sparc_morphology import parse_sparc_profile
from voidscreen.unified import A0_M_S2, rar_acceleration


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_standardized_linear(
    train: pd.DataFrame, test: pd.DataFrame, predictors: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    train_values = train[predictors].to_numpy(dtype=float)
    test_values = test[predictors].to_numpy(dtype=float)
    mean = np.mean(train_values, axis=0)
    scale = np.std(train_values, axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    train_design = np.column_stack(
        [np.ones(len(train)), (train_values - mean) / scale]
    )
    test_design = np.column_stack([np.ones(len(test)), (test_values - mean) / scale])
    coefficients = np.linalg.lstsq(
        train_design, train["target_log10_gobs_over_fixed_RAR"], rcond=None
    )[0]
    return test_design @ coefficients, coefficients


def cross_validated_predictions(frame: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    predictions = np.full(len(frame), np.nan)
    for fold in sorted(frame["fold"].unique()):
        train = frame.loc[frame["fold"] != fold]
        test = frame.loc[frame["fold"] == fold]
        predicted, _ = fit_standardized_linear(train, test, predictors)
        predictions[test.index.to_numpy()] = predicted
    if np.any(~np.isfinite(predictions)):
        raise RuntimeError("cross-validation did not predict every galaxy")
    return predictions


def rmse(observed: np.ndarray | pd.Series, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(observed, dtype=float) - predicted))))


def full_standardized_coefficients(
    frame: pd.DataFrame, predictors: list[str]
) -> dict[str, float]:
    _, coefficients = fit_standardized_linear(frame, frame, predictors)
    return {
        name: float(value)
        for name, value in zip(["intercept", *predictors], coefficients, strict=True)
    }


def build_targets(
    catalog: pd.DataFrame,
    data_directory: Path,
    *,
    disk_mass_to_light: float,
    bulge_mass_to_light: float,
    helium_factor: float,
    minimum_outer_radius: float,
    minimum_outer_points: int,
) -> pd.DataFrame:
    rows = []
    for record in catalog.loc[catalog["morphology_input_pass"]].itertuples(index=False):
        profile = parse_sparc_profile(
            data_directory / "rotmod" / f"{record.galaxy}_rotmod.dat"
        )
        radius_ratio = profile.radius_kpc / record.disk_scale_kpc
        selected = radius_ratio >= minimum_outer_radius
        gas_v2 = np.sign(profile.gas_velocity_km_s) * np.square(
            profile.gas_velocity_km_s
        )
        baryonic_v2 = (
            gas_v2
            + disk_mass_to_light * np.square(profile.disk_velocity_unit_ml_km_s)
            + bulge_mass_to_light * np.square(profile.bulge_velocity_unit_ml_km_s)
        )
        valid = (
            selected
            & np.isfinite(baryonic_v2)
            & (baryonic_v2 > 0.0)
            & np.isfinite(profile.observed_velocity_km_s)
            & (profile.observed_velocity_km_s > 0.0)
        )
        if int(valid.sum()) < minimum_outer_points:
            continue
        radius_m = profile.radius_kpc[valid] * KPC_M
        gbar = baryonic_v2[valid] * 1.0e6 / radius_m
        gobs = np.square(profile.observed_velocity_km_s[valid]) * 1.0e6 / radius_m
        fixed_rar = rar_acceleration(gbar, A0_M_S2)

        disk_mass = disk_mass_to_light * record.disk_luminosity_fit_solar
        bulge_mass = bulge_mass_to_light * record.bulge_luminosity_fit_solar
        gas_mass = helium_factor * record.HI_mass_billion_solar * 1.0e9
        baryonic_mass = disk_mass + bulge_mass + gas_mass
        baryonic_bt = bulge_mass / baryonic_mass
        rows.append(
            {
                "galaxy": record.galaxy,
                "fold": int(record.fold),
                "disk_mass_to_light": disk_mass_to_light,
                "bulge_mass_to_light": bulge_mass_to_light,
                "outer_points": int(valid.sum()),
                "target_log10_gobs_over_fixed_RAR": float(
                    np.median(np.log10(gobs / fixed_rar))
                ),
                "log10_baryonic_mass": math.log10(baryonic_mass),
                "log10_disk_scale": math.log10(record.disk_scale_kpc),
                "log10_effective_surface_brightness": math.log10(
                    record.effective_surface_brightness
                ),
                "gas_fraction": gas_mass / baryonic_mass,
                "median_log10_gbar": float(np.median(np.log10(gbar))),
                "median_radius_over_Rdisk": float(np.median(radius_ratio[valid])),
                "hubble_type": float(record.hubble_type),
                "baryonic_bulge_fraction": baryonic_bt,
                "bulge_scale_over_disk_scale": float(
                    record.bulge_scale_over_disk_scale
                    if math.isfinite(record.bulge_scale_over_disk_scale)
                    else 0.0
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("galaxy", kind="stable").reset_index(drop=True)
    frame.index = np.arange(len(frame))
    return frame


def matched_disk_bulge_effect(
    frame: pd.DataFrame, matching_predictors: list[str]
) -> dict[str, object]:
    bulges = frame.loc[frame["baryonic_bulge_fraction"] > 0.0]
    disks = frame.loc[frame["baryonic_bulge_fraction"] == 0.0]
    all_values = frame[matching_predictors].to_numpy(dtype=float)
    mean = np.mean(all_values, axis=0)
    scale = np.std(all_values, axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    disk_values = (disks[matching_predictors].to_numpy(dtype=float) - mean) / scale
    rows = []
    for record in bulges.itertuples():
        bulge_values = (
            np.asarray([getattr(record, key) for key in matching_predictors]) - mean
        ) / scale
        distance = np.sqrt(np.sum(np.square(disk_values - bulge_values), axis=1))
        disk_position = int(np.argmin(distance))
        disk = disks.iloc[disk_position]
        difference = (
            record.target_log10_gobs_over_fixed_RAR
            - disk["target_log10_gobs_over_fixed_RAR"]
        )
        rows.append(
            {
                "bulge_galaxy": record.galaxy,
                "disk_control": disk["galaxy"],
                "standardized_distance": float(distance[disk_position]),
                "bulge_minus_disk_target": float(difference),
            }
        )
    differences = np.asarray([row["bulge_minus_disk_target"] for row in rows])
    return {
        "pairs": rows,
        "pair_count": int(len(rows)),
        "mean_bulge_minus_disk_dex": float(np.mean(differences)),
        "median_bulge_minus_disk_dex": float(np.median(differences)),
        "fraction_in_predicted_negative_direction": float(np.mean(differences < 0.0)),
    }


def evaluate_assumption(
    frame: pd.DataFrame,
    *,
    baseline_predictors: list[str],
    morphology_predictors: list[str],
    permutations: int,
    seed: int,
) -> dict[str, object]:
    models = {
        "mass_only": ["log10_baryonic_mass"],
        "structure": baseline_predictors,
        "structure_plus_hubble_type": [*baseline_predictors, "hubble_type"],
        "structure_plus_morphology": [*baseline_predictors, *morphology_predictors],
        "structure_hubble_and_morphology": [
            *baseline_predictors,
            "hubble_type",
            *morphology_predictors,
        ],
    }
    target = frame["target_log10_gobs_over_fixed_RAR"].to_numpy(dtype=float)
    predictions = {
        name: cross_validated_predictions(frame, predictors)
        for name, predictors in models.items()
    }
    heldout_rmse = {
        name: rmse(target, values) for name, values in predictions.items()
    }
    baseline_rmse = heldout_rmse["structure"]
    morphology_rmse = heldout_rmse["structure_plus_morphology"]
    improvement = (baseline_rmse - morphology_rmse) / baseline_rmse

    rng = np.random.default_rng(seed)
    permuted_improvements = []
    morphology_values = frame[morphology_predictors].to_numpy(dtype=float)
    full_predictors = models["structure_plus_morphology"]
    for _ in range(permutations):
        permuted = frame.copy()
        permuted[morphology_predictors] = morphology_values[rng.permutation(len(frame))]
        permuted_prediction = cross_validated_predictions(permuted, full_predictors)
        permuted_improvements.append(
            (baseline_rmse - rmse(target, permuted_prediction)) / baseline_rmse
        )
    permuted_array = np.asarray(permuted_improvements)
    permutation_p = float(
        (1.0 + np.sum(permuted_array >= improvement)) / (permutations + 1.0)
    )
    coefficients = full_standardized_coefficients(frame, full_predictors)
    fold_coefficients = []
    for held_out_fold in sorted(frame["fold"].unique()):
        training = frame.loc[frame["fold"] != held_out_fold]
        fold_coefficients.append(
            {
                "held_out_fold": int(held_out_fold),
                **full_standardized_coefficients(training, full_predictors),
            }
        )
    matching_predictors = [
        "log10_baryonic_mass",
        "log10_disk_scale",
        "log10_effective_surface_brightness",
        "gas_fraction",
    ]
    return {
        "galaxies": int(len(frame)),
        "galaxies_with_bulge": int((frame["baryonic_bulge_fraction"] > 0.0).sum()),
        "heldout_RMSE_dex": heldout_rmse,
        "relative_structure_to_morphology_RMSE_improvement": float(improvement),
        "permutation_p_one_sided": permutation_p,
        "permuted_improvement_quantiles": {
            label: float(value)
            for label, value in zip(
                ["minimum", "p05", "median", "p95", "maximum"],
                np.quantile(permuted_array, [0.0, 0.05, 0.5, 0.95, 1.0]),
                strict=True,
            )
        },
        "standardized_full_sample_coefficients": coefficients,
        "fold_training_coefficients": fold_coefficients,
        "matched_disk_bulge": matched_disk_bulge_effect(frame, matching_predictors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "nbp0_morphology_protocol.json",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "data" / "derived" / "nbp0_sparc_morphology.csv",
    )
    parser.add_argument(
        "--synthetic-report",
        type=Path,
        default=ROOT / "results" / "nbp0_morphology_sweep" / "report.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "results" / "nbp0_sparc_morphology_test",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    synthetic_report = json.loads(args.synthetic_report.read_text(encoding="utf-8"))
    if synthetic_report["protocol_sha256"] != sha256(args.protocol):
        raise RuntimeError("synthetic sweep does not match the frozen morphology protocol")
    catalog = pd.read_csv(args.catalog)
    empirical = protocol["empirical_morphology_test"]
    sparc = protocol["SPARC_inputs"]
    baseline_predictors = list(empirical["baseline_predictors"])
    morphology_predictors = list(empirical["morphology_predictors"])
    sensitivities = empirical["mass_to_light_sensitivity"]
    all_targets = []
    results = []
    for disk_ml in sensitivities["disk_values"]:
        for bulge_ml in sensitivities["bulge_values"]:
            frame = build_targets(
                catalog,
                ROOT / sparc["directory"],
                disk_mass_to_light=float(disk_ml),
                bulge_mass_to_light=float(bulge_ml),
                helium_factor=float(sparc["helium_factor"]),
                minimum_outer_radius=float(empirical["outer_radius_min_over_Rdisk"]),
                minimum_outer_points=int(empirical["minimum_outer_points_per_galaxy"]),
            )
            all_targets.append(frame)
            evaluated = evaluate_assumption(
                frame,
                baseline_predictors=baseline_predictors,
                morphology_predictors=morphology_predictors,
                permutations=int(empirical["permutations"]),
                seed=int(empirical["fold_seed"])
                + int(round(100 * disk_ml))
                + int(round(1000 * bulge_ml)),
            )
            evaluated["disk_mass_to_light"] = float(disk_ml)
            evaluated["bulge_mass_to_light"] = float(bulge_ml)
            results.append(evaluated)
            print(
                f"completed M/L disk={disk_ml:.1f}, bulge={bulge_ml:.1f}", flush=True
            )

    primary_pair = tuple(float(value) for value in sensitivities["primary_pair"])
    primary = next(
        result
        for result in results
        if (result["disk_mass_to_light"], result["bulge_mass_to_light"])
        == primary_pair
    )
    gates = empirical["advance_gates"]
    primary_coefficient = primary["standardized_full_sample_coefficients"][
        "baryonic_bulge_fraction"
    ]
    empirical_gate_results = {
        "heldout_RMSE_improvement": bool(
            primary["relative_structure_to_morphology_RMSE_improvement"]
            >= gates["relative_heldout_RMSE_improvement_min"]
        ),
        "permutation_p": bool(
            primary["permutation_p_one_sided"] <= gates["permutation_p_max"]
        ),
        "negative_partial_bulge_coefficient": bool(primary_coefficient < 0.0),
        "synthetic_predicted_sign": bool(
            synthetic_report["synthetic_predicted_sign"]["passes_gate"]
        ),
    }
    report = {
        "report_version": "NBP0-M1-SPARC-morphology-test-0.1",
        "status": "completed frozen SPARC morphology test",
        "protocol": str(args.protocol.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(args.protocol),
        "synthetic_report": str(args.synthetic_report.relative_to(ROOT)).replace("\\", "/"),
        "primary_mass_to_light_pair": list(primary_pair),
        "primary_result": primary,
        "mass_to_light_sensitivity_results": results,
        "primary_gate_results": empirical_gate_results,
        "all_primary_gates_pass": bool(all(empirical_gate_results.values())),
        "interpretation_limits": [
            "Only galaxies passing the residual-blind morphology audit and with at least two points beyond three disk scales enter the test.",
            "Seventeen audited systems have measurable bulges before the outer-radius requirement, limiting empirical power.",
            "SPARC has no local vertical thickness measurements; the observed test uses bulge fraction and fitted bulge scale but not disk thickness.",
            "The fixed RAR is a control relation, not evidence that the tested permittivity model is MOND or a covariant lensing theory.",
        ],
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    pd.concat(all_targets, ignore_index=True).to_csv(
        args.output_directory / "galaxy_targets.csv", index=False
    )
    (args.output_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
