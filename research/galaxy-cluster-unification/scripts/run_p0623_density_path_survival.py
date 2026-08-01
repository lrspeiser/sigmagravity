#!/usr/bin/env python3
"""P0623: test baryonic density/crowding control of the P0554 response."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_arc_invariant_absolute_lensing import (  # noqa: E402
    prepare_galaxies,
    response_for_frame,
)


A0 = 1.2e-10
G_SI = 6.67430e-11
KPC_M = 3.085677581491367e19
M_SUN_KG = 1.98847e30


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    feature: str
    slope: float | None
    parameter_count: int
    predicted_sign: str


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def strict_json(value):
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def safe_positive(values, floor: float = 1.0e-30) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), floor)


def profile_components(row: pd.Series, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized component masses and exponential scales in kpc."""
    minimum = float(settings["minimum_scale_kpc"])
    disk_scale = max(float(row.disk_scale_kpc), minimum)
    disk_mass = max(float(row.disk_mass_solar), 0.0)
    bulge_mass = max(float(row.bulge_mass_solar), 0.0)
    gas_mass = max(float(row.gas_mass_solar), 0.0)
    bulge_scale = float(row.bulge_scale_fit_kpc)
    if not np.isfinite(bulge_scale) or bulge_scale <= 0.0:
        bulge_scale = 0.2 * disk_scale
    hi_radius = float(row.HI_radius_kpc)
    gas_scale = hi_radius / 3.2 if np.isfinite(hi_radius) and hi_radius > 0.0 else 2.0 * disk_scale
    masses = np.asarray([disk_mass, bulge_mass, gas_mass], dtype=float)
    scales = np.maximum(np.asarray([disk_scale, bulge_scale, gas_scale], dtype=float), minimum)
    if masses.sum() <= 0.0:
        masses = np.asarray([1.0, 0.0, 0.0])
    return masses / masses.sum(), scales


def pair_proximity(fractions: np.ndarray, scales: np.ndarray, kernel_kpc: float) -> float:
    """Normalized Gaussian-kernel pair overlap for moment-matched exponential profiles."""
    sigma = np.sqrt(3.0) * np.asarray(scales, dtype=float)
    denominator = kernel_kpc**2 + sigma[:, None] ** 2 + sigma[None, :] ** 2
    kernel = kernel_kpc**2 / denominator
    return float(np.sum(fractions[:, None] * fractions[None, :] * kernel))


def build_feature_frame(points: pd.DataFrame, morphology: pd.DataFrame, protocol: dict):
    profile_settings = protocol["baryonic_profile"]
    morphology = morphology.copy()
    needed = [
        "galaxy",
        "disk_mass_solar",
        "bulge_mass_solar",
        "gas_mass_solar",
        "bulge_scale_fit_kpc",
        "HI_radius_kpc",
        "disk_scale_kpc",
        "disk_central_surface_brightness",
        "inclination_deg",
        "hubble_type",
    ]
    morphology = morphology[needed].rename(columns={"disk_scale_kpc": "profile_disk_scale_kpc"})
    galaxy = points.sort_values("galaxy").drop_duplicates("galaxy").copy()
    galaxy = galaxy.merge(morphology, on="galaxy", how="left", validate="one_to_one")
    galaxy["disk_scale_kpc"] = galaxy.profile_disk_scale_kpc.fillna(galaxy.disk_scale_kpc)
    for column in ("disk_mass_solar", "bulge_mass_solar", "gas_mass_solar"):
        galaxy[column] = galaxy[column].fillna(0.0)
    galaxy["baryonic_profile_mass_solar"] = galaxy[
        ["disk_mass_solar", "bulge_mass_solar", "gas_mass_solar"]
    ].sum(axis=1)
    galaxy["baryonic_profile_mass_solar"] = galaxy.baryonic_profile_mass_solar.where(
        galaxy.baryonic_profile_mass_solar > 0.0,
        galaxy.force_equivalent_mass_solar,
    )

    mass = safe_positive(galaxy.force_equivalent_mass_solar)
    r80 = safe_positive(galaxy.force_equivalent_r80_kpc)
    rd = safe_positive(galaxy.disk_scale_kpc)
    disk_mass = safe_positive(galaxy.disk_mass_solar)
    galaxy["mean_surface_R80"] = mass / (np.pi * r80**2)
    galaxy["mean_volume_R80"] = mass / ((4.0 / 3.0) * np.pi * r80**3)
    galaxy["disk_surface_density"] = disk_mass / (2.0 * np.pi * rd**2)
    galaxy["disk_volume_density"] = disk_mass / (4.0 * np.pi * rd**3)
    galaxy["acceleration_R80"] = G_SI * mass * M_SUN_KG / np.square(r80 * KPC_M)
    galaxy["baryonic_mass"] = mass
    galaxy["R80"] = r80

    pair_specs: list[dict] = []
    for kernel in profile_settings["physical_kernel_scales_kpc"]:
        label = str(kernel).replace(".", "p")
        for name in ("pair_proximity", "pair_surface", "pair_count"):
            pair_specs.append(
                {
                    "feature": f"{name}_L{label}kpc",
                    "base": name,
                    "kernel_mode": "physical",
                    "kernel": float(kernel),
                }
            )
    for multiple in profile_settings["relative_kernel_scales_R80"]:
        label = str(multiple).replace(".", "p")
        for name in ("pair_proximity", "pair_surface"):
            pair_specs.append(
                {
                    "feature": f"{name}_L{label}R80",
                    "base": name,
                    "kernel_mode": "relative",
                    "kernel": float(multiple),
                }
            )

    pair_values = {spec["feature"]: [] for spec in pair_specs}
    component_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in galaxy.itertuples(index=False):
        series = pd.Series(row._asdict())
        fractions, scales = profile_components(series, profile_settings)
        component_cache[str(row.galaxy)] = (fractions, scales)
        total_mass = float(row.force_equivalent_mass_solar)
        for spec in pair_specs:
            kernel = spec["kernel"]
            if spec["kernel_mode"] == "relative":
                kernel *= float(row.force_equivalent_r80_kpc)
            proximity = pair_proximity(fractions, scales, max(kernel, 1.0e-6))
            if spec["base"] == "pair_proximity":
                value = proximity
            elif spec["base"] == "pair_surface":
                value = total_mass * proximity / (2.0 * np.pi * kernel**2)
            else:
                value = total_mass**2 * proximity
            pair_values[spec["feature"]].append(value)
    for feature, values in pair_values.items():
        galaxy[feature] = values

    feature_columns = [
        "mean_surface_R80",
        "mean_volume_R80",
        "disk_surface_density",
        "disk_volume_density",
        "acceleration_R80",
        "potential_depth",
        "baryonic_mass",
        "R80",
        *pair_values.keys(),
    ]
    # Some frozen P0554 invariants (notably potential_depth) already live on every
    # point. Merge only newly constructed global features so pandas cannot silently
    # create _x/_y versions of a scientific input.
    new_global_features = [name for name in feature_columns if name not in points.columns]
    global_values = galaxy[["galaxy", *new_global_features]].copy()
    frame = points.merge(global_values, on="galaxy", how="left", validate="many_to_one")

    local_surface, local_volume, outward_column = [], [], []
    for row in frame.itertuples(index=False):
        fractions, scales = component_cache[str(row.galaxy)]
        total_mass = float(row.force_equivalent_mass_solar)
        masses = total_mass * fractions
        radius = float(row.radius_adjusted_kpc)
        attenuation = np.exp(-radius / scales)
        local_surface.append(float(np.sum(masses / (2.0 * np.pi * scales**2) * attenuation)))
        local_volume.append(float(np.sum(masses / (4.0 * np.pi * scales**3) * attenuation)))
        outward_column.append(float(np.sum(masses / (2.0 * np.pi * scales) * attenuation)))
    frame["local_face_on_surface_density"] = local_surface
    frame["local_volume_density"] = local_volume
    frame["outward_radial_column"] = outward_column
    radius_m = safe_positive(frame.radius_adjusted_kpc) * KPC_M
    enclosed_mass_solar = (
        frame.g_bar_m_s2.to_numpy(float) * radius_m**2 / (G_SI * M_SUN_KG)
    )
    frame["enclosed_mean_volume_density"] = enclosed_mass_solar / (
        (4.0 / 3.0) * np.pi * safe_positive(frame.radius_adjusted_kpc) ** 3
    )
    feature_columns.extend(
        [
            "local_face_on_surface_density",
            "local_volume_density",
            "outward_radial_column",
            "enclosed_mean_volume_density",
        ]
    )

    catalog_rows = []
    for feature in feature_columns:
        if feature == "potential_depth":
            kind = "local_field_invariant"
        elif feature.startswith("pair_"):
            kind = "global_pair"
        elif feature.startswith("local_") or feature in (
            "outward_radial_column",
            "enclosed_mean_volume_density",
        ):
            kind = "local_path"
        else:
            kind = "global_density_or_control"
        catalog_rows.append(
            {
                "feature": feature,
                "kind": kind,
                "varies_with_radius": bool(kind in ("local_path", "local_field_invariant")),
                "uses_observed_velocity": False,
                "finite_fraction": float(np.mean(np.isfinite(frame[feature]))),
                "positive_fraction": float(np.mean(frame[feature].to_numpy(float) > 0.0)),
            }
        )
    return frame, galaxy, pd.DataFrame(catalog_rows), feature_columns


def build_candidates(features: list[str], protocol: dict) -> list[Candidate]:
    response = protocol["candidate_response_families"]
    candidates = [Candidate("constant", "constant", "none", None, 1, "none")]
    for feature in features:
        for slope in response["inverse_hill_zero_floor"]["fixed_slopes"]:
            candidates.append(
                Candidate(
                    f"inverse_hill0_m{slope:g}__{feature}",
                    "inverse_hill_zero_floor",
                    feature,
                    float(slope),
                    2,
                    "inverse",
                )
            )
        for slope in response["inverse_hill_free_floor"]["fixed_slopes"]:
            candidates.append(
                Candidate(
                    f"inverse_hillfloor_m{slope:g}__{feature}",
                    "inverse_hill_free_floor",
                    feature,
                    float(slope),
                    3,
                    "inverse",
                )
            )
        candidates.extend(
            [
                Candidate(
                    f"inverse_loglinear__{feature}",
                    "inverse_log_linear",
                    feature,
                    None,
                    2,
                    "inverse",
                ),
                Candidate(
                    f"free_loglinear__{feature}",
                    "unconstrained_log_linear",
                    feature,
                    None,
                    2,
                    "learned",
                ),
                Candidate(
                    f"quadratic_log__{feature}",
                    "quadratic_log_response_diagnostic",
                    feature,
                    None,
                    3,
                    "nonmonotonic",
                ),
            ]
        )
        for slope in response["direct_hill_wrong_sign_control"]["fixed_slopes"]:
            candidates.append(
                Candidate(
                    f"direct_hill_m{slope:g}__{feature}",
                    "direct_hill_wrong_sign_control",
                    feature,
                    float(slope),
                    2,
                    "direct_wrong_sign",
                )
            )
    return candidates


def standardize_feature(train: pd.DataFrame, other: pd.DataFrame, feature: str):
    train_log = np.log10(safe_positive(train[feature]))
    other_log = np.log10(safe_positive(other[feature]))
    center = float(np.median(train_log))
    q25, q75 = np.quantile(train_log, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1.0e-6:
        scale = float(np.std(train_log))
    if not np.isfinite(scale) or scale < 1.0e-6:
        scale = 1.0
    return (train_log - center) / scale, (other_log - center) / scale, center, scale


def q_from_parameters(candidate: Candidate, parameters: np.ndarray, z: np.ndarray) -> np.ndarray:
    q_max = 6.0
    family = candidate.family
    if family == "constant":
        return np.full(len(z), float(parameters[0]))
    if family == "inverse_hill_zero_floor":
        ceiling, transition = parameters
        return ceiling * expit(-float(candidate.slope) * (z - transition))
    if family == "inverse_hill_free_floor":
        floor, span_fraction, transition = parameters
        ceiling = floor + (q_max - floor) * span_fraction
        return floor + (ceiling - floor) * expit(-float(candidate.slope) * (z - transition))
    if family in ("inverse_log_linear", "unconstrained_log_linear"):
        log_q, slope = parameters
        return np.clip(np.exp(np.clip(log_q + slope * z, -20.0, 4.0)), 0.0, q_max)
    if family == "direct_hill_wrong_sign_control":
        ceiling, transition = parameters
        return ceiling * expit(float(candidate.slope) * (z - transition))
    if family == "quadratic_log_response_diagnostic":
        log_q, linear, quadratic = parameters
        return np.clip(
            np.exp(np.clip(log_q + linear * z + quadratic * z**2, -20.0, 4.0)),
            0.0,
            q_max,
        )
    raise ValueError(f"unknown family {family}")


def parameter_bounds(candidate: Candidate):
    if candidate.family == "constant":
        return [(0.0, 6.0)]
    if candidate.family in ("inverse_hill_zero_floor", "direct_hill_wrong_sign_control"):
        return [(0.0, 6.0), (-6.0, 6.0)]
    if candidate.family == "inverse_hill_free_floor":
        return [(0.0, 6.0), (0.0, 1.0), (-6.0, 6.0)]
    if candidate.family == "inverse_log_linear":
        return [(-6.0, math.log(6.0)), (-4.0, 0.0)]
    if candidate.family == "unconstrained_log_linear":
        return [(-6.0, math.log(6.0)), (-4.0, 4.0)]
    if candidate.family == "quadratic_log_response_diagnostic":
        return [(-6.0, math.log(6.0)), (-4.0, 4.0), (-2.0, 2.0)]
    raise ValueError(candidate.family)


def parameter_starts(candidate: Candidate) -> list[np.ndarray]:
    q0 = 1.23
    if candidate.family == "constant":
        return [np.asarray([q0])]
    if candidate.family in ("inverse_hill_zero_floor", "direct_hill_wrong_sign_control"):
        return [np.asarray([2.0, transition]) for transition in (-1.0, 0.0, 1.0)]
    if candidate.family == "inverse_hill_free_floor":
        return [np.asarray([0.5, 0.35, transition]) for transition in (-1.0, 0.0, 1.0)]
    if candidate.family == "inverse_log_linear":
        return [np.asarray([math.log(q0), slope]) for slope in (-0.1, -0.5, -1.0)]
    if candidate.family == "unconstrained_log_linear":
        return [np.asarray([math.log(q0), slope]) for slope in (-0.5, 0.0, 0.5)]
    if candidate.family == "quadratic_log_response_diagnostic":
        return [
            np.asarray([math.log(q0), 0.0, 0.0]),
            np.asarray([math.log(q0), -0.5, 0.1]),
            np.asarray([math.log(q0), 0.5, -0.1]),
        ]
    raise ValueError(candidate.family)


def predict_velocity(frame: pd.DataFrame, q_eff: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.maximum(
            frame.g_bar_m_s2.to_numpy(float)
            * (1.0 + q_eff * frame.unit_P0554_response.to_numpy(float))
            * frame.radius_adjusted_kpc.to_numpy(float)
            * KPC_M
            / 1.0e6,
            0.0,
        )
    )


def score_arrays(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    residual = prediction - frame.velocity_observed_adjusted_km_s.to_numpy(float)
    local = pd.DataFrame(
        {"galaxy": frame.galaxy.to_numpy(), "squared": residual**2, "residual": residual}
    )
    per_galaxy = local.groupby("galaxy", sort=False).agg(
        mse=("squared", "mean"), mean_residual=("residual", "mean")
    )
    return {
        "galaxies": int(frame.galaxy.nunique()),
        "points": int(len(frame)),
        "pooled_RMSE_km_s": float(np.sqrt(np.mean(residual**2))),
        "equal_galaxy_RMSE_km_s": float(np.sqrt(per_galaxy.mse.mean())),
        "mean_residual_km_s": float(np.mean(residual)),
        "median_galaxy_mean_residual_km_s": float(per_galaxy.mean_residual.median()),
    }


def objective(frame: pd.DataFrame, candidate: Candidate, parameters: np.ndarray, z: np.ndarray):
    q_eff = q_from_parameters(candidate, parameters, z)
    prediction = predict_velocity(frame, q_eff)
    residual2 = np.square(prediction - frame.velocity_observed_adjusted_km_s.to_numpy(float))
    codes, _ = pd.factorize(frame.galaxy, sort=False)
    means = np.bincount(codes, weights=residual2) / np.bincount(codes)
    return float(np.mean(means))


def fit_candidate(frame: pd.DataFrame, candidate: Candidate, z: np.ndarray) -> np.ndarray:
    if candidate.family == "constant":
        result = minimize_scalar(
            lambda value: objective(frame, candidate, np.asarray([value]), z),
            bounds=(0.0, 6.0),
            method="bounded",
            options={"xatol": 1.0e-7, "maxiter": 300},
        )
        if not result.success:
            raise RuntimeError("constant q optimization failed")
        return np.asarray([float(result.x)])
    best = None
    for start in parameter_starts(candidate):
        result = minimize(
            lambda values: objective(frame, candidate, values, z),
            start,
            method="L-BFGS-B",
            bounds=parameter_bounds(candidate),
            options={"maxiter": 300, "ftol": 1.0e-10, "gtol": 1.0e-6},
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError(f"optimization failed for {candidate.candidate_id}")
    return np.asarray(best.x, dtype=float)


def evaluate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: Candidate,
    *,
    split_label: str,
):
    if candidate.family == "constant":
        z_train = np.zeros(len(train))
        z_test = np.zeros(len(test))
        feature_center = np.nan
        feature_scale = np.nan
    else:
        z_train, z_test, feature_center, feature_scale = standardize_feature(
            train, test, candidate.feature
        )
    parameters = fit_candidate(train, candidate, z_train)
    q_train = q_from_parameters(candidate, parameters, z_train)
    q_test = q_from_parameters(candidate, parameters, z_test)
    train_metrics = score_arrays(train, predict_velocity(train, q_train))
    test_metrics = score_arrays(test, predict_velocity(test, q_test))
    row = {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "feature": candidate.feature,
        "fixed_slope": candidate.slope,
        "parameter_count": candidate.parameter_count,
        "predicted_sign": candidate.predicted_sign,
        "split": split_label,
        "feature_log10_center": feature_center,
        "feature_log10_IQR": feature_scale,
        "parameters_json": json.dumps([float(value) for value in parameters]),
        "train_q_min": float(np.min(q_train)),
        "train_q_median": float(np.median(q_train)),
        "train_q_max": float(np.max(q_train)),
        "test_q_min": float(np.min(q_test)),
        "test_q_median": float(np.median(q_test)),
        "test_q_max": float(np.max(q_test)),
    }
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    predictions = test[["galaxy", "galaxy_fold", "radius_adjusted_kpc"]].copy()
    predictions["candidate_id"] = candidate.candidate_id
    predictions["q_eff"] = q_test
    predictions["velocity_predicted_km_s"] = predict_velocity(test, q_test)
    predictions["velocity_observed_km_s"] = test.velocity_observed_adjusted_km_s.to_numpy(float)
    return row, predictions


def run_cv(frame: pd.DataFrame, candidates: list[Candidate], development_folds: list[int]):
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        if index == 1 or index % 50 == 0:
            print(f"CV candidate {index}/{len(candidates)}: {candidate.candidate_id}", flush=True)
        for heldout_fold in development_folds:
            train = frame[
                frame.galaxy_fold.isin(development_folds) & frame.galaxy_fold.ne(heldout_fold)
            ]
            test = frame[frame.galaxy_fold.eq(heldout_fold)]
            row, _ = evaluate_split(
                train,
                test,
                candidate,
                split_label=f"development_fold_{heldout_fold}",
            )
            row["heldout_fold"] = int(heldout_fold)
            rows.append(row)
    folds = pd.DataFrame(rows)
    baseline = folds[folds.candidate_id.eq("constant")][
        ["heldout_fold", "test_equal_galaxy_RMSE_km_s"]
    ].rename(columns={"test_equal_galaxy_RMSE_km_s": "baseline_fold_RMSE_km_s"})
    folds = folds.merge(baseline, on="heldout_fold", how="left", validate="many_to_one")
    folds["fold_improvement_fraction"] = (
        1.0 - folds.test_equal_galaxy_RMSE_km_s / folds.baseline_fold_RMSE_km_s
    )
    aggregate = folds.groupby(
        [
            "candidate_id",
            "family",
            "feature",
            "fixed_slope",
            "parameter_count",
            "predicted_sign",
        ],
        dropna=False,
        sort=False,
    ).agg(
        cv_equal_galaxy_MSE=("test_equal_galaxy_RMSE_km_s", lambda values: np.mean(np.square(values))),
        cv_pooled_MSE=("test_pooled_RMSE_km_s", lambda values: np.mean(np.square(values))),
        mean_fold_improvement_fraction=("fold_improvement_fraction", "mean"),
        median_fold_improvement_fraction=("fold_improvement_fraction", "median"),
        fold_wins=("fold_improvement_fraction", lambda values: int(np.sum(np.asarray(values) > 0.0))),
        q_min=("test_q_min", "min"),
        q_median=("test_q_median", "median"),
        q_max=("test_q_max", "max"),
    ).reset_index()
    aggregate["cv_equal_galaxy_RMSE_km_s"] = np.sqrt(aggregate.pop("cv_equal_galaxy_MSE"))
    aggregate["cv_pooled_RMSE_km_s"] = np.sqrt(aggregate.pop("cv_pooled_MSE"))
    baseline_rmse = float(
        aggregate.loc[aggregate.candidate_id.eq("constant"), "cv_equal_galaxy_RMSE_km_s"].iloc[0]
    )
    aggregate["improvement_vs_constant_fraction"] = (
        1.0 - aggregate.cv_equal_galaxy_RMSE_km_s / baseline_rmse
    )
    aggregate = aggregate.sort_values(
        ["cv_equal_galaxy_RMSE_km_s", "parameter_count", "candidate_id"]
    ).reset_index(drop=True)
    aggregate["rank"] = np.arange(1, len(aggregate) + 1)
    return aggregate, folds


def choose_candidates(scores: pd.DataFrame, candidates: list[Candidate], protocol: dict):
    selection = protocol["selection"]
    inverse = scores[
        scores.predicted_sign.eq("inverse")
        & (scores.improvement_vs_constant_fraction >= float(selection["minimum_internal_cv_improvement_vs_refit_constant_fraction"]))
        & (scores.fold_wins >= int(selection["minimum_fold_wins_out_of_4"]))
    ]
    best_inverse_id = str(inverse.iloc[0].candidate_id) if len(inverse) else None
    best_overall_id = str(scores.iloc[0].candidate_id)
    wrong = scores[scores.predicted_sign.eq("direct_wrong_sign")]
    best_wrong_id = str(wrong.iloc[0].candidate_id) if len(wrong) else None
    chosen_ids = ["constant", best_inverse_id, best_overall_id, best_wrong_id]
    chosen_ids = [value for index, value in enumerate(chosen_ids) if value and value not in chosen_ids[:index]]
    lookup = {candidate.candidate_id: candidate for candidate in candidates}
    return [lookup[value] for value in chosen_ids], best_inverse_id, best_overall_id, best_wrong_id


def score_holdout(
    frame: pd.DataFrame,
    selected: list[Candidate],
    development_folds: list[int],
    holdout_fold: int,
):
    train = frame[frame.galaxy_fold.isin(development_folds)]
    test = frame[frame.galaxy_fold.eq(holdout_fold)]
    rows, prediction_blocks = [], []
    for candidate in selected:
        row, predictions = evaluate_split(
            train,
            test,
            candidate,
            split_label=f"chronological_fold_{holdout_fold}",
        )
        rows.append(row)
        prediction_blocks.append(predictions)
    scores = pd.DataFrame(rows)
    baseline = float(
        scores.loc[scores.candidate_id.eq("constant"), "test_equal_galaxy_RMSE_km_s"].iloc[0]
    )
    scores["holdout_improvement_vs_constant_fraction"] = (
        1.0 - scores.test_equal_galaxy_RMSE_km_s / baseline
    )
    return scores.sort_values("test_equal_galaxy_RMSE_km_s"), pd.concat(prediction_blocks)


def fit_full_predictions(frame: pd.DataFrame, selected: list[Candidate]):
    rows, blocks = [], []
    for candidate in selected:
        row, predictions = evaluate_split(frame, frame, candidate, split_label="full_descriptive_refit")
        rows.append(row)
        blocks.append(predictions)
    return pd.DataFrame(rows), pd.concat(blocks, ignore_index=True)


def per_galaxy_scores(frame: pd.DataFrame, predictions: pd.DataFrame, galaxy_features: pd.DataFrame):
    joined = predictions.merge(
        frame[
            [
                "galaxy",
                "radius_adjusted_kpc",
                "velocity_RAR_same_nuisance_km_s",
                "baryonic_mass",
                "gas_fraction",
                "stellar_bulge_fraction",
            ]
        ],
        on=["galaxy", "radius_adjusted_kpc"],
        how="left",
        validate="many_to_one",
    )
    joined["squared"] = np.square(
        joined.velocity_predicted_km_s - joined.velocity_observed_km_s
    )
    joined["residual"] = joined.velocity_predicted_km_s - joined.velocity_observed_km_s
    joined["rar_squared"] = np.square(
        joined.velocity_RAR_same_nuisance_km_s - joined.velocity_observed_km_s
    )
    result = joined.groupby(["candidate_id", "galaxy"], sort=True).agg(
        points=("galaxy", "size"),
        MSE=("squared", "mean"),
        mean_residual_km_s=("residual", "mean"),
        mean_q_eff=("q_eff", "mean"),
        RAR_MSE=("rar_squared", "mean"),
        baryonic_mass_solar=("baryonic_mass", "first"),
        gas_fraction=("gas_fraction", "first"),
        stellar_bulge_fraction=("stellar_bulge_fraction", "first"),
    ).reset_index()
    result["RMSE_km_s"] = np.sqrt(result.MSE)
    result["RAR_RMSE_km_s"] = np.sqrt(result.RAR_MSE)
    result["mass_regime"] = np.select(
        [result.baryonic_mass_solar < 1.0e9, result.baryonic_mass_solar > 1.0e10],
        ["dwarf_below_1e9", "giant_above_1e10"],
        default="intermediate_1e9_to_1e10",
    )
    return result


def regime_scores(per_galaxy: pd.DataFrame):
    rows = []
    for (candidate_id, regime), block in per_galaxy.groupby(["candidate_id", "mass_regime"]):
        rows.append(
            {
                "candidate_id": candidate_id,
                "dimension": "mass_regime",
                "regime": regime,
                "galaxies": len(block),
                "equal_galaxy_RMSE_km_s": float(np.sqrt(block.MSE.mean())),
                "mean_galaxy_residual_km_s": float(block.mean_residual_km_s.mean()),
                "mean_q_eff": float(block.mean_q_eff.mean()),
                "RAR_equal_galaxy_RMSE_km_s": float(np.sqrt(block.RAR_MSE.mean())),
            }
        )
    return pd.DataFrame(rows)


def write_figure(output: Path, cv: pd.DataFrame, holdout: pd.DataFrame, regimes: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    top = cv.head(20).iloc[::-1]
    colors = np.where(top.predicted_sign.eq("inverse"), "#2b8cbe", "#9e9ac8")
    axes[0].barh(top.candidate_id, 100.0 * top.improvement_vs_constant_fraction, color=colors)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("development-CV improvement vs constant (%)")
    axes[0].set_title("Broad search: best 20")
    axes[0].tick_params(axis="y", labelsize=6)

    axes[1].bar(
        holdout.candidate_id,
        holdout.test_equal_galaxy_RMSE_km_s,
        color="#31a354",
    )
    axes[1].set_ylabel("equal-galaxy RMSE (km/s)")
    axes[1].set_title("Chronological P0623 fold")
    axes[1].tick_params(axis="x", rotation=70, labelsize=7)

    pivot = regimes.pivot(index="regime", columns="candidate_id", values="mean_galaxy_residual_km_s")
    pivot.plot(kind="bar", ax=axes[2])
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("mean residual (km/s)")
    axes[2].set_title("Dwarf-to-giant bias after full refit")
    axes[2].tick_params(axis="x", rotation=30, labelsize=7)
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_summary(output: Path, report: dict, cv: pd.DataFrame, holdout: pd.DataFrame):
    best = cv.iloc[0]
    selected = report["selection"]
    holdout_lines = []
    for row in holdout.itertuples(index=False):
        holdout_lines.append(
            f"- `{row.candidate_id}`: {row.test_equal_galaxy_RMSE_km_s:.3f} km/s "
            f"({100.0 * row.holdout_improvement_vs_constant_fraction:+.2f}% vs constant)."
        )
    text = f"""# P0623 density/path-survival galaxy screen

## Result at this checkpoint

The broad search evaluated **{report['counts']['candidate_formulas']} formulas** made from
**{report['counts']['features']} baryon-only features**. The best development-CV candidate was
`{best.candidate_id}` at {best.cv_equal_galaxy_RMSE_km_s:.3f} km/s, an improvement of
{100.0 * best.improvement_vs_constant_fraction:.2f}% over the fold-refit constant-strength control.

The best preregistered inverse-density candidate was
`{selected['best_inverse_candidate']}`. Mechanism support at the development gate is
**{selected['inverse_development_gate_pass']}**.

## Chronological P0623 fold

{chr(10).join(holdout_lines)}

The chronological fold is untouched only by P0623 selection; the galaxies are project-spent and
this is not independent external confirmation.

## Interpretation limits

The pair statistic is a deliberately testable path-encounter proxy, not a derivation from QED.
This checkpoint concerns galaxy behavior only. Cluster and Solar transfer are required before any
cross-domain promotion.
"""
    output.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/p0623_density_path_survival_protocol.json",
    )
    args = parser.parse_args()
    protocol = load_json(args.config)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)

    parent = load_json(protocol["inputs"]["SPARC_parent_protocol"])
    points, _ = prepare_galaxies(parent, A0)
    outer = points[points.split.eq("outer_holdout")].copy()
    if outer.galaxy.nunique() != int(protocol["sample"]["expected_galaxies"]):
        raise ValueError("P0623 galaxy count does not match frozen protocol")
    if len(outer) != int(protocol["sample"]["expected_points"]):
        raise ValueError("P0623 outer-point count does not match frozen protocol")
    p0554 = load_json(protocol["inputs"]["P0554_protocol"])
    spec = dict(p0554["baseline"])
    spec.pop("universal_q")
    spec["candidate_id"] = "P0623_unit_parent"
    response = response_for_frame(
        outer,
        spec,
        q=1.0,
        a0=A0,
        radius_column="radius_adjusted_kpc",
        gbar_column="g_bar_m_s2",
    )
    outer["unit_P0554_response"] = response["unit_fractional_response"]
    morphology = pd.read_csv(ROOT / protocol["inputs"]["SPARC_morphology"])
    frame, galaxy_features, feature_catalog, feature_columns = build_feature_frame(
        outer, morphology, protocol
    )
    feature_catalog.to_csv(output / protocol["outputs"]["feature_catalog"], index=False)
    galaxy_feature_columns = [
        "galaxy",
        "galaxy_fold",
        "force_equivalent_mass_solar",
        "force_equivalent_r80_kpc",
        "gas_fraction",
        "stellar_bulge_fraction",
        "inclination_adjusted_deg",
        *[name for name in feature_columns if name in galaxy_features],
    ]
    galaxy_features[galaxy_feature_columns].to_csv(
        output / protocol["outputs"]["feature_values"], index=False
    )

    candidates = build_candidates(feature_columns, protocol)
    development_folds = [int(value) for value in protocol["sample"]["development_galaxy_folds"]]
    cv_scores, cv_folds = run_cv(frame, candidates, development_folds)
    cv_scores.to_csv(output / protocol["outputs"]["cv_candidates"], index=False)
    cv_folds.to_csv(output / protocol["outputs"]["cv_folds"], index=False)

    selected, best_inverse, best_overall, best_wrong = choose_candidates(
        cv_scores, candidates, protocol
    )
    survivor_count = int(protocol["selection"]["maximum_survivors_for_refinement"])
    survivors = cv_scores.head(survivor_count).copy()
    survivors["selected_for_holdout"] = survivors.candidate_id.isin(
        [candidate.candidate_id for candidate in selected]
    )
    survivors.to_csv(output / protocol["outputs"]["survivors"], index=False)

    holdout_fold = int(protocol["sample"]["chronological_formula_holdout_fold"])
    holdout, holdout_predictions = score_holdout(
        frame, selected, development_folds, holdout_fold
    )
    holdout.to_csv(output / protocol["outputs"]["holdout"], index=False)
    full_fit, full_predictions = fit_full_predictions(frame, selected)
    per_galaxy = per_galaxy_scores(frame, full_predictions, galaxy_features)
    per_galaxy.to_csv(output / protocol["outputs"]["per_galaxy"], index=False)
    regimes = regime_scores(per_galaxy)
    regimes.to_csv(output / protocol["outputs"]["regimes"], index=False)

    best_inverse_row = (
        cv_scores[cv_scores.candidate_id.eq(best_inverse)].iloc[0] if best_inverse else None
    )
    inverse_gate = bool(
        best_inverse_row is not None
        and best_inverse_row.improvement_vs_constant_fraction
        >= float(protocol["selection"]["minimum_internal_cv_improvement_vs_refit_constant_fraction"])
        and int(best_inverse_row.fold_wins)
        >= int(protocol["selection"]["minimum_fold_wins_out_of_4"])
    )
    holdout_lookup = holdout.set_index("candidate_id")
    inverse_holdout_pass = bool(
        best_inverse is not None
        and best_inverse in holdout_lookup.index
        and float(holdout_lookup.loc[best_inverse, "holdout_improvement_vs_constant_fraction"])
        >= float(protocol["selection"]["holdout_must_not_worsen_constant_RMSE_fraction"])
    )
    report = {
        "protocol_version": protocol["protocol_version"],
        "status": "galaxy_screen_complete_cluster_and_solar_transfer_pending",
        "counts": {
            "galaxies": int(frame.galaxy.nunique()),
            "outer_points": int(len(frame)),
            "features": len(feature_columns),
            "candidate_formulas": len(candidates),
            "candidate_fold_fits": len(cv_folds),
        },
        "selection": {
            "best_overall_candidate": best_overall,
            "best_inverse_candidate": best_inverse,
            "best_wrong_sign_control": best_wrong,
            "holdout_candidates": [candidate.candidate_id for candidate in selected],
            "inverse_development_gate_pass": inverse_gate,
            "inverse_chronological_holdout_pass": inverse_holdout_pass,
            "galaxy_promotion_pass": bool(inverse_gate and inverse_holdout_pass),
        },
        "best_cv": strict_json(cv_scores.iloc[0].to_dict()),
        "chronological_holdout": strict_json(
            holdout[
                [
                    "candidate_id",
                    "test_equal_galaxy_RMSE_km_s",
                    "test_pooled_RMSE_km_s",
                    "holdout_improvement_vs_constant_fraction",
                    "test_q_min",
                    "test_q_median",
                    "test_q_max",
                ]
            ].to_dict(orient="records")
        ),
        "full_descriptive_fits": strict_json(full_fit.to_dict(orient="records")),
        "regime_scores": strict_json(regimes.to_dict(orient="records")),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(strict_json(report), indent=2), encoding="utf-8"
    )
    write_figure(output / protocol["outputs"]["figure"], cv_scores, holdout, regimes)
    write_summary(output / protocol["outputs"]["summary"], report, cv_scores, holdout)
    print(json.dumps(report["selection"], indent=2), flush=True)


if __name__ == "__main__":
    main()
