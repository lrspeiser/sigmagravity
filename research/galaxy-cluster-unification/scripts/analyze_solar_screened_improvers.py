#!/usr/bin/env python3
"""Forensic statistical audit of galaxies where the screened law beats fixed RAR."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_sparc_independent_nuisance_refit import build_frame


ROOT = Path(__file__).resolve().parents[1]
KPC_M = 3.085677581491367e19
warnings.simplefilter("ignore", PerformanceWarning)


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


def fdr_bh(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not np.any(valid):
        return result
    raw = values[valid]
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    unsorted = np.empty_like(adjusted)
    unsorted[order] = adjusted
    result[valid] = unsorted
    return result


def safe_slope(radius, velocity) -> float:
    radius = np.asarray(radius, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    valid = np.isfinite(radius) & np.isfinite(velocity) & (radius > 0.0) & (velocity > 0.0)
    if valid.sum() < 3:
        return math.nan
    return float(np.polyfit(np.log(radius[valid]), np.log(velocity[valid]), 1)[0])


def safe_curvature(radius, velocity) -> float:
    radius = np.asarray(radius, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    valid = np.isfinite(radius) & np.isfinite(velocity) & (radius > 0.0) & (velocity > 0.0)
    if valid.sum() < 4:
        return math.nan
    x = np.log(radius[valid])
    x -= np.mean(x)
    return float(np.polyfit(x, np.log(velocity[valid]), 2)[0])


def parse_table1_extended(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(
                {
                    "galaxy": line[0:11].strip(),
                    "distance_error_mpc": float(line[22:27]),
                    "distance_method": int(line[28:29]),
                    "inclination_error_deg": float(line[35:39]),
                    "luminosity_error_billion_solar": float(line[48:55]),
                    "reference_codes": line[116:130].strip(),
                }
            )
    return pd.DataFrame(rows)


def compute_outcomes(points: pd.DataFrame) -> pd.DataFrame:
    selected = points[points["model"].isin(
        ["solar_screened_isothermal", "fixed_RAR", "simple_MOND"]
    )].copy()
    selected["residual_squared"] = np.square(
        selected["velocity_predicted_km_s"]
        - selected["velocity_observed_adjusted_km_s"]
    )
    selected["catalog_residual_squared"] = np.square(
        selected["velocity_predicted_catalog_km_s"]
        - selected["velocity_observed_catalog_kms"]
    )
    selected["standardized_residual_squared"] = (
        selected["residual_squared"] / np.square(selected["velocity_error_total_km_s"])
    )
    outer = selected[selected["split"] == "outer_holdout"].copy()
    grouped = (
        outer.groupby(["galaxy", "model"], sort=True)
        .agg(
            points=("residual_squared", "size"),
            MSE=("residual_squared", "mean"),
            catalog_MSE=("catalog_residual_squared", "mean"),
            chi2=("standardized_residual_squared", "mean"),
            mean_residual=(
                "velocity_predicted_km_s",
                lambda values: 0.0,
            ),
            observed_mean_square=(
                "velocity_observed_adjusted_km_s",
                lambda values: float(np.mean(np.square(values))),
            ),
        )
        .reset_index()
    )
    actual_residual = (
        outer.assign(
            residual=(
                outer["velocity_predicted_km_s"]
                - outer["velocity_observed_adjusted_km_s"]
            )
        )
        .groupby(["galaxy", "model"])["residual"]
        .mean()
        .rename("mean_residual_actual")
        .reset_index()
    )
    grouped = grouped.drop(columns="mean_residual").merge(
        actual_residual, on=["galaxy", "model"], validate="one_to_one"
    )
    wide = grouped.pivot(index="galaxy", columns="model")
    output = pd.DataFrame(index=wide.index)
    model_names = {
        "screened": "solar_screened_isothermal",
        "RAR": "fixed_RAR",
        "MOND": "simple_MOND",
    }
    for short, model in model_names.items():
        for metric in [
            "points",
            "MSE",
            "catalog_MSE",
            "chi2",
            "mean_residual_actual",
            "observed_mean_square",
        ]:
            output[f"outer_{metric}_{short}"] = wide[(metric, model)]

    solar_outer = outer[outer["model"] == "solar_screened_isothermal"].copy()
    solar_outer["RAR_same_nuisance_residual_squared"] = np.square(
        solar_outer["velocity_RAR_same_nuisance_km_s"]
        - solar_outer["velocity_observed_adjusted_km_s"]
    )
    same_nuisance = solar_outer.groupby("galaxy").agg(
        outer_MSE_RAR_same_screened_nuisance=("RAR_same_nuisance_residual_squared", "mean")
    )
    output = output.join(same_nuisance)
    denom = output["outer_observed_mean_square_RAR"]
    output["primary_improver"] = output["outer_MSE_screened"] < output["outer_MSE_RAR"]
    output["continuous_skill"] = (
        output["outer_MSE_RAR"] - output["outer_MSE_screened"]
    ) / denom
    output["fractional_MSE_change_screened_vs_RAR"] = (
        output["outer_MSE_screened"] / output["outer_MSE_RAR"] - 1.0
    )
    output["improver_vs_MOND"] = output["outer_MSE_screened"] < output["outer_MSE_MOND"]
    output["improver_by_chi2"] = output["outer_chi2_screened"] < output["outer_chi2_RAR"]
    output["improver_in_catalog_space"] = (
        output["outer_catalog_MSE_screened"] < output["outer_catalog_MSE_RAR"]
    )
    output["tail_helps_at_same_screened_nuisance"] = (
        output["outer_MSE_screened"]
        < output["outer_MSE_RAR_same_screened_nuisance"]
    )
    output["improvement_at_least_10pct"] = (
        output["outer_MSE_screened"] <= 0.9 * output["outer_MSE_RAR"]
    )
    output["improvement_at_least_20pct"] = (
        output["outer_MSE_screened"] <= 0.8 * output["outer_MSE_RAR"]
    )
    output["nuisance_refit_advantage_fraction"] = (
        output["outer_MSE_RAR_same_screened_nuisance"]
        - output["outer_MSE_RAR"]
    ) / denom
    return output.reset_index()


def build_features(
    frame: pd.DataFrame,
    points: pd.DataFrame,
    morphology: pd.DataFrame,
    assignments: pd.DataFrame,
    extended_table1: pd.DataFrame,
    coordinates: pd.DataFrame,
    cf4: pd.DataFrame,
    void_wall: pd.DataFrame,
    void_cage: pd.DataFrame,
    screened_fits: pd.DataFrame,
    baseline_fits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = sorted(frame["galaxy"].unique())
    features = pd.DataFrame({"galaxy": names})
    manifest_rows: list[dict] = []

    def add(
        name: str,
        values,
        *,
        tier: str,
        source: str,
        description: str,
        kind: str = "numeric",
        leakage: bool = False,
        pairwise: bool = False,
    ) -> None:
        nonlocal features
        if name in features.columns:
            raise ValueError(f"duplicate feature {name}")
        if isinstance(values, pd.Series) and values.index.dtype == object:
            mapped = features["galaxy"].map(values)
        elif isinstance(values, dict):
            mapped = features["galaxy"].map(values)
        else:
            array = np.asarray(values)
            if len(array) != len(features):
                raise ValueError(f"feature {name} has wrong length")
            mapped = pd.Series(array, index=features.index)
        features[name] = mapped
        manifest_rows.append(
            {
                "feature": name,
                "tier": tier,
                "source": source,
                "description": description,
                "kind": kind,
                "uses_outer_observed_velocity": leakage,
                "pairwise_candidate": pairwise,
            }
        )

    catalog = morphology[morphology["galaxy"].isin(names)].set_index("galaxy")
    excluded_catalog = {"morphology_input_pass", "fold"}
    leakage_catalog = {"flat_velocity_km_s", "flat_velocity_error_km_s"}
    for column in morphology.columns:
        if column == "galaxy" or column in excluded_catalog:
            continue
        numeric = pd.to_numeric(catalog[column], errors="coerce")
        leakage = column in leakage_catalog
        add(
            f"catalog__{column}",
            numeric,
            tier="outer_descriptive_leakage" if leakage else "core_preoutcome",
            source="SPARC global catalog / derived morphology",
            description=column.replace("_", " "),
            leakage=leakage,
            pairwise=not leakage,
        )

    extended = extended_table1.set_index("galaxy")
    for column in [
        "distance_error_mpc",
        "inclination_error_deg",
        "luminosity_error_billion_solar",
    ]:
        add(
            f"measurement__{column}",
            extended[column],
            tier="core_preoutcome",
            source="SPARC table1",
            description=column.replace("_", " "),
            pairwise=True,
        )
    add(
        "category__distance_method",
        extended["distance_method"].astype(str),
        tier="core_preoutcome",
        source="SPARC table1",
        description="distance-estimation method code",
        kind="categorical",
    )
    reference_sets = extended["reference_codes"].fillna("").map(
        lambda value: {part.strip() for part in str(value).split(",") if part.strip()}
    )
    all_references = sorted(set().union(*reference_sets.tolist()))
    add(
        "measurement__reference_code_count",
        reference_sets.map(len),
        tier="core_preoutcome",
        source="SPARC table1",
        description="number of cited rotation-curve reference codes",
        pairwise=False,
    )
    for code in all_references:
        indicator = reference_sets.map(lambda values: code in values)
        if int(indicator.sum()) >= 5:
            add(
                f"reference__{code}",
                indicator,
                tier="core_preoutcome",
                source="SPARC table1",
                description=f"rotation-curve references include {code}",
                kind="boolean",
            )

    assigned = assignments.set_index("galaxy")
    for column in [
        "stellar_structure",
        "hubble_family",
        "surface_brightness_family",
        "baryonic_mass_family",
        "gas_fraction_family",
        "inclination_family",
    ]:
        add(
            f"category__{column}",
            assigned[column].astype(str),
            tier="core_preoutcome",
            source="frozen morphology assignment",
            description=column.replace("_", " "),
            kind="categorical",
        )
    add(
        "category__outer_rotation_shape",
        assigned["outer_rotation_shape"].astype(str),
        tier="outer_descriptive_leakage",
        source="observed outer rotation curve",
        description="declining, approximately flat, or rising outer observed curve",
        kind="categorical",
        leakage=True,
    )

    coords = coordinates.rename(columns={"name": "galaxy"}).set_index("galaxy")
    for column in ["ra_deg", "dec_deg"]:
        add(
            f"coordinate__{column}",
            coords[column],
            tier="core_preoutcome",
            source="NED-resolved SPARC coordinates",
            description=column,
        )

    cf4_index = cf4.set_index("galaxy")
    for column in cf4.columns:
        if column == "galaxy":
            continue
        add(
            f"environment_cf4__{column}",
            pd.to_numeric(cf4_index[column], errors="coerce"),
            tier="core_preoutcome",
            source="independent Cosmicflows-4 density grids",
            description=column.replace("_", " "),
            pairwise=column.startswith("void_score") or column.startswith("delta_"),
        )

    wall_index = void_wall.set_index("galaxy")
    for column in void_wall.columns:
        if column == "galaxy" or column in {"void_index"}:
            continue
        if column == "inside_catalog_void":
            values = wall_index[column].astype(str).str.lower().map({"true": 1.0, "false": 0.0})
            kind = "boolean"
        else:
            values = pd.to_numeric(wall_index[column], errors="coerce")
            kind = "numeric"
        add(
            f"environment_voidwall__{column}",
            values,
            tier="core_preoutcome",
            source="independent Local-Voids Voronoi wall catalog",
            description=column.replace("_", " "),
            kind=kind,
            pairwise=column in {"inside_catalog_void", "void_wall_score", "wall_distance_hmpc"},
        )

    cage_index = void_cage.set_index("galaxy")
    cage_skip = {"galaxy", "ra_deg", "dec_deg", "distance_mpc", "sgx_hmpc", "sgy_hmpc", "sgz_hmpc"}
    for column in void_cage.columns:
        if column in cage_skip:
            continue
        raw = cage_index[column]
        if raw.astype(str).str.lower().isin(["true", "false"]).all():
            values = raw.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
            kind = "boolean"
        else:
            values = pd.to_numeric(raw, errors="coerce")
            kind = "numeric"
        add(
            f"environment_cage__{column}",
            values,
            tier="extended_environment",
            source="multiscale Cosmicflows void-cage reconstruction",
            description=column.replace("_", " "),
            kind=kind,
        )

    xyz = cf4_index.loc[names, ["sgx_hmpc", "sgy_hmpc", "sgz_hmpc"]].to_numpy(dtype=float)
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    ordered_distances = np.sort(distances, axis=1)
    add(
        "environment_sample__nearest_SPARC_neighbor_hmpc",
        ordered_distances[:, 0],
        tier="core_preoutcome",
        source="SPARC-sample 3-D positions",
        description="nearest neighbor within the incomplete SPARC sample",
        pairwise=True,
    )
    add(
        "environment_sample__fifth_neighbor_hmpc",
        ordered_distances[:, 4],
        tier="core_preoutcome",
        source="SPARC-sample 3-D positions",
        description="fifth-nearest neighbor within the incomplete SPARC sample",
        pairwise=True,
    )

    frame_indexed = frame.set_index("galaxy")
    for galaxy in names:
        block = frame_indexed.loc[[galaxy]].copy()
        disk_scale = float(block["disk_scale_kpc"].iloc[0])
        for split_name, selected_block in [
            ("all", block),
            ("inner", block[block["split"] == "inner_train"]),
            ("outer", block[block["split"] == "outer_holdout"]),
        ]:
            radius = selected_block["radius_catalog_kpc"].to_numpy(dtype=float)
            gas = selected_block["gas_velocity_component_km_s"].to_numpy(dtype=float)
            disk = selected_block["disk_velocity_unit_ml_km_s"].to_numpy(dtype=float)
            bulge = selected_block["bulge_velocity_unit_ml_km_s"].to_numpy(dtype=float)
            positive = np.square(gas) + 0.5 * np.square(disk) + 0.7 * np.square(bulge)
            component_values = {
                f"component__{split_name}_radius_min_kpc": float(np.min(radius)),
                f"component__{split_name}_radius_max_kpc": float(np.max(radius)),
                f"component__{split_name}_radius_span_dex": float(np.log10(np.max(radius) / np.min(radius))),
                f"component__{split_name}_radius_max_over_Rdisk": float(np.max(radius) / disk_scale),
                f"component__{split_name}_gas_force_fraction_median": float(np.median(np.square(gas) / positive)),
                f"component__{split_name}_disk_force_fraction_median": float(np.median(0.5 * np.square(disk) / positive)),
                f"component__{split_name}_bulge_force_fraction_median": float(np.median(0.7 * np.square(bulge) / positive)),
                f"component__{split_name}_signed_gas_v2_fraction_median": float(np.median(np.sign(gas) * np.square(gas) / positive)),
                f"component__{split_name}_negative_gas_fraction": float(np.mean(gas < 0.0)),
                f"component__{split_name}_disk_surface_brightness_median": float(np.median(selected_block["disk_surface_brightness"])),
                f"component__{split_name}_bulge_surface_brightness_median": float(np.median(selected_block["bulge_surface_brightness"])),
            }
            if galaxy == names[0]:
                component_store = {key: {} for key in component_values}
            for key, value in component_values.items():
                component_store.setdefault(key, {})[galaxy] = value

            observed = selected_block["velocity_observed_catalog_kms"].to_numpy(dtype=float)
            errors = selected_block["velocity_error_catalog_kms"].to_numpy(dtype=float)
            observed_values = {
                f"curve__{split_name}_observed_velocity_mean_km_s": float(np.mean(observed)),
                f"curve__{split_name}_observed_velocity_max_km_s": float(np.max(observed)),
                f"curve__{split_name}_observed_velocity_last_km_s": float(observed[-1]),
                f"curve__{split_name}_observed_log_slope": safe_slope(radius, observed),
                f"curve__{split_name}_observed_log_curvature": safe_curvature(radius, observed),
                f"curve__{split_name}_observed_fractional_scatter": float(np.std(observed) / np.mean(observed)),
                f"curve__{split_name}_velocity_error_median_km_s": float(np.median(errors)),
                f"curve__{split_name}_fractional_error_median": float(np.median(errors / observed)),
                f"curve__{split_name}_gobs_median_m_s2": float(np.median(np.square(observed) * 1.0e6 / (radius * KPC_M))),
            }
            if galaxy == names[0]:
                observed_store = {key: {} for key in observed_values}
            for key, value in observed_values.items():
                observed_store.setdefault(key, {})[galaxy] = value

    for key, values in component_store.items():
        add(
            key,
            values,
            tier="core_preoutcome",
            source="SPARC radius and gas/disk/bulge force components",
            description=key.replace("_", " "),
            pairwise=key.startswith("component__outer_") or key.startswith("component__all_"),
        )
    for key, values in observed_store.items():
        split_name = key.split("__", 1)[1].split("_", 1)[0]
        leakage = split_name in {"outer", "all"} and "error" not in key
        add(
            key,
            values,
            tier="outer_descriptive_leakage" if leakage else "core_preoutcome",
            source="SPARC observed rotation curve",
            description=key.replace("_", " "),
            leakage=leakage,
            pairwise=not leakage and split_name == "inner",
        )

    fit_blocks = {
        "screened": screened_fits.set_index("galaxy"),
        "RAR": baseline_fits[
            (baseline_fits["model"] == "fixed_RAR") & (baseline_fits["scenario"] == "invariant")
        ].set_index("galaxy"),
        "MOND": baseline_fits[
            (baseline_fits["model"] == "simple_MOND") & (baseline_fits["scenario"] == "invariant")
        ].set_index("galaxy"),
        "NFW": baseline_fits[
            (baseline_fits["model"] == "NFW") & (baseline_fits["scenario"] == "invariant")
        ].set_index("galaxy"),
    }
    fit_columns = [
        "objective_inner",
        "evaluations",
        "any_parameter_at_boundary",
        "disk_mass_to_light",
        "bulge_mass_to_light",
        "distance_scale",
        "inclination_adjusted_deg",
        "disk_log_shift",
        "bulge_log_shift",
        "distance_z",
        "inclination_z",
        "nfw_V200_km_s",
        "nfw_concentration",
    ]
    for label, fit in fit_blocks.items():
        for column in fit_columns:
            if column not in fit.columns:
                continue
            raw = fit[column]
            if column == "any_parameter_at_boundary":
                values = raw.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
                kind = "boolean"
            else:
                values = pd.to_numeric(raw, errors="coerce")
                kind = "numeric"
            add(
                f"innerfit__{label}_{column}",
                values,
                tier="mechanistic_inner_fit",
                source=f"{label} inner-radius nuisance fit",
                description=f"{label} {column.replace('_', ' ')}",
                kind=kind,
                pairwise=column in {
                    "objective_inner",
                    "disk_mass_to_light",
                    "bulge_mass_to_light",
                    "distance_scale",
                    "inclination_adjusted_deg",
                },
            )
    for column in [
        "objective_inner",
        "disk_mass_to_light",
        "bulge_mass_to_light",
        "distance_scale",
        "inclination_adjusted_deg",
        "distance_z",
        "inclination_z",
    ]:
        values = pd.to_numeric(fit_blocks["screened"][column], errors="coerce") - pd.to_numeric(
            fit_blocks["RAR"][column], errors="coerce"
        )
        add(
            f"fitshift__screened_minus_RAR_{column}",
            values,
            tier="mechanistic_inner_fit",
            source="difference between independently inner-fitted nuisances",
            description=f"screened minus RAR {column.replace('_', ' ')}",
            pairwise=True,
        )

    selected_points = points[points["model"].isin(
        ["solar_screened_isothermal", "fixed_RAR", "simple_MOND", "NFW"]
    )].copy()
    selected_points["residual"] = (
        selected_points["velocity_predicted_km_s"]
        - selected_points["velocity_observed_adjusted_km_s"]
    )
    for model, label in [
        ("solar_screened_isothermal", "screened"),
        ("fixed_RAR", "RAR"),
        ("simple_MOND", "MOND"),
        ("NFW", "NFW"),
    ]:
        inner = selected_points[
            (selected_points["model"] == model) & (selected_points["split"] == "inner_train")
        ]
        grouped_inner = inner.groupby("galaxy").agg(
            RMSE=("residual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            bias=("residual", "mean"),
        )
        for column in grouped_inner.columns:
            add(
                f"innerfit__{label}_inner_{column}",
                grouped_inner[column],
                tier="mechanistic_inner_fit",
                source=f"{label} inner-radius residual",
                description=f"{label} inner {column}",
                pairwise=True,
            )

    solar = selected_points[selected_points["model"] == "solar_screened_isothermal"].copy()
    solar["tail_v2_km2_s2"] = (
        solar["screened_tail_acceleration_m_s2"]
        * solar["radius_adjusted_kpc"]
        * KPC_M
        / 1.0e6
    )
    solar["total_v2_km2_s2"] = np.square(solar["velocity_predicted_km_s"])
    solar["tail_v2_fraction"] = solar["tail_v2_km2_s2"] / solar["total_v2_km2_s2"]
    solar["delta_velocity_vs_RAR_same_nuisance"] = (
        solar["velocity_predicted_km_s"] - solar["velocity_RAR_same_nuisance_km_s"]
    )
    for split_name in ["inner_train", "outer_holdout"]:
        block = solar[solar["split"] == split_name]
        grouped_solar = block.groupby("galaxy").agg(
            source_baryonic_mass_solar=("source_baryonic_mass_solar", "median"),
            screen_median=("screened_tail_factor", "median"),
            screen_min=("screened_tail_factor", "min"),
            gbar_median_m_s2=("g_bar_m_s2", "median"),
            gbar_min_m_s2=("g_bar_m_s2", "min"),
            tail_acceleration_median_m_s2=("screened_tail_acceleration_m_s2", "median"),
            tail_v2_fraction_median=("tail_v2_fraction", "median"),
            tail_v2_fraction_max=("tail_v2_fraction", "max"),
            predicted_delta_vs_RAR_same_nuisance_median_km_s=(
                "delta_velocity_vs_RAR_same_nuisance", "median"
            ),
        )
        for column in grouped_solar.columns:
            add(
                f"tail__{split_name}_{column}",
                grouped_solar[column],
                tier="mechanistic_inner_fit",
                source="screened-law prediction using inner-fitted nuisances",
                description=f"{split_name} {column.replace('_', ' ')}",
                pairwise=True,
            )

    solar_outer = solar[solar["split"] == "outer_holdout"][
        ["galaxy", "radius_catalog_kpc", "velocity_predicted_km_s"]
    ].rename(columns={"velocity_predicted_km_s": "screened_prediction"})
    rar_outer = selected_points[
        (selected_points["model"] == "fixed_RAR")
        & (selected_points["split"] == "outer_holdout")
    ][["galaxy", "radius_catalog_kpc", "velocity_predicted_km_s"]].rename(
        columns={"velocity_predicted_km_s": "RAR_prediction"}
    )
    predicted_gap = solar_outer.merge(
        rar_outer, on=["galaxy", "radius_catalog_kpc"], validate="one_to_one"
    )
    predicted_gap["gap"] = predicted_gap["screened_prediction"] - predicted_gap["RAR_prediction"]
    gap_summary = predicted_gap.groupby("galaxy").agg(
        median_predicted_gap_screened_minus_RAR_km_s=("gap", "median"),
        minimum_predicted_gap_screened_minus_RAR_km_s=("gap", "min"),
        maximum_predicted_gap_screened_minus_RAR_km_s=("gap", "max"),
    )
    for column in gap_summary.columns:
        add(
            f"mechanism__{column}",
            gap_summary[column],
            tier="mechanistic_inner_fit",
            source="outer predictions from independently inner-fitted models; no outer observed speed",
            description=column.replace("_", " "),
            pairwise=True,
        )

    for raw_name in [
        "catalog__baryonic_mass_solar",
        "catalog__stellar_mass_solar",
        "catalog__gas_mass_solar",
        "catalog__luminosity_3p6_billion_solar",
        "catalog__disk_central_surface_brightness",
        "catalog__effective_surface_brightness",
    ]:
        if raw_name in features:
            values = pd.to_numeric(features[raw_name], errors="coerce")
            transformed = np.log10(values.where(values > 0.0))
            tier = next(row["tier"] for row in manifest_rows if row["feature"] == raw_name)
            add(
                f"transform__log10_{raw_name.removeprefix('catalog__')}",
                transformed.to_numpy(),
                tier=tier,
                source="deterministic log10 transform",
                description=f"log10 of {raw_name}",
                leakage=False,
                pairwise=True,
            )

    manifest = pd.DataFrame(manifest_rows)
    for row in list(manifest_rows):
        if row["kind"] == "categorical":
            column = row["feature"]
            values = features[column].fillna("missing").astype(str)
            for level, count in values.value_counts().items():
                if count < 5:
                    continue
                safe_level = "".join(char if char.isalnum() else "_" for char in level).strip("_")
                add(
                    f"onehot__{column.removeprefix('category__')}__{safe_level}",
                    (values == level).astype(float).to_numpy(),
                    tier=row["tier"],
                    source=f"one-hot encoding of {column}",
                    description=f"{column} equals {level}",
                    kind="boolean",
                    leakage=bool(row["uses_outer_observed_velocity"]),
                )
    manifest = pd.DataFrame(manifest_rows)
    for row in list(manifest_rows):
        if row["kind"] == "categorical":
            continue
        missing = pd.to_numeric(features[row["feature"]], errors="coerce").isna()
        if 0.05 <= missing.mean() <= 0.80:
            add(
                f"missing__{row['feature']}",
                missing.astype(float).to_numpy(),
                tier=row["tier"],
                source=f"missingness indicator for {row['feature']}",
                description=f"whether {row['feature']} is missing",
                kind="boolean",
                leakage=bool(row["uses_outer_observed_velocity"]),
            )

    manifest = pd.DataFrame(manifest_rows)
    for column in features.columns:
        if column == "galaxy":
            continue
        record = manifest.loc[manifest["feature"] == column].iloc[0]
        if record["kind"] == "categorical":
            nonmissing = features[column].notna()
            unique = int(features.loc[nonmissing, column].nunique())
        else:
            numeric = pd.to_numeric(features[column], errors="coerce")
            features[column] = numeric
            nonmissing = numeric.notna()
            unique = int(numeric.loc[nonmissing].nunique())
        manifest.loc[manifest["feature"] == column, "nonmissing"] = int(nonmissing.sum())
        manifest.loc[manifest["feature"] == column, "missing_fraction"] = float(1.0 - nonmissing.mean())
        manifest.loc[manifest["feature"] == column, "unique_values"] = unique
    return features, manifest


def univariate_tests(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    minimum_nonmissing: int,
    minimum_unique: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = outcomes.merge(features, on="galaxy", validate="one_to_one")
    binary = joined["primary_improver"].astype(bool).to_numpy()
    skill = joined["continuous_skill"].to_numpy(dtype=float)
    numeric_rows = []
    categorical_rows = []
    for record in manifest.itertuples(index=False):
        column = record.feature
        if record.kind == "categorical":
            values = joined[column].fillna("missing").astype(str)
            table = pd.crosstab(values, joined["primary_improver"])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            if table.shape == (2, 2):
                _, p_value = fisher_exact(table.to_numpy(), alternative="two-sided")
                test = "Fisher exact"
            else:
                _, p_value, _, _ = chi2_contingency(table.to_numpy(), correction=False)
                test = "chi-square"
            n = int(table.to_numpy().sum())
            phi2 = max(0.0, float(chi2_contingency(table.to_numpy(), correction=False)[0]) / n)
            rows, columns = table.shape
            corrected_phi2 = max(0.0, phi2 - (columns - 1) * (rows - 1) / max(n - 1, 1))
            corrected_rows = rows - (rows - 1) ** 2 / max(n - 1, 1)
            corrected_columns = columns - (columns - 1) ** 2 / max(n - 1, 1)
            denom = min(corrected_columns - 1, corrected_rows - 1)
            cramer = math.sqrt(corrected_phi2 / denom) if denom > 0.0 else 0.0
            level_summary = {
                level: {
                    "n": int(count),
                    "improvers": int(
                        joined.loc[values == level, "primary_improver"].sum()
                    ),
                    "improvement_rate": float(
                        joined.loc[values == level, "primary_improver"].mean()
                    ),
                }
                for level, count in values.value_counts().sort_index().items()
            }
            categorical_rows.append(
                {
                    "feature": column,
                    "tier": record.tier,
                    "uses_outer_observed_velocity": record.uses_outer_observed_velocity,
                    "test": test,
                    "p_value": float(p_value),
                    "bias_corrected_Cramer_V": cramer,
                    "levels": int(rows),
                    "level_summary_json": json.dumps(level_summary, sort_keys=True),
                }
            )
            continue

        values = pd.to_numeric(joined[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(skill)
        if valid.sum() < minimum_nonmissing or np.unique(values[valid]).size < minimum_unique:
            continue
        improve_values = values[valid & binary]
        worse_values = values[valid & ~binary]
        if len(improve_values) < 5 or len(worse_values) < 5:
            continue
        rho, spearman_p = spearmanr(values[valid], skill[valid])
        u_stat, mann_p = mannwhitneyu(
            improve_values, worse_values, alternative="two-sided", method="auto"
        )
        pooled_n = len(improve_values) + len(worse_values) - 2
        pooled_variance = (
            (len(improve_values) - 1) * np.var(improve_values, ddof=1)
            + (len(worse_values) - 1) * np.var(worse_values, ddof=1)
        ) / pooled_n
        if pooled_variance > 0.0:
            cohens_d = (np.mean(improve_values) - np.mean(worse_values)) / math.sqrt(
                pooled_variance
            )
            correction = 1.0 - 3.0 / (4.0 * pooled_n - 1.0)
            hedges_g = correction * cohens_d
        else:
            hedges_g = 0.0
        cliffs_delta = 2.0 * float(u_stat) / (len(improve_values) * len(worse_values)) - 1.0
        raw_auc = roc_auc_score(binary[valid].astype(int), values[valid])
        direction = "higher_in_improvers" if raw_auc >= 0.5 else "lower_in_improvers"
        numeric_rows.append(
            {
                "feature": column,
                "tier": record.tier,
                "source": record.source,
                "description": record.description,
                "uses_outer_observed_velocity": record.uses_outer_observed_velocity,
                "nonmissing": int(valid.sum()),
                "improver_nonmissing": int(len(improve_values)),
                "worsener_nonmissing": int(len(worse_values)),
                "improver_mean": float(np.mean(improve_values)),
                "worsener_mean": float(np.mean(worse_values)),
                "improver_median": float(np.median(improve_values)),
                "worsener_median": float(np.median(worse_values)),
                "spearman_rho_with_skill": float(rho),
                "spearman_p": float(spearman_p),
                "mann_whitney_p": float(mann_p),
                "hedges_g_improver_minus_worsener": float(hedges_g),
                "cliffs_delta": cliffs_delta,
                "univariate_AUC_direction_free": float(max(raw_auc, 1.0 - raw_auc)),
                "direction": direction,
            }
        )
    numeric_frame = pd.DataFrame(numeric_rows)
    if not numeric_frame.empty:
        numeric_frame["spearman_q_within_all_numeric"] = fdr_bh(numeric_frame["spearman_p"])
        numeric_frame["mann_whitney_q_within_all_numeric"] = fdr_bh(
            numeric_frame["mann_whitney_p"]
        )
        for tier, index in numeric_frame.groupby("tier").groups.items():
            numeric_frame.loc[index, "spearman_q_within_tier"] = fdr_bh(
                numeric_frame.loc[index, "spearman_p"]
            )
            numeric_frame.loc[index, "mann_whitney_q_within_tier"] = fdr_bh(
                numeric_frame.loc[index, "mann_whitney_p"]
            )
        numeric_frame = numeric_frame.sort_values(
            ["uses_outer_observed_velocity", "mann_whitney_q_within_all_numeric", "mann_whitney_p"]
        )
    categorical_frame = pd.DataFrame(categorical_rows)
    if not categorical_frame.empty:
        categorical_frame["q_within_all_categorical"] = fdr_bh(categorical_frame["p_value"])
        for tier, index in categorical_frame.groupby("tier").groups.items():
            categorical_frame.loc[index, "q_within_tier"] = fdr_bh(
                categorical_frame.loc[index, "p_value"]
            )
        categorical_frame = categorical_frame.sort_values(
            ["uses_outer_observed_velocity", "q_within_all_categorical", "p_value"]
        )
    return numeric_frame, categorical_frame


def feature_sets(manifest: pd.DataFrame) -> dict[str, list[str]]:
    eligible = manifest[
        (manifest["kind"] != "categorical")
        & (manifest["nonmissing"] >= 30)
        & (manifest["unique_values"] >= 2)
        & (manifest["missing_fraction"] <= 0.70)
    ].copy()
    tiers = eligible.set_index("feature")["tier"]
    core = tiers[tiers == "core_preoutcome"].index.tolist()
    extended = tiers[tiers == "extended_environment"].index.tolist()
    mechanism = tiers[tiers == "mechanistic_inner_fit"].index.tolist()
    leakage = tiers[tiers == "outer_descriptive_leakage"].index.tolist()
    core_environment = [
        name for name in core if "environment_" in name or name.startswith("coordinate__")
    ]
    environment_only = list(dict.fromkeys(core_environment + extended))
    intrinsic = [
        name
        for name in core
        if "environment_" not in name
        and not name.startswith("coordinate__")
        and not name.startswith("reference__")
    ]
    return {
        "intrinsic_catalog_and_curve": intrinsic,
        "environment_only": environment_only,
        "mechanistic_inner_fit_only": mechanism,
        "intrinsic_plus_mechanistic": intrinsic + mechanism,
        "core_preoutcome": core,
        "core_plus_extended_environment": core + extended,
        "core_plus_mechanistic": core + mechanism,
        "all_preoutcome": core + extended + mechanism,
        "all_plus_outer_descriptive_leakage": core + extended + mechanism + leakage,
    }


def model_pipelines(config: dict, *, classification: bool) -> dict:
    if classification:
        logistic = config["classification_models"]["logistic_L2"]
        forest = config["classification_models"]["random_forest"]
        return {
            "logistic_L2": make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(
                    C=float(logistic["C"]),
                    class_weight=logistic["class_weight"],
                    max_iter=int(logistic["max_iter"]),
                    solver="liblinear",
                    random_state=int(config["seed"]),
                ),
            ),
            "random_forest": make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                RandomForestClassifier(
                    n_estimators=int(forest["trees"]),
                    max_depth=int(forest["max_depth"]),
                    min_samples_leaf=int(forest["min_samples_leaf"]),
                    class_weight=forest["class_weight"],
                    random_state=int(config["seed"]),
                    n_jobs=-1,
                ),
            ),
        }
    ridge = config["regression_models"]["ridge"]
    forest = config["regression_models"]["random_forest"]
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=float(ridge["alpha"])),
        ),
        "random_forest_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestRegressor(
                n_estimators=int(forest["trees"]),
                max_depth=int(forest["max_depth"]),
                min_samples_leaf=int(forest["min_samples_leaf"]),
                random_state=int(config["seed"]),
                n_jobs=-1,
            ),
        ),
    }


def cross_validated_models(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    sets: dict[str, list[str]],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = outcomes.merge(features, on="galaxy", validate="one_to_one")
    y_binary = joined["primary_improver"].astype(int).to_numpy()
    y_skill = joined["continuous_skill"].to_numpy(dtype=float)
    folds = int(config["folds"])
    repeats = int(config["repeats"])
    seed = int(config["seed"])
    classification_models = model_pipelines(config, classification=True)
    regression_models = model_pipelines(config, classification=False)
    score_rows = []
    oof_rows = []
    aggregate_rows = []
    for set_index, (set_name, columns) in enumerate(sets.items()):
        X = joined[columns].to_numpy(dtype=float)
        for model_index, (model_name, estimator) in enumerate(classification_models.items()):
            prediction_sum = np.zeros(len(joined), dtype=float)
            prediction_count = np.zeros(len(joined), dtype=int)
            for repeat in range(repeats):
                splitter = StratifiedKFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=seed + 1009 * repeat + 101 * set_index + model_index,
                )
                repeat_prediction = np.full(len(joined), np.nan)
                for train, test in splitter.split(X, y_binary):
                    estimator.fit(X[train], y_binary[train])
                    repeat_prediction[test] = estimator.predict_proba(X[test])[:, 1]
                prediction_sum += repeat_prediction
                prediction_count += 1
                hard = repeat_prediction >= 0.5
                score_rows.append(
                    {
                        "task": "classification",
                        "feature_set": set_name,
                        "features": len(columns),
                        "model": model_name,
                        "repeat": repeat,
                        "ROC_AUC": float(roc_auc_score(y_binary, repeat_prediction)),
                        "average_precision": float(
                            average_precision_score(y_binary, repeat_prediction)
                        ),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(y_binary, hard)
                        ),
                        "Brier_score": float(brier_score_loss(y_binary, repeat_prediction)),
                    }
                )
            aggregate = prediction_sum / prediction_count
            aggregate_rows.append(
                {
                    "task": "classification",
                    "feature_set": set_name,
                    "features": len(columns),
                    "model": model_name,
                    "aggregate_repeated_OOF_ROC_AUC": float(roc_auc_score(y_binary, aggregate)),
                    "aggregate_repeated_OOF_average_precision": float(
                        average_precision_score(y_binary, aggregate)
                    ),
                    "aggregate_repeated_OOF_balanced_accuracy": float(
                        balanced_accuracy_score(y_binary, aggregate >= 0.5)
                    ),
                    "aggregate_repeated_OOF_Brier_score": float(
                        brier_score_loss(y_binary, aggregate)
                    ),
                }
            )
            for galaxy, observed, probability in zip(
                joined["galaxy"], y_binary, aggregate, strict=True
            ):
                oof_rows.append(
                    {
                        "task": "classification",
                        "feature_set": set_name,
                        "model": model_name,
                        "galaxy": galaxy,
                        "observed_improver": observed,
                        "OOF_probability_improver": probability,
                    }
                )

        for model_index, (model_name, estimator) in enumerate(regression_models.items()):
            prediction_sum = np.zeros(len(joined), dtype=float)
            prediction_count = np.zeros(len(joined), dtype=int)
            for repeat in range(repeats):
                splitter = KFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=seed + 2003 * repeat + 107 * set_index + model_index,
                )
                repeat_prediction = np.full(len(joined), np.nan)
                for train, test in splitter.split(X):
                    estimator.fit(X[train], y_skill[train])
                    repeat_prediction[test] = estimator.predict(X[test])
                prediction_sum += repeat_prediction
                prediction_count += 1
                rho = spearmanr(y_skill, repeat_prediction).statistic
                score_rows.append(
                    {
                        "task": "regression",
                        "feature_set": set_name,
                        "features": len(columns),
                        "model": model_name,
                        "repeat": repeat,
                        "R2": float(r2_score(y_skill, repeat_prediction)),
                        "Spearman": float(rho),
                        "MAE": float(mean_absolute_error(y_skill, repeat_prediction)),
                    }
                )
            aggregate = prediction_sum / prediction_count
            aggregate_rows.append(
                {
                    "task": "regression",
                    "feature_set": set_name,
                    "features": len(columns),
                    "model": model_name,
                    "aggregate_repeated_OOF_R2": float(r2_score(y_skill, aggregate)),
                    "aggregate_repeated_OOF_Spearman": float(
                        spearmanr(y_skill, aggregate).statistic
                    ),
                    "aggregate_repeated_OOF_MAE": float(
                        mean_absolute_error(y_skill, aggregate)
                    ),
                }
            )
            for galaxy, observed, prediction in zip(
                joined["galaxy"], y_skill, aggregate, strict=True
            ):
                oof_rows.append(
                    {
                        "task": "regression",
                        "feature_set": set_name,
                        "model": model_name,
                        "galaxy": galaxy,
                        "observed_skill": observed,
                        "OOF_predicted_skill": prediction,
                    }
                )
    return pd.DataFrame(score_rows), pd.DataFrame(aggregate_rows), pd.DataFrame(oof_rows)


def fixed_logistic_permutation_test(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    columns: list[str],
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    joined = outcomes.merge(features, on="galaxy", validate="one_to_one")
    X = joined[columns].to_numpy(dtype=float)
    y = joined["primary_improver"].astype(int).to_numpy()
    pipeline = model_pipelines(config, classification=True)["logistic_L2"]
    permutation_config = config["fixed_primary_permutation_test"]
    permutations = int(permutation_config["permutations"])
    repeats = int(permutation_config.get("permutation_repeats", 5))
    folds = int(config["folds"])
    seed = int(config["seed"])

    def mean_auc(labels, offset: int) -> float:
        aucs = []
        for repeat in range(repeats):
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed + 7919 * repeat + offset,
            )
            prediction = np.full(len(labels), np.nan)
            for train, test in splitter.split(X, labels):
                pipeline.fit(X[train], labels[train])
                prediction[test] = pipeline.predict_proba(X[test])[:, 1]
            aucs.append(roc_auc_score(labels, prediction))
        return float(np.mean(aucs))

    observed = mean_auc(y, 0)
    rng = np.random.default_rng(seed + 99991)
    null = []
    for index in range(permutations):
        permuted = rng.permutation(y)
        null.append(mean_auc(permuted, index + 1))
    null_array = np.asarray(null)
    p_value = float((1 + np.sum(null_array >= observed)) / (permutations + 1))
    report = {
        "feature_set": "core_preoutcome",
        "model": "logistic_L2",
        "features": len(columns),
        "folds": folds,
        "repeats": repeats,
        "permutations": permutations,
        "observed_mean_ROC_AUC": observed,
        "permutation_p_one_sided": p_value,
        "null_mean": float(np.mean(null_array)),
        "null_standard_deviation": float(np.std(null_array, ddof=1)),
        "null_quantiles": list(map(float, np.percentile(null_array, [2.5, 50.0, 97.5]))),
    }
    return report, pd.DataFrame({"permutation": np.arange(permutations), "mean_ROC_AUC": null_array})


def subgroup_stat(mask: np.ndarray, y: np.ndarray, skill: np.ndarray) -> dict:
    inside = np.asarray(mask, dtype=bool)
    outside = ~inside
    table = np.asarray(
        [
            [np.sum(y[inside]), np.sum(~y[inside])],
            [np.sum(y[outside]), np.sum(~y[outside])],
        ]
    )
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    return {
        "subgroup_n": int(np.sum(inside)),
        "complement_n": int(np.sum(outside)),
        "subgroup_improvers": int(np.sum(y[inside])),
        "complement_improvers": int(np.sum(y[outside])),
        "subgroup_improvement_rate": float(np.mean(y[inside])),
        "complement_improvement_rate": float(np.mean(y[outside])),
        "risk_difference": float(np.mean(y[inside]) - np.mean(y[outside])),
        "odds_ratio": float(odds_ratio),
        "fisher_p": float(p_value),
        "subgroup_mean_skill": float(np.mean(skill[inside])),
        "complement_mean_skill": float(np.mean(skill[outside])),
        "skill_difference": float(np.mean(skill[inside]) - np.mean(skill[outside])),
    }


def one_feature_subgroups(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    minimum_group: int,
) -> pd.DataFrame:
    joined = outcomes.merge(features, on="galaxy", validate="one_to_one")
    y = joined["primary_improver"].astype(bool).to_numpy()
    skill = joined["continuous_skill"].to_numpy(dtype=float)
    rows = []
    for record in manifest.itertuples(index=False):
        column = record.feature
        if record.kind == "categorical":
            values = joined[column].fillna("missing").astype(str)
            for level in sorted(values.unique()):
                mask = (values == level).to_numpy()
                if mask.sum() < minimum_group or (~mask).sum() < minimum_group:
                    continue
                rows.append(
                    {
                        "feature": column,
                        "tier": record.tier,
                        "uses_outer_observed_velocity": record.uses_outer_observed_velocity,
                        "rule": f"{column} == {level}",
                        "operator": "equals",
                        "threshold_or_level": level,
                        **subgroup_stat(mask, y, skill),
                    }
                )
            continue
        values = pd.to_numeric(joined[column], errors="coerce").to_numpy(dtype=float)
        valid_values = np.unique(values[np.isfinite(values)])
        if len(valid_values) < 2:
            continue
        thresholds = (valid_values[:-1] + valid_values[1:]) / 2.0
        for threshold in thresholds:
            for operator, mask in [
                ("<=", np.isfinite(values) & (values <= threshold)),
                (">", np.isfinite(values) & (values > threshold)),
            ]:
                if mask.sum() < minimum_group or (~mask).sum() < minimum_group:
                    continue
                rows.append(
                    {
                        "feature": column,
                        "tier": record.tier,
                        "uses_outer_observed_velocity": record.uses_outer_observed_velocity,
                        "rule": f"{column} {operator} {threshold:.10g}",
                        "operator": operator,
                        "threshold_or_level": threshold,
                        **subgroup_stat(mask, y, skill),
                    }
                )
    result = pd.DataFrame(rows)
    result["fisher_q_all_one_feature_rules"] = fdr_bh(result["fisher_p"])
    for tier, index in result.groupby("tier").groups.items():
        result.loc[index, "fisher_q_within_tier"] = fdr_bh(result.loc[index, "fisher_p"])
    return result.sort_values(
        ["uses_outer_observed_velocity", "fisher_q_all_one_feature_rules", "fisher_p"]
    )


def pairwise_subgroups(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    minimum_group: int,
) -> pd.DataFrame:
    joined = outcomes.merge(features, on="galaxy", validate="one_to_one")
    y = joined["primary_improver"].astype(bool).to_numpy()
    skill = joined["continuous_skill"].to_numpy(dtype=float)
    candidates = manifest[
        manifest["pairwise_candidate"].astype(bool)
        & ~manifest["uses_outer_observed_velocity"].astype(bool)
        & manifest["tier"].isin(["core_preoutcome", "mechanistic_inner_fit"])
        & (manifest["kind"] != "categorical")
        & (manifest["nonmissing"] >= 60)
        & (manifest["unique_values"] >= 2)
    ]
    rules = []
    seen = set()
    for record in candidates.itertuples(index=False):
        values = pd.to_numeric(joined[record.feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if record.kind == "boolean" or np.unique(values[finite]).size == 2:
            masks = [("true", finite & (values > 0.5))]
        else:
            quantiles = np.unique(np.quantile(values[finite], [0.25, 0.5, 0.75]))
            masks = []
            for threshold in quantiles:
                masks.extend(
                    [
                        (f"<= {threshold:.8g}", finite & (values <= threshold)),
                        (f"> {threshold:.8g}", finite & (values > threshold)),
                    ]
                )
        for label, mask in masks:
            if mask.sum() < minimum_group or (~mask).sum() < minimum_group:
                continue
            key = mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            rules.append(
                {
                    "feature": record.feature,
                    "label": label,
                    "rule": f"{record.feature} {label}",
                    "mask": mask,
                }
            )
    rows = []
    for first, second in itertools.combinations(rules, 2):
        if first["feature"] == second["feature"]:
            continue
        mask = first["mask"] & second["mask"]
        if mask.sum() < minimum_group or (~mask).sum() < minimum_group:
            continue
        rows.append(
            {
                "feature_1": first["feature"],
                "feature_2": second["feature"],
                "rule": f"({first['rule']}) AND ({second['rule']})",
                **subgroup_stat(mask, y, skill),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fisher_q_all_pairwise_rules"] = fdr_bh(result["fisher_p"])
    return result.sort_values(["fisher_q_all_pairwise_rules", "fisher_p"])


def bootstrap_top_rule_intervals(
    rules: pd.DataFrame,
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    if rules.empty:
        return rules
    joined = outcomes.merge(features, on="galaxy", validate="one_to_one")
    y = joined["primary_improver"].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    top = rules.head(25).copy()
    for index, row in top.iterrows():
        feature = row["feature"]
        if row["operator"] == "equals":
            mask = joined[feature].fillna("missing").astype(str).to_numpy() == str(
                row["threshold_or_level"]
            )
        else:
            values = pd.to_numeric(joined[feature], errors="coerce").to_numpy(dtype=float)
            threshold = float(row["threshold_or_level"])
            mask = np.isfinite(values) & (
                values <= threshold if row["operator"] == "<=" else values > threshold
            )
        bootstrap = []
        for _ in range(draws):
            sampled_inside = rng.choice(np.flatnonzero(mask), size=int(mask.sum()), replace=True)
            sampled_outside = rng.choice(np.flatnonzero(~mask), size=int((~mask).sum()), replace=True)
            bootstrap.append(float(np.mean(y[sampled_inside]) - np.mean(y[sampled_outside])))
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        top.loc[index, "risk_difference_bootstrap_95_low"] = low
        top.loc[index, "risk_difference_bootstrap_95_high"] = high
    return top


def robustness_summary(outcomes: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = outcomes.merge(features, on="galaxy", validate="one_to_one")
    primary = joined["primary_improver"].astype(bool)
    definitions = [
        "primary_improver",
        "improver_vs_MOND",
        "improver_by_chi2",
        "improver_in_catalog_space",
        "tail_helps_at_same_screened_nuisance",
        "improvement_at_least_10pct",
        "improvement_at_least_20pct",
    ]
    rows = []
    for column in definitions:
        values = joined[column].astype(bool)
        intersection = int(np.sum(values & primary))
        union = int(np.sum(values | primary))
        rows.append(
            {
                "scenario": column,
                "galaxies": len(values),
                "wins": int(values.sum()),
                "win_rate": float(values.mean()),
                "agreement_with_primary": float(np.mean(values == primary)),
                "Jaccard_with_primary": float(intersection / union) if union else 1.0,
            }
        )
    subset_rules = {
        "quality_1_only": joined["catalog__quality"].to_numpy(dtype=float) == 1.0,
        "screened_fit_not_at_boundary": joined[
            "innerfit__screened_any_parameter_at_boundary"
        ].to_numpy(dtype=float) < 0.5,
        "RAR_fit_not_at_boundary": joined[
            "innerfit__RAR_any_parameter_at_boundary"
        ].to_numpy(dtype=float) < 0.5,
        "both_fits_not_at_boundary": (
            joined["innerfit__screened_any_parameter_at_boundary"].to_numpy(dtype=float) < 0.5
        )
        & (joined["innerfit__RAR_any_parameter_at_boundary"].to_numpy(dtype=float) < 0.5),
    }
    for name, mask in subset_rules.items():
        rows.append(
            {
                "scenario": name,
                "galaxies": int(mask.sum()),
                "wins": int(primary[mask].sum()),
                "win_rate": float(primary[mask].mean()),
                "mean_continuous_skill": float(joined.loc[mask, "continuous_skill"].mean()),
            }
        )

    full_mean = float(joined["continuous_skill"].mean())
    influence = []
    for index, row in joined.iterrows():
        retained = joined.index != index
        leave_mean = float(joined.loc[retained, "continuous_skill"].mean())
        influence.append(
            {
                "galaxy": row["galaxy"],
                "primary_improver": bool(row["primary_improver"]),
                "continuous_skill": float(row["continuous_skill"]),
                "full_sample_mean_skill": full_mean,
                "leave_one_out_mean_skill": leave_mean,
                "change_in_mean_when_removed": leave_mean - full_mean,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(influence).sort_values(
        "change_in_mean_when_removed", key=lambda values: np.abs(values), ascending=False
    )


def plot_results(
    joined: pd.DataFrame,
    univariate: pd.DataFrame,
    cv_aggregate: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = np.where(joined["primary_improver"], "#059669", "#dc2626")
    axes[0, 0].scatter(
        np.log10(joined["catalog__baryonic_mass_solar"]),
        joined["continuous_skill"],
        c=colors,
        alpha=0.78,
        s=32,
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set(
        xlabel="log10 baryonic mass (solar masses)",
        ylabel="Screened-over-RAR improvement skill (positive is better)",
        title="Mass and individual-galaxy outcome",
    )
    axes[0, 1].scatter(
        joined["outer_mean_residual_actual_RAR"],
        joined["continuous_skill"],
        c=colors,
        alpha=0.78,
        s=32,
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].axvline(0.0, color="black", linewidth=1)
    axes[0, 1].set(
        xlabel="RAR outer mean residual (km/s; outer-outcome leakage)",
        ylabel="Screened-over-RAR improvement skill",
        title="Descriptive reason for wins",
    )

    top = univariate[~univariate["uses_outer_observed_velocity"].astype(bool)].copy()
    top = top.nsmallest(12, "mann_whitney_q_within_all_numeric").sort_values(
        "hedges_g_improver_minus_worsener"
    )
    axes[1, 0].barh(
        np.arange(len(top)),
        top["hedges_g_improver_minus_worsener"],
        color=np.where(top["hedges_g_improver_minus_worsener"] >= 0.0, "#2563eb", "#f59e0b"),
    )
    axes[1, 0].set_yticks(
        np.arange(len(top)), [name.replace("__", ": ")[-48:] for name in top["feature"]]
    )
    axes[1, 0].axvline(0.0, color="black", linewidth=1)
    axes[1, 0].set(xlabel="Hedges g (improvers minus worseners)", title="Top target-independent contrasts")

    cv = cv_aggregate[cv_aggregate["task"] == "classification"].copy()
    labels = cv["feature_set"] + "\n" + cv["model"]
    axes[1, 1].bar(
        np.arange(len(cv)),
        cv["aggregate_repeated_OOF_ROC_AUC"],
        color=["#7c3aed" if "outer" in name else "#64748b" for name in cv["feature_set"]],
    )
    axes[1, 1].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set_xticks(np.arange(len(cv)), labels, rotation=55, ha="right", fontsize=8)
    axes[1, 1].set(
        ylabel="Repeated out-of-fold ROC AUC",
        title="Can features predict which galaxies win?",
        ylim=(0.35, 1.0),
    )
    figure.suptitle("Forensic analysis of the 41 screened-law SPARC wins")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "solar_screened_improver_forensics_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "solar_screened_improver_forensics",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    base_protocol_path = ROOT / protocol["sample"]["base_protocol"]
    base_protocol = json.loads(base_protocol_path.read_text(encoding="utf-8"))
    morphology_path = ROOT / "data" / "derived" / "nbp0_sparc_morphology.csv"
    points_path = ROOT / "results" / "solar_screened_galaxy_morphology" / "point_predictions.csv"
    assignments_path = ROOT / "results" / "solar_screened_galaxy_morphology" / "morphology_assignments.csv"
    screened_fits_path = ROOT / "results" / "solar_screened_galaxy_morphology" / "screened_tail_galaxy_fits.csv"
    baseline_fits_path = ROOT / "results" / "sparc_independent_nuisance_refit" / "galaxy_fits.csv"
    sparc = ROOT / "data" / "raw" / "sparc"
    cf4_path = ROOT / "data" / "derived" / "void_scores_cf4.csv"
    void_wall_path = ROOT / "data" / "derived" / "void_wall_scores_local.csv"
    void_cage_path = ROOT / "data" / "derived" / "void_cage_geometry.csv"

    print("loading frozen sample and constructing outcomes", flush=True)
    points = pd.read_csv(points_path)
    outcomes = compute_outcomes(points)
    if len(outcomes) != int(protocol["sample"]["galaxies"]):
        raise ValueError("outcome sample does not match protocol")
    if int(outcomes["primary_improver"].sum()) != 41:
        raise ValueError("primary 41/90 split changed")

    frame = build_frame(base_protocol, sparc, morphology_path)
    print("building complete local feature inventory", flush=True)
    features, manifest = build_features(
        frame,
        points,
        pd.read_csv(morphology_path),
        pd.read_csv(assignments_path),
        parse_table1_extended(sparc / "table1.dat"),
        pd.read_csv(sparc / "coordinates.csv"),
        pd.read_csv(cf4_path),
        pd.read_csv(void_wall_path),
        pd.read_csv(void_cage_path),
        pd.read_csv(screened_fits_path),
        pd.read_csv(baseline_fits_path),
    )
    joined = outcomes.merge(features, on="galaxy", validate="one_to_one")
    print(f"features={len(manifest)}", flush=True)

    tests = protocol["univariate_tests"]
    univariate, categorical = univariate_tests(
        features,
        manifest,
        outcomes,
        minimum_nonmissing=int(tests["minimum_nonmissing_per_feature"]),
        minimum_unique=int(tests["minimum_unique_values"]),
    )
    sets = feature_sets(manifest)
    print(
        "feature sets " + ", ".join(f"{name}={len(columns)}" for name, columns in sets.items()),
        flush=True,
    )
    cv_scores, cv_aggregate, oof = cross_validated_models(
        features, outcomes, sets, protocol["cross_validation"]
    )
    print("running fixed core logistic permutation test", flush=True)
    permutation_report, permutation_frame = fixed_logistic_permutation_test(
        features,
        outcomes,
        sets["core_preoutcome"],
        protocol["cross_validation"],
    )

    subgroup_config = protocol["subgroup_scan"]
    print("scanning every eligible one-feature threshold", flush=True)
    one_rules = one_feature_subgroups(
        features,
        manifest,
        outcomes,
        minimum_group=int(subgroup_config["minimum_one_feature_group"]),
    )
    top_rule_intervals = bootstrap_top_rule_intervals(
        one_rules,
        features,
        outcomes,
        draws=int(subgroup_config["bootstrap_draws_for_top_rules"]),
        seed=int(protocol["cross_validation"]["seed"]) + 17,
    )
    print("scanning pairwise conjunction scenarios", flush=True)
    pair_rules = pairwise_subgroups(
        features,
        manifest,
        outcomes,
        minimum_group=int(subgroup_config["minimum_pairwise_group"]),
    )
    robustness, influence = robustness_summary(outcomes, features)

    improvers = joined[joined["primary_improver"]].copy().sort_values(
        "continuous_skill", ascending=False
    )
    improvers.insert(1, "improvement_rank", np.arange(1, len(improvers) + 1))
    improvers["outer_RMSE_screened_km_s"] = np.sqrt(improvers["outer_MSE_screened"])
    improvers["outer_RMSE_RAR_km_s"] = np.sqrt(improvers["outer_MSE_RAR"])

    target_independent = univariate[
        ~univariate["uses_outer_observed_velocity"].astype(bool)
    ].copy()
    leakage = univariate[univariate["uses_outer_observed_velocity"].astype(bool)].copy()
    significant_target_independent = target_independent[
        (target_independent["mann_whitney_q_within_all_numeric"] <= 0.05)
        | (target_independent["spearman_q_within_all_numeric"] <= 0.05)
    ]
    significant_one_rules = one_rules[
        (~one_rules["uses_outer_observed_velocity"].astype(bool))
        & (one_rules["fisher_q_all_one_feature_rules"] <= 0.05)
    ]
    significant_pair_rules = (
        pair_rules[pair_rules["fisher_q_all_pairwise_rules"] <= 0.05]
        if not pair_rules.empty
        else pair_rules
    )
    cv_classification = cv_aggregate[cv_aggregate["task"] == "classification"].copy()
    best_preoutcome = cv_classification[
        cv_classification["feature_set"] != "all_plus_outer_descriptive_leakage"
    ].sort_values("aggregate_repeated_OOF_ROC_AUC", ascending=False).iloc[0]
    best_with_leakage = cv_classification[
        cv_classification["feature_set"] == "all_plus_outer_descriptive_leakage"
    ].sort_values(
        "aggregate_repeated_OOF_ROC_AUC", ascending=False
    ).iloc[0]
    cv_repeat_summary = (
        cv_scores.groupby(["task", "feature_set", "features", "model"], dropna=False)
        .agg(
            ROC_AUC_mean=("ROC_AUC", "mean"),
            ROC_AUC_std=("ROC_AUC", "std"),
            ROC_AUC_min=("ROC_AUC", "min"),
            ROC_AUC_max=("ROC_AUC", "max"),
            average_precision_mean=("average_precision", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            Brier_score_mean=("Brier_score", "mean"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            Spearman_mean=("Spearman", "mean"),
            MAE_mean=("MAE", "mean"),
        )
        .reset_index()
    )
    significance_by_tier = []
    for tier, group in target_independent.groupby("tier", sort=True):
        significance_by_tier.append(
            {
                "tier": tier,
                "tests": len(group),
                "Mann_Whitney_global_FDR_0p05": int(
                    (group["mann_whitney_q_within_all_numeric"] <= 0.05).sum()
                ),
                "Spearman_global_FDR_0p05": int(
                    (group["spearman_q_within_all_numeric"] <= 0.05).sum()
                ),
                "either_test_global_FDR_0p05": int(
                    (
                        (group["mann_whitney_q_within_all_numeric"] <= 0.05)
                        | (group["spearman_q_within_all_numeric"] <= 0.05)
                    ).sum()
                ),
            }
        )
    mechanism_diagnostics = []
    for is_improver, group in outcomes.groupby("primary_improver", sort=False):
        label = "improvers" if bool(is_improver) else "worseners"
        mechanism_diagnostics.append(
            {
                "group": label,
                "galaxies": len(group),
                "mean_outer_RAR_residual_predicted_minus_observed_km_s": float(
                    group["outer_mean_residual_actual_RAR"].mean()
                ),
                "median_outer_RAR_residual_predicted_minus_observed_km_s": float(
                    group["outer_mean_residual_actual_RAR"].median()
                ),
                "mean_outer_screened_residual_predicted_minus_observed_km_s": float(
                    group["outer_mean_residual_actual_screened"].mean()
                ),
                "median_outer_screened_residual_predicted_minus_observed_km_s": float(
                    group["outer_mean_residual_actual_screened"].median()
                ),
                "mean_fractional_MSE_change_screened_vs_RAR": float(
                    group["fractional_MSE_change_screened_vs_RAR"].mean()
                ),
                "tail_helps_at_same_screened_nuisance_fraction": float(
                    group["tail_helps_at_same_screened_nuisance"].mean()
                ),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(args.output / "galaxy_outcomes.csv", index=False)
    features.to_csv(args.output / "galaxy_features.csv", index=False)
    manifest.to_csv(args.output / "feature_manifest.csv", index=False)
    improvers.to_csv(args.output / "complete_improver_roster.csv", index=False)
    univariate.to_csv(args.output / "univariate_numeric_tests.csv", index=False)
    categorical.to_csv(args.output / "categorical_tests.csv", index=False)
    one_rules.to_csv(args.output / "one_feature_subgroup_scan.csv", index=False)
    top_rule_intervals.to_csv(args.output / "top_rule_bootstrap_intervals.csv", index=False)
    pair_rules.to_csv(args.output / "pairwise_subgroup_scan.csv", index=False)
    cv_scores.to_csv(args.output / "cross_validation_repeats.csv", index=False)
    cv_aggregate.to_csv(args.output / "cross_validation_aggregate.csv", index=False)
    cv_repeat_summary.to_csv(args.output / "cross_validation_summary.csv", index=False)
    oof.to_csv(args.output / "cross_validated_predictions.csv", index=False)
    permutation_frame.to_csv(args.output / "core_logistic_permutations.csv", index=False)
    robustness.to_csv(args.output / "robustness_scenarios.csv", index=False)
    influence.to_csv(args.output / "leave_one_out_influence.csv", index=False)
    plot_results(joined, univariate, cv_aggregate, args.output / "forensic_summary.png")

    report = {
        "report_version": "SOLAR-SCREENED-IMPROVER-FORENSICS-0.1.0",
        "status": "completed exhaustive local post-result forensics",
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(args.protocol),
            "status": protocol["status"],
            "disclosure": protocol["disclosure"],
        },
        "inputs": {
            "points_sha256": sha256(points_path),
            "morphology_sha256": sha256(morphology_path),
            "cf4_environment_sha256": sha256(cf4_path),
            "local_void_wall_sha256": sha256(void_wall_path),
            "void_cage_geometry_sha256": sha256(void_cage_path),
            "screened_fits_sha256": sha256(screened_fits_path),
            "baseline_fits_sha256": sha256(baseline_fits_path),
        },
        "sample": {
            "galaxies": len(outcomes),
            "primary_improvers": int(outcomes["primary_improver"].sum()),
            "primary_worseners": int((~outcomes["primary_improver"]).sum()),
            "features_in_inventory": len(manifest),
            "feature_tier_counts": {
                str(key): int(value) for key, value in manifest["tier"].value_counts().items()
            },
            "numeric_univariate_tests": len(univariate),
            "categorical_tests": len(categorical),
            "one_feature_rules_scanned": len(one_rules),
            "pairwise_rules_scanned": len(pair_rules),
        },
        "multiple_testing": {
            "target_independent_features_flagged_global_FDR_0p05": len(
                significant_target_independent
            ),
            "target_independent_one_feature_rules_flagged_global_FDR_0p05": len(
                significant_one_rules
            ),
            "pairwise_rules_flagged_FDR_0p05": len(significant_pair_rules),
            "target_independent_numeric_significance_by_tier": significance_by_tier,
        },
        "top_target_independent_numeric_features": strict_json(
            target_independent.head(20).to_dict(orient="records")
        ),
        "top_outer_descriptive_features": strict_json(
            leakage.head(15).to_dict(orient="records")
        ),
        "top_categorical_features": strict_json(categorical.head(15).to_dict(orient="records")),
        "top_one_feature_subgroups": strict_json(one_rules.head(20).to_dict(orient="records")),
        "top_pairwise_subgroups": strict_json(pair_rules.head(20).to_dict(orient="records")),
        "cross_validation": {
            "feature_set_sizes": {name: len(columns) for name, columns in sets.items()},
            "aggregate": strict_json(cv_aggregate.to_dict(orient="records")),
            "repeat_summary": strict_json(cv_repeat_summary.to_dict(orient="records")),
            "best_preoutcome": strict_json(best_preoutcome.to_dict()),
            "best_including_outer_descriptive_leakage": strict_json(
                best_with_leakage.to_dict()
            ),
            "fixed_primary_permutation_test": permutation_report,
        },
        "outcome_mechanism_diagnostics": mechanism_diagnostics,
        "robustness": strict_json(robustness.to_dict(orient="records")),
        "largest_leave_one_out_influences": strict_json(
            influence.head(15).to_dict(orient="records")
        ),
        "known_unavailable_distinctions": protocol["known_unavailable_distinctions"],
        "claim_rules": protocol["claim_rules"],
    }
    (args.output / "report.json").write_text(
        json.dumps(strict_json(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            strict_json(
                {
                    "sample": report["sample"],
                    "multiple_testing": report["multiple_testing"],
                    "best_preoutcome": report["cross_validation"]["best_preoutcome"],
                    "best_with_leakage": report["cross_validation"][
                        "best_including_outer_descriptive_leakage"
                    ],
                    "permutation": permutation_report,
                    "robustness": report["robustness"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
