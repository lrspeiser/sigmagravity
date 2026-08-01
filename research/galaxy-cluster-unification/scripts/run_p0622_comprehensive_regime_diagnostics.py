#!/usr/bin/env python3
"""Build the P0622 cross-domain regime and failure-mode diagnostic suite."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_arc_apogee_cross_domain import velocity_prediction  # noqa: E402
from run_arc_invariant_absolute_lensing import (  # noqa: E402
    prepare_galaxies,
    response_for_frame,
)
from run_p0554_local_cross_domain_sensitivity import A0  # noqa: E402
from run_solar_screened_galaxy_morphology import (  # noqa: E402
    TYPE_COLUMNS,
    classify_galaxies,
)


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


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


def rms(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    p_values = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p_values)
    result = np.full(len(p_values), np.nan)
    if not valid.any():
        return result
    local = p_values[valid]
    order = np.argsort(local)
    ranked = local[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


def partial_rank_correlation(x, y, control) -> tuple[float, float]:
    values = np.column_stack([x, y, control]).astype(float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 5:
        return np.nan, np.nan
    ranked = np.column_stack([rankdata(values[:, index]) for index in range(3)])
    design = np.column_stack([np.ones(len(ranked)), ranked[:, 2]])
    residual_x = ranked[:, 0] - design @ np.linalg.lstsq(design, ranked[:, 0], rcond=None)[0]
    residual_y = ranked[:, 1] - design @ np.linalg.lstsq(design, ranked[:, 1], rcond=None)[0]
    result = spearmanr(residual_x, residual_y)
    return float(result.statistic), float(result.pvalue)


def score_galaxy_points(block: pd.DataFrame) -> dict:
    if block.empty:
        raise ValueError("cannot score an empty galaxy block")
    arc_residual = block.velocity_P0554_km_s - block.velocity_observed_adjusted_km_s
    rar_residual = block.velocity_fixed_RAR_km_s - block.velocity_observed_adjusted_km_s
    local = block.assign(
        arc_residual_squared=np.square(arc_residual),
        rar_residual_squared=np.square(rar_residual),
    )
    per_galaxy = local.groupby("galaxy").agg(
        arc_mse=("arc_residual_squared", "mean"),
        rar_mse=("rar_residual_squared", "mean"),
    )
    arc_rmse = rms(arc_residual)
    rar_rmse = rms(rar_residual)
    return {
        "galaxies": int(block.galaxy.nunique()),
        "points": int(len(block)),
        "P0554_RMSE_km_s": arc_rmse,
        "fixed_RAR_RMSE_km_s": rar_rmse,
        "P0554_to_RAR_ratio": arc_rmse / rar_rmse,
        "P0554_equal_galaxy_RMSE_km_s": float(np.sqrt(per_galaxy.arc_mse.mean())),
        "RAR_equal_galaxy_RMSE_km_s": float(np.sqrt(per_galaxy.rar_mse.mean())),
        "P0554_mean_residual_km_s": float(arc_residual.mean()),
        "RAR_mean_residual_km_s": float(rar_residual.mean()),
        "P0554_galaxies_better_fraction": float(
            np.mean(per_galaxy.arc_mse < per_galaxy.rar_mse)
        ),
    }


def build_galaxy_data(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent = load_json(protocol["inputs"]["SPARC_parent_protocol"])
    points, _ = prepare_galaxies(parent, A0)
    p0554_protocol = load_json(protocol["inputs"]["P0554_protocol"])
    spec = dict(p0554_protocol["baseline"])
    universal_q = float(spec.pop("universal_q"))
    spec["candidate_id"] = "P0554"
    response = response_for_frame(
        points,
        spec,
        q=1.0,
        a0=A0,
        radius_column="radius_adjusted_kpc",
        gbar_column="g_bar_m_s2",
    )
    points = points.copy()
    points["arc_coordinate"] = response["unit_fractional_response"]
    points["velocity_P0554_km_s"] = velocity_prediction(points, universal_q)
    points["velocity_fixed_RAR_km_s"] = points.velocity_RAR_same_nuisance_km_s

    morphology = classify_galaxies(
        points,
        load_csv(protocol["inputs"]["SPARC_morphology"]),
    )
    outer = points[points.split.eq("outer_holdout")].copy()
    outer = outer.merge(morphology, on="galaxy", how="left", validate="many_to_one")
    outer["radius_over_R80"] = (
        outer.radius_adjusted_kpc / outer.force_equivalent_r80_kpc
    )
    outer["outer_acceleration_family"] = np.select(
        [
            outer.g_bar_m_s2 < 0.03 * A0,
            outer.g_bar_m_s2 < 0.10 * A0,
            outer.g_bar_m_s2 < A0,
        ],
        ["very_deep_below_0p03_a0", "deep_0p03_to_0p10_a0", "transition_0p10_to_1_a0"],
        default="above_a0",
    )
    outer["outer_potential_family"] = np.select(
        [outer.potential_depth < 3.0e-8, outer.potential_depth < 1.0e-7],
        ["shallow_below_3e-8", "intermediate_3e-8_to_1e-7"],
        default="deep_above_1e-7",
    )
    outer["radius_over_R80_family"] = np.select(
        [outer.radius_over_R80 < 1.0, outer.radius_over_R80 < 2.0],
        ["inside_R80", "R80_to_2R80"],
        default="beyond_2R80",
    )
    outer["P0554_residual_km_s"] = (
        outer.velocity_P0554_km_s - outer.velocity_observed_adjusted_km_s
    )
    outer["RAR_residual_km_s"] = (
        outer.velocity_fixed_RAR_km_s - outer.velocity_observed_adjusted_km_s
    )

    score_rows = [{"dimension": "all", "bin": "all", **score_galaxy_points(outer)}]
    for dimension in protocol["regimes"]["galaxy_dimensions"]:
        for label, block in outer.groupby(dimension, sort=True):
            score_rows.append(
                {"dimension": dimension, "bin": label, **score_galaxy_points(block)}
            )
    return outer, pd.DataFrame(score_rows)


def per_galaxy_metrics(outer: pd.DataFrame) -> pd.DataFrame:
    local = outer.assign(
        arc_squared=np.square(outer.P0554_residual_km_s),
        rar_squared=np.square(outer.RAR_residual_km_s),
    )
    aggregate = local.groupby("galaxy", sort=True).agg(
        points=("galaxy", "size"),
        P0554_MSE=("arc_squared", "mean"),
        RAR_MSE=("rar_squared", "mean"),
        P0554_mean_residual_km_s=("P0554_residual_km_s", "mean"),
        RAR_mean_residual_km_s=("RAR_residual_km_s", "mean"),
        median_outer_gbar=("g_bar_m_s2", "median"),
        median_radius_over_R80=("radius_over_R80", "median"),
    ).reset_index()
    aggregate["P0554_RMSE_km_s"] = np.sqrt(aggregate.P0554_MSE)
    aggregate["RAR_RMSE_km_s"] = np.sqrt(aggregate.RAR_MSE)
    aggregate["P0554_to_RAR_ratio"] = (
        aggregate.P0554_RMSE_km_s / aggregate.RAR_RMSE_km_s
    )
    aggregate["log_error_ratio"] = np.log(aggregate.P0554_to_RAR_ratio)
    feature_columns = [
        "galaxy",
        "hubble_type_y",
        "inclination_deg",
        "disk_scale_kpc_y",
        "disk_central_surface_brightness",
        "baryonic_mass_solar_x",
        "stellar_bulge_fraction_x",
        "gas_fraction_x",
        "outer_log_velocity_slope",
        "potential_depth",
        *TYPE_COLUMNS,
    ]
    available = [column for column in feature_columns if column in outer.columns]
    features = outer.sort_values("galaxy").drop_duplicates("galaxy")[available].copy()
    features = features.rename(
        columns={
            "disk_scale_kpc_y": "disk_scale_kpc",
            "hubble_type_y": "hubble_type",
            "baryonic_mass_solar_x": "baryonic_mass_solar",
            "stellar_bulge_fraction_x": "stellar_bulge_fraction",
            "gas_fraction_x": "gas_fraction",
        }
    )
    return aggregate.merge(features, on="galaxy", how="left", validate="one_to_one")


def bootstrap_ratio(block: pd.DataFrame, *, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arc = block.P0554_MSE.to_numpy(float)
    rar = block.RAR_MSE.to_numpy(float)
    samples = rng.integers(0, len(block), size=(draws, len(block)))
    ratios = np.sqrt(np.mean(arc[samples], axis=1)) / np.sqrt(
        np.mean(rar[samples], axis=1)
    )
    return tuple(float(value) for value in np.quantile(ratios, [0.025, 0.975]))


def galaxy_interactions(
    galaxies: pd.DataFrame, protocol: dict
) -> pd.DataFrame:
    rows = []
    minimum = int(protocol["regimes"]["minimum_interaction_galaxies"])
    draws = int(protocol["regimes"]["bootstrap_draws"])
    seed = int(protocol["regimes"]["bootstrap_seed"])
    for pair_index, (left, right) in enumerate(protocol["regimes"]["galaxy_interactions"]):
        for labels, block in galaxies.groupby([left, right], sort=True):
            if len(block) < minimum:
                continue
            ratio = float(np.sqrt(block.P0554_MSE.mean()) / np.sqrt(block.RAR_MSE.mean()))
            low, high = bootstrap_ratio(
                block, draws=draws, seed=seed + 100 * pair_index + len(rows)
            )
            rows.append(
                {
                    "left_dimension": left,
                    "left_bin": labels[0],
                    "right_dimension": right,
                    "right_bin": labels[1],
                    "galaxies": len(block),
                    "P0554_equal_galaxy_RMSE_km_s": float(np.sqrt(block.P0554_MSE.mean())),
                    "RAR_equal_galaxy_RMSE_km_s": float(np.sqrt(block.RAR_MSE.mean())),
                    "P0554_to_RAR_ratio": ratio,
                    "ratio_bootstrap_2p5": low,
                    "ratio_bootstrap_97p5": high,
                    "P0554_mean_residual_km_s": float(block.P0554_mean_residual_km_s.mean()),
                    "P0554_better_fraction": float(np.mean(block.P0554_MSE < block.RAR_MSE)),
                    "descriptive_outcome": "P0554_better" if ratio < 1.0 else "fixed_RAR_better",
                }
            )
    return pd.DataFrame(rows).sort_values("P0554_to_RAR_ratio")


def galaxy_correlations(galaxies: pd.DataFrame) -> pd.DataFrame:
    feature_map = {
        "log10_baryonic_mass": np.log10(galaxies.baryonic_mass_solar),
        "gas_fraction": galaxies.gas_fraction,
        "stellar_bulge_fraction": galaxies.stellar_bulge_fraction,
        "disk_central_surface_brightness": galaxies.disk_central_surface_brightness,
        "hubble_type": galaxies.hubble_type,
        "inclination": galaxies.inclination_deg,
        "disk_scale_kpc": galaxies.disk_scale_kpc,
        "outer_rotation_slope_observed": galaxies.outer_log_velocity_slope,
        "median_outer_gbar": np.log10(galaxies.median_outer_gbar),
        "potential_depth": np.log10(galaxies.potential_depth),
        "median_radius_over_R80": galaxies.median_radius_over_R80,
    }
    targets = {
        "log_P0554_to_RAR_error_ratio": galaxies.log_error_ratio,
        "P0554_mean_velocity_residual": galaxies.P0554_mean_residual_km_s,
    }
    control = np.log10(galaxies.baryonic_mass_solar).to_numpy(float)
    rows = []
    for feature, values in feature_map.items():
        for target, outcome in targets.items():
            frame = np.column_stack([values, outcome]).astype(float)
            frame = frame[np.isfinite(frame).all(axis=1)]
            result = spearmanr(frame[:, 0], frame[:, 1])
            partial_rho, partial_p = partial_rank_correlation(values, outcome, control)
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "galaxies": len(frame),
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "partial_rho_controlling_log_mass": partial_rho,
                    "partial_p_value_controlling_log_mass": partial_p,
                    "input_safe_for_blind_prediction": feature != "outer_rotation_slope_observed",
                }
            )
    result = pd.DataFrame(rows)
    result["BH_q_value"] = bh_qvalues(result.p_value.to_numpy(float))
    result["partial_BH_q_value"] = bh_qvalues(
        result.partial_p_value_controlling_log_mass.to_numpy(float)
    )
    return result.sort_values(["target", "BH_q_value", "feature"])


def galaxy_outliers(galaxies: pd.DataFrame) -> pd.DataFrame:
    selections = []
    for label, frame in (
        ("lowest_error_ratio", galaxies.nsmallest(12, "P0554_to_RAR_ratio")),
        ("highest_error_ratio", galaxies.nlargest(12, "P0554_to_RAR_ratio")),
        ("most_negative_bias", galaxies.nsmallest(12, "P0554_mean_residual_km_s")),
        ("most_positive_bias", galaxies.nlargest(12, "P0554_mean_residual_km_s")),
    ):
        local = frame.copy()
        local.insert(0, "outlier_type", label)
        selections.append(local)
    columns = [
        "outlier_type",
        "galaxy",
        "points",
        "P0554_RMSE_km_s",
        "RAR_RMSE_km_s",
        "P0554_to_RAR_ratio",
        "P0554_mean_residual_km_s",
        "baryonic_mass_solar",
        "gas_fraction",
        "stellar_bulge_fraction",
        "disk_central_surface_brightness",
        "outer_log_velocity_slope",
        *TYPE_COLUMNS,
    ]
    return pd.concat(selections, ignore_index=True)[columns]


def build_cluster_data(
    protocol: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states = load_csv(protocol["inputs"]["P0615_states"])
    scores = load_csv(protocol["inputs"]["P0618_scores"])
    preferences = load_csv(protocol["inputs"]["P0618_preferences"])
    alignment = load_csv(protocol["inputs"]["P0617_alignment"])
    controls = scores[scores.variant_id.eq("scalar_control")][
        ["system_label", "heldout_RMS_arcsec"]
    ].rename(columns={"heldout_RMS_arcsec": "scalar_control_RMS_arcsec"})
    phases = scores[~scores.variant_id.eq("scalar_control")].merge(
        controls, on="system_label", validate="many_to_one"
    )
    phases["complete"] = phases.heldout_all_roots.astype(bool)
    phases["improvement_fraction"] = np.where(
        phases.complete,
        1.0 - phases.heldout_RMS_arcsec / phases.scalar_control_RMS_arcsec,
        np.nan,
    )
    cluster_rows = []
    for label, block in phases.groupby("system_label", sort=True):
        phase90 = block[np.isclose(block.phase_degrees, 90.0)].iloc[0]
        complete = block[block.complete & np.isfinite(block.improvement_fraction)]
        pref = preferences[preferences.system_label.eq(label)].iloc[0]
        cluster_rows.append(
            {
                "system_label": label,
                "evidence_class": "spent_diagnostic_fixed_geometry",
                "scalar_control_RMS_arcsec": float(phase90.scalar_control_RMS_arcsec),
                "phase90_RMS_arcsec": float(phase90.heldout_RMS_arcsec),
                "phase90_improvement_fraction": float(phase90.improvement_fraction),
                "phase90_all_roots": bool(phase90.complete),
                "root_complete_phase_fraction": float(block.complete.mean()),
                "phase_response_span": float(
                    complete.improvement_fraction.max() - complete.improvement_fraction.min()
                ),
                "preferred_phase_degrees_diagnostic_only": float(pref.preferred_phase_degrees),
                "best_phase_improvement_diagnostic_only": float(pref.best_diagnostic_improvement),
                "phase90_outcome": (
                    "strong_improvement"
                    if phase90.improvement_fraction > 0.01
                    else "weak_improvement"
                    if phase90.improvement_fraction >= 0.0
                    else "worse"
                ),
            }
        )
    clusters = pd.DataFrame(cluster_rows)
    clusters = clusters.merge(states, on="system_label", validate="one_to_one")
    clusters = clusters.merge(alignment, on="system_label", validate="one_to_one")

    p0619_states = load_csv(protocol["inputs"]["P0619_states"])
    p0619_scores = load_csv(protocol["inputs"]["P0619_scores"])
    for _, state in p0619_states.iterrows():
        label = state.system_label
        control = p0619_scores[
            p0619_scores.system_label.eq(label)
            & p0619_scores.variant_id.eq("P0554_scalar_control")
        ].iloc[0]
        candidate = p0619_scores[
            p0619_scores.system_label.eq(label)
            & p0619_scores.variant_id.eq("P0619_tangential_self_route")
        ].iloc[0]
        complete = bool(
            int(control.heldout_roots_converged) == int(control.heldout_images)
            and int(candidate.heldout_roots_converged) == int(candidate.heldout_images)
            and np.isfinite(float(control.heldout_RMS_arcsec))
            and np.isfinite(float(candidate.heldout_RMS_arcsec))
        )
        improvement = (
            1.0 - float(candidate.heldout_RMS_arcsec) / float(control.heldout_RMS_arcsec)
            if complete
            else np.nan
        )
        row = {column: np.nan for column in clusters.columns}
        row.update(state.to_dict())
        row.update(
            {
                "system_label": label,
                "cohort": "P0619_transfer",
                "evidence_class": "chronological_formula_transfer_full_refit",
                "scalar_control_RMS_arcsec": float(control.heldout_RMS_arcsec),
                "phase90_RMS_arcsec": float(candidate.heldout_RMS_arcsec),
                "phase90_improvement_fraction": improvement,
                "phase90_all_roots": complete,
                "root_complete_phase_fraction": np.nan,
                "phase_response_span": np.nan,
                "preferred_phase_degrees_diagnostic_only": np.nan,
                "best_phase_improvement_diagnostic_only": np.nan,
                "phase90_outcome": (
                    "root_incomplete"
                    if not complete
                    else "weak_improvement"
                    if improvement >= 0.0
                    else "worse"
                ),
            }
        )
        clusters = pd.concat([clusters, pd.DataFrame([row])], ignore_index=True)

    analysis = clusters[clusters.evidence_class.eq("spent_diagnostic_fixed_geometry")]
    correlation_rows = []
    features = [
        "R80_kpc",
        "Delta80",
        "self_routed_fraction",
        "quadrupole_Q",
        "epsilon_quadratic_Q2_over_total",
        "scalar_control_RMS_arcsec",
        "weighted_alignment_cosine",
        "route_shift_RMS_arcsec",
    ]
    for feature in features:
        block = analysis[[feature, "phase90_improvement_fraction"]].dropna()
        result = spearmanr(block[feature], block.phase90_improvement_fraction)
        correlation_rows.append(
            {
                "feature": feature,
                "target": "phase90_improvement_fraction",
                "systems": len(block),
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
                "interpretation_limit": "descriptive only; five spent fixed-geometry systems",
            }
        )
    correlations = pd.DataFrame(correlation_rows)
    correlations["BH_q_value"] = bh_qvalues(correlations.p_value.to_numpy(float))

    improvements = analysis.set_index("system_label").phase90_improvement_fraction
    all_mean = float(improvements.mean())
    leave_rows = [
        {
            "omitted_system": "none",
            "systems_retained": len(improvements),
            "mean_phase90_improvement_fraction": all_mean,
            "change_from_all_system_mean": 0.0,
        }
    ]
    for label in improvements.index:
        mean = float(improvements.drop(label).mean())
        leave_rows.append(
            {
                "omitted_system": label,
                "systems_retained": len(improvements) - 1,
                "mean_phase90_improvement_fraction": mean,
                "change_from_all_system_mean": mean - all_mean,
            }
        )
    return clusters, phases, pd.DataFrame(correlation_rows).assign(
        BH_q_value=correlations.BH_q_value
    ), pd.DataFrame(leave_rows)


def parameter_domain_matrix(protocol: dict) -> pd.DataFrame:
    local = load_csv(protocol["inputs"]["P0554_parameter_impacts"])
    multiscale = load_csv(protocol["inputs"]["P0554_multiscale"])[
        [
            "parameter",
            "stable_material_nonSolar_domains",
            "stable_nonSolar_directions_agree",
            "smallest_root_bifurcation_u",
            "root_bifurcation_at_smallest_step",
            "smallest_solar_boundary_crossing_u",
        ]
    ]
    local = local.merge(multiscale, on="parameter", how="left", validate="one_to_one")
    rows = []
    for _, row in local.iterrows():
        directions = [
            row.galaxy_better_direction,
            row.cluster_better_direction,
            row.RXJ2129_better_direction,
            row.four_cluster_better_direction,
        ]
        active = [value for value in directions if value in {"low", "high"}]
        rows.append(
            {
                "parameter_family": "P0554_scalar",
                "parameter": row.parameter,
                "concept": row.concept,
                "galaxy_normalized_response": row.galaxy_normalized_span,
                "derived_cluster_normalized_response": row.cluster_normalized_span,
                "raw_cluster_normalized_response": max(
                    row.RXJ2129_normalized_span, row.four_cluster_normalized_span
                ),
                "solar_margin_response": row.solar_margin_fraction_span,
                "root_count_span": abs(row.RXJ2129_low_roots - row.RXJ2129_high_roots)
                + abs(row.four_cluster_low_roots - row.four_cluster_high_roots),
                "cross_domain_direction_conflict": len(set(active)) > 1,
                "stable_nonSolar_directions_agree": row.stable_nonSolar_directions_agree,
                "smallest_root_bifurcation_fraction": row.smallest_root_bifurcation_u,
                "smallest_solar_boundary_crossing_fraction": row.smallest_solar_boundary_crossing_u,
                "main_lesson": "local scalar response; compare directions, roots, and Solar boundary separately",
            }
        )

    for _, row in load_csv(protocol["inputs"]["P0613_parameter_impacts"]).iterrows():
        rows.append(
            {
                "parameter_family": "bounded_route_factorial",
                "parameter": row.parameter,
                "concept": "route topology/support coordinate",
                "galaxy_normalized_response": row.SPARC_RMSE_span_km_s / 70.92603164566839,
                "derived_cluster_normalized_response": np.nan,
                "raw_cluster_normalized_response": row.maximum_system_root_pattern_span / 3.0,
                "solar_margin_response": 0.0,
                "root_count_span": row.mean_root_count_span,
                "cross_domain_direction_conflict": np.nan,
                "stable_nonSolar_directions_agree": np.nan,
                "smallest_root_bifurcation_fraction": np.nan,
                "smallest_solar_boundary_crossing_fraction": np.nan,
                "main_lesson": "route width and strength alter roots; contrast cap is nearly marginal",
            }
        )
    for _, row in load_csv(protocol["inputs"]["P0617_family_impacts"]).iterrows():
        if row.family == "baseline":
            continue
        rows.append(
            {
                "parameter_family": "self_coupled_route_support",
                "parameter": row.family,
                "concept": "baryon-derived support-law family",
                "galaxy_normalized_response": 0.0,
                "derived_cluster_normalized_response": np.nan,
                "raw_cluster_normalized_response": max(
                    row.mean_system_improvement_span,
                    row.RXJ2129_improvement_span,
                ),
                "solar_margin_response": 0.0,
                "root_count_span": row.maximum_combined_roots - row.minimum_combined_roots,
                "cross_domain_direction_conflict": np.nan,
                "stable_nonSolar_directions_agree": np.nan,
                "smallest_root_bifurcation_fraction": np.nan,
                "smallest_solar_boundary_crossing_fraction": np.nan,
                "main_lesson": "support changes magnitude but does not choose the beneficial angular sign",
            }
        )
    phase_report = load_json(protocol["inputs"]["P0618_report"])
    complete = [
        row
        for row in phase_report["universal_phase_responses"]
        if row["all_18_roots"]
    ]
    rows.append(
        {
            "parameter_family": "angular_route_phase",
            "parameter": "universal_phase",
            "concept": "rotation shared by every cluster",
            "galaxy_normalized_response": 0.0,
            "derived_cluster_normalized_response": np.nan,
            "raw_cluster_normalized_response": max(row["mean_system_improvement"] for row in complete)
            - min(row["mean_system_improvement"] for row in complete),
            "solar_margin_response": 0.0,
            "root_count_span": max(row["combined_roots"] for row in phase_report["universal_phase_responses"])
            - min(row["combined_roots"] for row in phase_report["universal_phase_responses"]),
            "cross_domain_direction_conflict": True,
            "stable_nonSolar_directions_agree": False,
            "smallest_root_bifurcation_fraction": np.nan,
            "smallest_solar_boundary_crossing_fraction": np.nan,
            "main_lesson": "largest recent raw-lens lever, but preferred direction remains cluster-dependent",
        }
    )
    result = pd.DataFrame(rows)
    response_columns = [
        "galaxy_normalized_response",
        "derived_cluster_normalized_response",
        "raw_cluster_normalized_response",
        "solar_margin_response",
    ]
    result["maximum_recorded_normalized_response"] = result[response_columns].max(axis=1)
    return result.sort_values(
        ["maximum_recorded_normalized_response", "root_count_span"], ascending=False
    )


def scenario_matrix(
    galaxy_scores: pd.DataFrame,
    clusters: pd.DataFrame,
    protocol: dict,
) -> pd.DataFrame:
    p0554 = load_json(protocol["inputs"]["P0554_report"])
    p0613 = load_json(protocol["inputs"]["P0613_report"])
    p0614 = load_json(protocol["inputs"]["P0614_report"])
    p0618 = load_json(protocol["inputs"]["P0618_report"])
    p0619 = load_json(protocol["inputs"]["P0619_report"])

    def galaxy_row(dimension, label):
        return galaxy_scores[
            galaxy_scores.dimension.eq(dimension) & galaxy_scores.bin.eq(label)
        ].iloc[0]

    rows = []

    def add(domain, condition, evidence, metric, value, comparator, result, lesson, passed):
        rows.append(
            {
                "domain": domain,
                "condition": condition,
                "evidence_class": evidence,
                "metric": metric,
                "candidate_value": value,
                "comparator": comparator,
                "result": result,
                "diagnostic_lesson": lesson,
                "passes_declared_gate": passed,
            }
        )

    overall = galaxy_row("all", "all")
    dwarf = galaxy_row("baryonic_mass_family", "dwarf_mass")
    giant = galaxy_row("baryonic_mass_family", "giant_mass")
    gas_rich = galaxy_row("gas_fraction_family", "gas_rich")
    gas_poor = galaxy_row("gas_fraction_family", "gas_poor")
    add("galaxy", "all 131 outer holdouts", "raw_observation", "RMSE km/s", overall.P0554_RMSE_km_s, f"fixed RAR {overall.fixed_RAR_RMSE_km_s:.3f}", "worse", "scalar parent is close but not at the fixed-RAR level", overall.P0554_to_RAR_ratio <= 1.1)
    add("galaxy", "dwarf mass", "raw_observation", "mean residual km/s", dwarf.P0554_mean_residual_km_s, f"RAR RMSE {dwarf.fixed_RAR_RMSE_km_s:.3f}", "underprediction", "largest mass-bin relative penalty", False)
    add("galaxy", "giant mass", "raw_observation", "mean residual km/s", giant.P0554_mean_residual_km_s, f"RAR RMSE {giant.fixed_RAR_RMSE_km_s:.3f}", "overprediction", "bias reverses sign from dwarfs to giants", False)
    add("galaxy", "gas rich", "raw_observation", "mean residual km/s", gas_rich.P0554_mean_residual_km_s, f"gas-poor {gas_poor.P0554_mean_residual_km_s:+.3f}", "mass-linked underprediction", "gas trend must be tested after controlling mass", False)
    add("galaxy", "axisymmetric route layer", "inherited_result", "route change", 0.0, "defined symmetry null", "compatible but untested", "zero change cannot validate the angular route", True)
    add("cluster", "20 CLASH derived acceleration profiles", "derived_observation", "RMSE dex", p0554["baseline"]["cluster_RMSE_dex"], "NFW-derived target", "descriptive only", "radial target is not raw lensing", None)
    add("cluster", "five fixed-geometry +90 phase systems", "spent_diagnostic", "mean improvement %", 100 * p0618["selected_universal_phase"]["mean_system_improvement"], "P0554 scalar", "3 of 5 improve", "mean is strongly influenced by RXJ2129", False)
    rxj = clusters[clusters.system_label.eq("RXJ2129")].iloc[0]
    add("cluster", "RXJ2129 +90 phase", "spent_diagnostic", "improvement %", 100 * rxj.phase90_improvement_fraction, "P0554 scalar", "large local gain", "one cluster supplies most of the mean phase gain", True)
    a383 = clusters[clusters.system_label.eq("A383")].iloc[0]
    add("cluster", "A383 frozen formula full refit", "raw_observation", "heldout RMS arcsec", a383.phase90_RMS_arcsec, f"P0554 {a383.scalar_control_RMS_arcsec:.3f}", "0.174% gain but large absolute error", "direction transfers; adequacy does not", False)
    ms = clusters[clusters.system_label.eq("MS2137")].iloc[0]
    add("cluster", "MS2137 frozen formula full refit", "raw_observation", "heldout roots", 2.0, "3 required", "root incomplete", "no RMS claim is permitted", False)
    add("cluster", "raw validation aggregate", "raw_observation", "RMS arcsec", 19.07556990937222, "compact halo 9.989", "1.91x worse", "current formula does not match the limited halo comparator", False)
    add("topology", "negative universal phase on RXJ2129", "spent_diagnostic", "combined roots", 17.0, "18 required", "one root lost", "phase response is non-smooth across a caustic", False)
    add("topology", "bounded width-strength factorial", "spent_diagnostic", "root-safe variants", p0613["coverage"]["root_safe_variants"], "27 variants", "narrow safe region", "RMS optimization must be subordinate to root completeness", False)
    add("Solar", "P0554 scalar proxies", "analytic_proxy", "Mercury mas/century", p0554["baseline"]["Mercury_precession_mas_per_century"], "absolute margin 3.1", "pass", "proxy safety is retained", True)
    add("Solar", "point-source angular route", "synthetic_invariant", "route change", 0.0, "exact null", "compatible but untested", "null says nothing about cluster correctness", True)
    add("universality", "same formula accounting", "spent_diagnostic", "formula promoted", 0.0, "all domains required", "no", "galaxy, absolute lensing, and universality gates do not all pass", False)
    return pd.DataFrame(rows)


def make_figure(
    galaxy_scores: pd.DataFrame,
    galaxies: pd.DataFrame,
    clusters: pd.DataFrame,
    phases: pd.DataFrame,
    parameters: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    selected = galaxy_scores[
        galaxy_scores.dimension.isin(
            ["baryonic_mass_family", "gas_fraction_family", "stellar_structure"]
        )
    ].copy()
    selected["label"] = selected.dimension.str.replace("_family", "", regex=False) + ": " + selected.bin
    selected = selected.sort_values("P0554_to_RAR_ratio")
    axes[0, 0].barh(selected.label, selected.P0554_to_RAR_ratio, color="#4c78a8")
    axes[0, 0].axvline(1.0, color="black", lw=0.9)
    axes[0, 0].set(xlabel="P0554 RMSE / fixed-RAR RMSE", title="Galaxy error is regime-dependent")

    scatter = axes[0, 1].scatter(
        np.log10(galaxies.baryonic_mass_solar),
        galaxies.P0554_mean_residual_km_s,
        c=galaxies.gas_fraction,
        cmap="viridis",
        alpha=0.8,
    )
    axes[0, 1].axhline(0.0, color="black", lw=0.9)
    axes[0, 1].set(xlabel="log10 baryonic mass", ylabel="mean P0554 residual (km/s)", title="Dwarfs tend low; giants cross to high")
    figure.colorbar(scatter, ax=axes[0, 1], label="gas fraction")

    phase_clusters = clusters[clusters.evidence_class.eq("spent_diagnostic_fixed_geometry")]
    colors = ["#59a14f" if value >= 0.0 else "#e15759" for value in phase_clusters.phase90_improvement_fraction]
    axes[0, 2].bar(phase_clusters.system_label, 100 * phase_clusters.phase90_improvement_fraction, color=colors)
    axes[0, 2].axhline(0.0, color="black", lw=0.9)
    axes[0, 2].tick_params(axis="x", rotation=35)
    axes[0, 2].set(ylabel="change vs scalar (%)", title="One phase produces mixed cluster signs")

    pivot = phases.pivot(index="system_label", columns="phase_degrees", values="improvement_fraction")
    image = axes[1, 0].imshow(100 * pivot.to_numpy(float), aspect="auto", cmap="RdBu", vmin=-1.0, vmax=1.0)
    axes[1, 0].set(
        xticks=np.arange(len(pivot.columns)),
        xticklabels=[f"{value:g}" for value in pivot.columns],
        yticks=np.arange(len(pivot.index)),
        yticklabels=pivot.index,
        xlabel="universal phase (degrees)",
        title="Phase response and root-loss gaps",
    )
    figure.colorbar(image, ax=axes[1, 0], label="improvement (%)")

    scalar = parameters[parameters.parameter_family.eq("P0554_scalar")].head(10)
    matrix = scalar[
        [
            "galaxy_normalized_response",
            "derived_cluster_normalized_response",
            "raw_cluster_normalized_response",
            "solar_margin_response",
        ]
    ].to_numpy(float)
    shown = np.clip(matrix, 0.0, 1.0)
    image = axes[1, 1].imshow(shown, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 1].set(
        xticks=np.arange(4),
        xticklabels=["galaxy", "derived lens", "raw lens", "Solar"],
        yticks=np.arange(len(scalar)),
        yticklabels=scalar.parameter,
        title="The same parameter rarely dominates every domain",
    )
    axes[1, 1].tick_params(axis="x", rotation=25)
    figure.colorbar(image, ax=axes[1, 1], label="normalized response (clipped at 1)")

    loo = leave_one_out[~leave_one_out.omitted_system.eq("none")]
    axes[1, 2].barh(loo.omitted_system, 100 * loo.mean_phase90_improvement_fraction, color="#f28e2b")
    all_mean = float(leave_one_out.iloc[0].mean_phase90_improvement_fraction)
    axes[1, 2].axvline(100 * all_mean, color="black", ls="--", label="all five")
    axes[1, 2].set(xlabel="mean improvement after omission (%)", title="Leave-one-cluster-out influence")
    axes[1, 2].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    protocol_path = ROOT / "configs/p0622_comprehensive_regime_diagnostics_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_P0622_regime_aggregation":
        raise RuntimeError("P0622 protocol is not frozen")

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    outer, galaxy_scores = build_galaxy_data(protocol)
    galaxies = per_galaxy_metrics(outer)
    interactions = galaxy_interactions(galaxies, protocol)
    correlations = galaxy_correlations(galaxies)
    outliers = galaxy_outliers(galaxies)
    clusters, phases, cluster_correlations, leave_one_out = build_cluster_data(protocol)
    parameters = parameter_domain_matrix(protocol)
    scenarios = scenario_matrix(galaxy_scores, clusters, protocol)

    point_columns = [
        "galaxy",
        "radius_catalog_kpc",
        "radius_adjusted_kpc",
        "radius_over_R80",
        "g_bar_m_s2",
        "potential_depth",
        "velocity_observed_adjusted_km_s",
        "velocity_error_total_km_s",
        "velocity_P0554_km_s",
        "velocity_fixed_RAR_km_s",
        "P0554_residual_km_s",
        "RAR_residual_km_s",
        *TYPE_COLUMNS,
        "outer_acceleration_family",
        "outer_potential_family",
        "radius_over_R80_family",
    ]
    outer[point_columns].to_csv(output / protocol["outputs"]["galaxy_points"], index=False)
    galaxy_scores.to_csv(output / protocol["outputs"]["galaxy_scores"], index=False)
    interactions.to_csv(output / protocol["outputs"]["galaxy_interactions"], index=False)
    correlations.to_csv(output / protocol["outputs"]["galaxy_correlations"], index=False)
    outliers.to_csv(output / protocol["outputs"]["galaxy_outliers"], index=False)
    clusters.to_csv(output / protocol["outputs"]["cluster_scores"], index=False)
    phases.to_csv(output / protocol["outputs"]["cluster_phases"], index=False)
    cluster_correlations.to_csv(output / protocol["outputs"]["cluster_correlations"], index=False)
    leave_one_out.to_csv(output / protocol["outputs"]["cluster_leave_one_out"], index=False)
    parameters.to_csv(output / protocol["outputs"]["parameter_matrix"], index=False)
    scenarios.to_csv(output / protocol["outputs"]["scenario_matrix"], index=False)

    overall = galaxy_scores[
        galaxy_scores.dimension.eq("all") & galaxy_scores.bin.eq("all")
    ].iloc[0]
    mass = galaxy_scores[galaxy_scores.dimension.eq("baryonic_mass_family")].set_index("bin")
    gas = galaxy_scores[galaxy_scores.dimension.eq("gas_fraction_family")].set_index("bin")
    phase_cluster = clusters[clusters.evidence_class.eq("spent_diagnostic_fixed_geometry")]
    no_rxj = leave_one_out[leave_one_out.omitted_system.eq("RXJ2129")].iloc[0]
    strongest_interaction = interactions.iloc[-1]
    weakest_interaction = interactions.iloc[0]
    strongest_safe_correlation = correlations[
        correlations.input_safe_for_blind_prediction
        & correlations.target.eq("log_P0554_to_RAR_error_ratio")
    ].sort_values("BH_q_value").iloc[0]
    p0614 = load_json(protocol["inputs"]["P0614_report"])
    p0620 = load_json(protocol["inputs"]["P0620_report"])
    solar_pass = bool(p0614["gates"]["Solar_all_proxies_pass"])
    raw_halo_ratio = next(
        row["ratio_to_comparator"]
        for row in p0614["scorecard"]
        if row["domain"] == "raw validation clusters" and row["comparator"] == "compact halo"
    )
    decision = {
        "galaxy_parity_gate_pass": bool(
            overall.P0554_to_RAR_ratio
            <= protocol["decision_rules"]["galaxy_RMSE_to_fixed_RAR_max"]
        ),
        "raw_cluster_halo_gate_pass": bool(
            raw_halo_ratio
            <= protocol["decision_rules"]["cluster_raw_RMS_to_compact_halo_max"]
        ),
        "Solar_proxy_gate_pass": solar_pass,
        "universal_phase_sign_gate_pass": bool(
            np.sum(phase_cluster.phase90_improvement_fraction >= 0.0) == len(phase_cluster)
        ),
        "per_object_phase_selection_used": False,
        "formula_promoted": False,
    }
    decision["all_promotion_gates_pass"] = bool(
        decision["galaxy_parity_gate_pass"]
        and decision["raw_cluster_halo_gate_pass"]
        and decision["Solar_proxy_gate_pass"]
        and decision["universal_phase_sign_gate_pass"]
    )

    report = {
        "report_version": "P0622-COMPREHENSIVE-REGIME-DIAGNOSTICS-RESULTS-0.1.0",
        "status": "complete_cross_domain_regime_diagnostic",
        "coverage": {
            "SPARC_galaxies": int(outer.galaxy.nunique()),
            "SPARC_outer_points": len(outer),
            "galaxy_regime_rows": len(galaxy_scores),
            "galaxy_interaction_rows_minimum_five_systems": len(interactions),
            "galaxy_continuous_correlation_tests": len(correlations),
            "fixed_geometry_phase_systems": len(phase_cluster),
            "phase_variants_per_system": int(phases.phase_degrees.nunique()),
            "chronological_full_refit_transfer_systems": 2,
            "parameter_domain_rows": len(parameters),
            "scenario_rows": len(scenarios),
            "new_fitted_gravity_parameters": 0,
            "per_object_gravity_parameters": 0,
        },
        "formula": protocol["formula"],
        "suite_design": protocol["suite_design"],
        "headline_scores": {
            "P0554_SPARC_outer_RMSE_km_s": float(overall.P0554_RMSE_km_s),
            "fixed_RAR_SPARC_outer_RMSE_km_s": float(overall.fixed_RAR_RMSE_km_s),
            "P0554_to_RAR_ratio": float(overall.P0554_to_RAR_ratio),
            "phase90_five_system_mean_improvement": float(
                phase_cluster.phase90_improvement_fraction.mean()
            ),
            "phase90_five_system_median_improvement": float(
                phase_cluster.phase90_improvement_fraction.median()
            ),
            "phase90_mean_without_RXJ2129": float(
                no_rxj.mean_phase90_improvement_fraction
            ),
            "phase90_systems_improved": int(
                np.sum(phase_cluster.phase90_improvement_fraction >= 0.0)
            ),
            "phase90_systems_tested": len(phase_cluster),
            "raw_validation_to_compact_halo_RMS_ratio": raw_halo_ratio,
            "Solar_all_proxies_pass": solar_pass,
        },
        "differential_findings": {
            "mass_bias_reversal": {
                "dwarf_mean_residual_km_s": float(mass.loc["dwarf_mass"].P0554_mean_residual_km_s),
                "giant_mean_residual_km_s": float(mass.loc["giant_mass"].P0554_mean_residual_km_s),
                "interpretation": "P0554 tends to underpredict dwarf outer speeds and overpredict giant outer speeds; this is a scalar amplitude/transition problem, not an angular-route effect.",
            },
            "gas_trend": {
                "gas_rich_mean_residual_km_s": float(gas.loc["gas_rich"].P0554_mean_residual_km_s),
                "gas_poor_mean_residual_km_s": float(gas.loc["gas_poor"].P0554_mean_residual_km_s),
                "interpretation": "The raw gas trend is strong, but much of it tracks baryonic mass; use the partial correlations before adding a gas parameter.",
            },
            "strongest_supported_continuous_galaxy_driver": strongest_safe_correlation.to_dict(),
            "best_interaction_bin": weakest_interaction.to_dict(),
            "worst_interaction_bin": strongest_interaction.to_dict(),
            "cluster_phase_heterogeneity": {
                "systems_improved": int(np.sum(phase_cluster.phase90_improvement_fraction >= 0.0)),
                "systems_worsened": int(np.sum(phase_cluster.phase90_improvement_fraction < 0.0)),
                "mean_improvement_fraction": float(phase_cluster.phase90_improvement_fraction.mean()),
                "mean_without_RXJ2129": float(no_rxj.mean_phase90_improvement_fraction),
                "interpretation": "RXJ2129 supplies most of the +90-degree mean gain; the other four systems are close to neutral and have mixed signs.",
            },
            "cluster_direction_lesson": "Quadrupole Q and Delta80 determine route amplitude, but the pre-existing residual-vector alignment determines whether the induced shift helps. A baryon-predicted direction is still missing.",
            "topology_lesson": "Small parameter or phase changes can cross caustics and lose or create roots; full root completeness must precede any RMS comparison.",
            "symmetry_null_lesson": "The exact galaxy and Solar route nulls protect compatibility but provide no affirmative test of the angular route physics.",
            "domain_conflict_lesson": "No scalar or route coordinate is both dominant and directionally consistent across galaxy rotation, raw cluster images, and Solar proxies.",
        },
        "decision": decision,
        "next_discriminating_tests": [
            "Fit a universal scalar correction only on a development subset, then require it to remove the dwarf-to-giant residual sign reversal on untouched SPARC galaxies without using observed rotation-curve shape as an input.",
            "Construct a baryon-only angular predictor from gas-star centroid offsets, external tidal axes, or resolved multipole orientation; freeze it before raw image scoring.",
            "Acquire at least five new complete-baseline clusters and require every held-out root plus a leave-one-cluster-out gain that stays positive when the most responsive system is removed.",
            "Replace Solar proxies with a joint multi-planet ephemeris likelihood before making a Solar-System compatibility claim stronger than provisional.",
            "Report dark-matter comparisons at matched flexibility: universal formula versus both compact-halo baselines and object-specific halo fits, with parameter counts explicit.",
        ],
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(strict_json(report), indent=2) + "\n", encoding="utf-8"
    )

    make_figure(
        galaxy_scores,
        galaxies,
        clusters,
        phases,
        parameters,
        leave_one_out,
        output / protocol["outputs"]["figure"],
    )
    summary = (
        "# P0622 comprehensive regime diagnostics\n\n"
        f"- Galaxy outer holdout: **{overall.P0554_RMSE_km_s:.3f} km/s** versus "
        f"fixed RAR **{overall.fixed_RAR_RMSE_km_s:.3f} km/s** "
        f"(**{overall.P0554_to_RAR_ratio:.3f}x**).\n"
        f"- Dwarf-to-giant bias changes sign: **{mass.loc['dwarf_mass'].P0554_mean_residual_km_s:+.3f}** "
        f"to **{mass.loc['giant_mass'].P0554_mean_residual_km_s:+.3f} km/s**.\n"
        f"- Shared +90-degree cluster phase: **{100*phase_cluster.phase90_improvement_fraction.mean():+.3f}%** "
        f"mean, but **{100*no_rxj.mean_phase90_improvement_fraction:+.3f}%** without RXJ2129; "
        f"only **{np.sum(phase_cluster.phase90_improvement_fraction >= 0.0)}/5** systems improve.\n"
        f"- Raw validation error remains **{raw_halo_ratio:.3f}x** the limited compact-halo comparator.\n"
        f"- Solar analytic proxies pass: **{solar_pass}**.\n"
        "- Main discriminator: route amplitude is baryon-derived, but the beneficial angular direction is not.\n"
        "- Formula promoted: **False**.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(strict_json(report["headline_scores"]), indent=2))


if __name__ == "__main__":
    main()
