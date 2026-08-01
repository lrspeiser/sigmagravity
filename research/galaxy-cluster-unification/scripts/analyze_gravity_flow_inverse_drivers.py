"""Exploratory driver analysis for gravity-flow inverse routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gravity_flow_inverse"


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def jackknife_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    values = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        values.append(float(spearmanr(x[keep], y[keep]).statistic))
    array = np.asarray(values)
    full_sign = np.sign(spearmanr(x, y).statistic)
    return (
        float(np.min(array)),
        float(np.median(array)),
        float(np.max(array)),
        float(np.mean(np.sign(array) == full_sign)),
    )


def source_environment(block: pd.DataFrame) -> pd.DataFrame:
    ordered = block.sort_values("source_index").copy()
    position = ordered[["source_x_kpc", "source_y_kpc"]].to_numpy(float)
    weight = ordered.source_weight.to_numpy(float)
    center = np.sum(position * weight[:, None], axis=0)
    delta = position[None, :, :] - position[:, None, :]
    distance2 = np.sum(np.square(delta), axis=2)
    np.fill_diagonal(distance2, np.inf)
    softening2 = 20.0**2
    potential = np.sum(weight[None, :] / np.sqrt(distance2 + softening2), axis=1)
    field = np.sum(
        weight[None, :, None]
        * delta
        / np.power(distance2[:, :, None] + softening2, 1.5),
        axis=1,
    )
    ordered["source_radius_from_light_centroid_kpc"] = np.linalg.norm(
        position - center[None, :], axis=1
    )
    ordered["log10_source_weight"] = np.log10(np.maximum(weight, 1e-12))
    ordered["local_potential_proxy"] = potential
    ordered["external_field_strength_proxy"] = np.linalg.norm(field, axis=1)
    ordered["nearest_neighbor_kpc"] = np.sqrt(np.min(distance2, axis=1))
    return ordered


def main():
    routes = pd.read_csv(RESULTS / "route_statistics.csv")
    routes = routes[
        routes.destination_kind.eq("local_projected_excess")
        & routes.entropy_length_kpc.eq(50.0)
    ].copy()
    morphology = pd.read_csv(
        ROOT / "results" / "gravity_arc_fresh_sample" / "source_morphology.csv"
    )
    nulls = pd.read_csv(RESULTS / "radial_shuffle_summary.csv")
    cluster = routes.merge(morphology, on="system", validate="many_to_one").merge(
        nulls[
            [
                "system",
                "target_kind",
                "improvement_over_shuffle_median_fraction",
                "one_sided_permutation_p",
            ]
        ],
        on=["system", "target_kind"],
        validate="one_to_one",
    )
    feature_columns = [
        "hard_member_count",
        "hard_effective_source_count",
        "brightest_light_fraction",
        "mean_radius_kpc",
        "r50_kpc",
        "r80_kpc",
        "radial_concentration_r50_over_r80",
        "baryonic_centroid_offset_kpc",
        "axis_ratio",
        "angular_dipole",
        "angular_quadrupole",
    ]
    response_columns = [
        "median_path_kpc",
        "p90_path_kpc",
        "mean_cos_inward",
        "fraction_ending_inward",
        "positive_target_weight_removed",
        "improvement_over_shuffle_median_fraction",
    ]
    cluster_rows = []
    for target_kind, method_block in cluster.groupby("target_kind"):
        for feature in feature_columns:
            for response in response_columns:
                x = method_block[feature].to_numpy(float)
                y = method_block[response].to_numpy(float)
                result = spearmanr(x, y)
                low, median, high, stable = jackknife_spearman(x, y)
                cluster_rows.append(
                    {
                        "target_kind": target_kind,
                        "feature": feature,
                        "response": response,
                        "spearman_rho": float(result.statistic),
                        "two_sided_p_unadjusted": float(result.pvalue),
                        "jackknife_rho_min": low,
                        "jackknife_rho_median": median,
                        "jackknife_rho_max": high,
                        "jackknife_same_sign_fraction": stable,
                    }
                )
    cluster_correlations = pd.DataFrame(cluster_rows)
    cluster_correlations["benjamini_hochberg_q"] = benjamini_hochberg(
        cluster_correlations.two_sided_p_unadjusted.to_numpy(float)
    )
    cluster_correlations["absolute_rho"] = np.abs(cluster_correlations.spearman_rho)
    cluster_correlations = cluster_correlations.sort_values(
        ["benjamini_hochberg_q", "absolute_rho"], ascending=[True, False]
    )
    cluster_correlations.to_csv(RESULTS / "cluster_driver_correlations.csv", index=False)
    cluster.to_csv(RESULTS / "cluster_driver_table.csv", index=False)

    source_routes = pd.read_csv(RESULTS / "source_routes.csv")
    enriched = []
    within_rows = []
    source_features = [
        "source_radius_from_light_centroid_kpc",
        "log10_source_weight",
        "local_potential_proxy",
        "external_field_strength_proxy",
        "nearest_neighbor_kpc",
    ]
    source_responses = [
        "conditional_mean_path_kpc",
        "expected_displacement_kpc",
        "expected_direction_cos_inward",
    ]
    for (system, target_kind), block in source_routes.groupby(
        ["system", "target_kind"], sort=False
    ):
        local = source_environment(block)
        enriched.append(local)
        for feature in source_features:
            for response in source_responses:
                result = spearmanr(local[feature], local[response])
                within_rows.append(
                    {
                        "system": system,
                        "target_kind": target_kind,
                        "feature": feature,
                        "response": response,
                        "spearman_rho": float(result.statistic),
                        "two_sided_p_unadjusted": float(result.pvalue),
                        "sources": len(local),
                    }
                )
    enriched_frame = pd.concat(enriched, ignore_index=True)
    within = pd.DataFrame(within_rows)
    aggregate_rows = []
    for (target_kind, feature, response), block in within.groupby(
        ["target_kind", "feature", "response"]
    ):
        rho = block.spearman_rho.to_numpy(float)
        positives = int(np.sum(rho > 0.0))
        negatives = int(np.sum(rho < 0.0))
        nonzero = positives + negatives
        sign_p = 1.0 if nonzero == 0 else float(
            binomtest(max(positives, negatives), nonzero, 0.5, alternative="greater").pvalue
        )
        aggregate_rows.append(
            {
                "target_kind": target_kind,
                "feature": feature,
                "response": response,
                "median_within_system_rho": float(np.median(rho)),
                "rho_p16": float(np.quantile(rho, 0.16)),
                "rho_p84": float(np.quantile(rho, 0.84)),
                "positive_systems": positives,
                "negative_systems": negatives,
                "same_sign_fraction": float(max(positives, negatives) / max(nonzero, 1)),
                "two_sided_sign_consistency_p": min(1.0, 2.0 * sign_p),
            }
        )
    source_aggregate = pd.DataFrame(aggregate_rows)
    source_aggregate["absolute_median_rho"] = np.abs(
        source_aggregate.median_within_system_rho
    )
    source_aggregate = source_aggregate.sort_values(
        ["same_sign_fraction", "absolute_median_rho"], ascending=[False, False]
    )
    enriched_frame.to_csv(RESULTS / "source_route_features.csv", index=False)
    within.to_csv(RESULTS / "source_driver_correlations_by_system.csv", index=False)
    source_aggregate.to_csv(RESULTS / "source_driver_correlations.csv", index=False)

    discoveries = cluster_correlations[
        cluster_correlations.benjamini_hochberg_q <= 0.05
    ]
    report = {
        "status": "completed exploratory inverse-route driver analysis",
        "cluster_level_tests": int(len(cluster_correlations)),
        "cluster_level_fdr_discoveries_q_le_0_05": int(len(discoveries)),
        "top_cluster_correlations": cluster_correlations.head(12).to_dict("records"),
        "top_source_within_system_patterns": source_aggregate.head(12).to_dict("records"),
        "interpretation_limit": "These ten clusters already generated the inverse paths. Driver relationships are post hoc and may propose, but cannot validate, a forward kernel.",
    }
    (RESULTS / "driver_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
