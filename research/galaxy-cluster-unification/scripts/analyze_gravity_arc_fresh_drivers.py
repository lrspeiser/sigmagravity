#!/usr/bin/env python3
"""Relate fresh gravity-arc performance to target-blind baryonic morphology."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "gravity_arc_fresh_sample_input_audit"
RESULTS = ROOT / "results" / "gravity_arc_fresh_sample"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(quantile, cumulative, values[order]))


def source_features(sources: pd.DataFrame, systems: pd.DataFrame) -> pd.DataFrame:
    records = []
    for system, group in sources.groupby("system", sort=False):
        hard = group.hard_member.astype(str).str.lower().eq("true")
        group = group[hard]
        positions = group[["x_kpc", "y_kpc"]].to_numpy(float)
        radius = np.linalg.norm(positions, axis=1)
        weight = np.maximum(group.f160w_flux_nJy.to_numpy(float), 0.0)
        weight /= np.sum(weight)
        centroid = np.sum(weight[:, None] * positions, axis=0)
        centered = positions - centroid
        covariance = np.einsum("n,ni,nj->ij", weight, centered, centered)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        axis_ratio = float(
            np.sqrt(eigenvalues[0] / max(eigenvalues[-1], np.finfo(float).tiny))
        )
        unit = positions / np.maximum(radius[:, None], np.finfo(float).tiny)
        angular_dipole = float(np.linalg.norm(np.sum(weight[:, None] * unit, axis=0)))
        angle = np.arctan2(positions[:, 1], positions[:, 0])
        angular_quadrupole = float(abs(np.sum(weight * np.exp(2j * angle))))
        r50 = weighted_quantile(radius, weight, 0.5)
        r80 = weighted_quantile(radius, weight, 0.8)
        audit = systems.loc[system]
        records.append(
            {
                "system": system,
                "redshift": float(audit.cluster_redshift),
                "hard_member_count": int(len(group)),
                "hard_effective_source_count": float(audit.hard_effective_source_count),
                "brightest_light_fraction": float(np.max(weight)),
                "mean_radius_kpc": float(np.sum(weight * radius)),
                "r50_kpc": r50,
                "r80_kpc": r80,
                "radial_concentration_r50_over_r80": r50 / max(r80, np.finfo(float).tiny),
                "baryonic_centroid_offset_kpc": float(np.linalg.norm(centroid)),
                "axis_ratio": axis_ratio,
                "ellipticity": 1.0 - axis_ratio,
                "angular_dipole": angular_dipole,
                "angular_quadrupole": angular_quadrupole,
            }
        )
    return pd.DataFrame(records)


def response_table(comparisons: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    response = comparisons[
        comparisons.target_kind.isin(["lenstool_ensemble_mean", "glafic_best"])
    ].pivot(index="system", columns="target_kind")
    records = []
    for system in response.index:
        record = {"system": system}
        for kind, prefix in [
            ("lenstool_ensemble_mean", "lenstool"),
            ("glafic_best", "glafic"),
        ]:
            row = comparisons[
                comparisons.system.eq(system) & comparisons.target_kind.eq(kind)
            ].iloc[0]
            record[f"{prefix}_arc_over_local"] = float(
                row.improvement_over_local_fraction
            )
            record[f"{prefix}_arc_over_central"] = float(
                row.improvement_over_central_fraction
            )
            record[f"{prefix}_arc_over_best_null"] = float(
                row.improvement_over_best_null_fraction
            )
            block = scores[scores.system.eq(system) & scores.target_kind.eq(kind)].set_index(
                "candidate_id"
            )
            record[f"{prefix}_W060_delta_JS"] = float(
                block.loc["W060", "jensen_shannon"]
                - block.loc["C0351", "jensen_shannon"]
            )
        records.append(record)
    return pd.DataFrame(records)


def candidate_aggregates(scores: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (target_kind, candidate_id), block in scores.groupby(
        ["target_kind", "candidate_id"], sort=False
    ):
        baseline = scores[
            scores.target_kind.eq(target_kind)
            & scores.system.isin(block.system)
            & scores.candidate_id.eq("C0351")
        ].set_index("system")
        local = scores[
            scores.target_kind.eq(target_kind)
            & scores.system.isin(block.system)
            & scores.candidate_id.eq("LOCAL75")
        ].set_index("system")
        central = scores[
            scores.target_kind.eq(target_kind)
            & scores.system.isin(block.system)
            & scores.candidate_id.eq("CENTRAL100")
        ].set_index("system")
        values = block.set_index("system")
        records.append(
            {
                "target_kind": target_kind,
                "candidate_id": candidate_id,
                "role": block.role.iloc[0],
                "changed_parameter": block.changed_parameter.iloc[0],
                "median_JS": float(values.jensen_shannon.median()),
                "mean_JS": float(values.jensen_shannon.mean()),
                "median_Pearson": float(values.pearson.median()),
                "median_delta_JS_vs_C0351": float(
                    (values.jensen_shannon - baseline.jensen_shannon).median()
                ),
                "median_improvement_over_LOCAL75": float(
                    (1.0 - values.jensen_shannon / local.jensen_shannon).median()
                ),
                "clusters_better_than_LOCAL75": int(
                    np.sum(values.jensen_shannon < local.jensen_shannon)
                ),
                "median_improvement_over_CENTRAL100": float(
                    (1.0 - values.jensen_shannon / central.jensen_shannon).median()
                ),
                "clusters_better_than_CENTRAL100": int(
                    np.sum(values.jensen_shannon < central.jensen_shannon)
                ),
            }
        )
    return pd.DataFrame(records)


def parameter_impact_ranking(impacts: pd.DataFrame) -> pd.DataFrame:
    records = []
    for parameter, block in impacts.groupby("changed_parameter", sort=False):
        lenstool = np.concatenate([[0.0], block.lenstool_median_delta_JS.to_numpy(float)])
        glafic = np.concatenate([[0.0], block.glafic_median_delta_JS.to_numpy(float)])
        records.append(
            {
                "parameter": parameter,
                "variants": ",".join(block.candidate_id),
                "lenstool_median_JS_span": float(np.max(lenstool) - np.min(lenstool)),
                "glafic_median_JS_span": float(np.max(glafic) - np.min(glafic)),
                "largest_absolute_lenstool_shift": float(np.max(np.abs(lenstool))),
                "largest_absolute_glafic_shift": float(np.max(np.abs(glafic))),
            }
        )
    return pd.DataFrame(records).sort_values(
        "lenstool_median_JS_span", ascending=False
    )


def main() -> None:
    sources = pd.read_csv(AUDIT / "sources.csv")
    systems = pd.read_csv(AUDIT / "systems.csv").set_index("system")
    comparisons = pd.read_csv(RESULTS / "locked_comparisons.csv")
    scores = pd.read_csv(RESULTS / "scores.csv")
    disagreement = pd.read_csv(RESULTS / "method_disagreement.csv")[
        ["system", "jensen_shannon"]
    ].rename(columns={"jensen_shannon": "method_disagreement_JS"})
    features = source_features(sources, systems)
    responses = response_table(comparisons, scores).merge(
        disagreement, on="system", validate="one_to_one"
    )
    table = features.merge(responses, on="system", validate="one_to_one")
    feature_columns = [name for name in features if name != "system"]
    response_columns = [name for name in responses if name != "system"]
    records = []
    for feature in feature_columns:
        for response in response_columns:
            rho, p_value = spearmanr(table[feature], table[response])
            jackknife = []
            for held_out in range(len(table)):
                keep = np.arange(len(table)) != held_out
                value, _ = spearmanr(table.loc[keep, feature], table.loc[keep, response])
                jackknife.append(float(value))
            records.append(
                {
                    "feature": feature,
                    "response": response,
                    "spearman_rho": float(rho),
                    "two_sided_p_value_unadjusted": float(p_value),
                    "systems": len(table),
                    "absolute_rho": abs(float(rho)),
                    "jackknife_rho_min": float(np.min(jackknife)),
                    "jackknife_rho_median": float(np.median(jackknife)),
                    "jackknife_rho_max": float(np.max(jackknife)),
                    "jackknife_same_sign_fraction": float(
                        np.mean(np.sign(jackknife) == np.sign(float(rho)))
                    ),
                }
            )
    correlations = pd.DataFrame(records)
    order = np.argsort(correlations.two_sided_p_value_unadjusted.to_numpy(float))
    ordered_p = correlations.two_sided_p_value_unadjusted.to_numpy(float)[order]
    raw_q = ordered_p * len(ordered_p) / np.arange(1, len(ordered_p) + 1)
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_values = np.empty_like(monotone_q)
    q_values[order] = np.clip(monotone_q, 0.0, 1.0)
    correlations["benjamini_hochberg_q"] = q_values
    correlations = correlations.sort_values(
        ["absolute_rho", "feature", "response"], ascending=[False, True, True]
    )
    impacts = pd.read_csv(RESULTS / "variant_impacts.csv")
    aggregates = candidate_aggregates(scores)
    parameter_ranking = parameter_impact_ranking(impacts)
    features.to_csv(RESULTS / "source_morphology.csv", index=False)
    table.to_csv(RESULTS / "driver_table.csv", index=False)
    correlations.to_csv(RESULTS / "driver_correlations.csv", index=False)
    aggregates.to_csv(RESULTS / "candidate_aggregate.csv", index=False)
    parameter_ranking.to_csv(RESULTS / "parameter_impact_ranking.csv", index=False)
    report = {
        "status": "completed exploratory post-confirmation driver analysis",
        "systems": len(table),
        "features": len(feature_columns),
        "responses": len(response_columns),
        "correlations_tested": len(correlations),
        "fdr_discoveries_q_le_0_05": int(
            np.sum(correlations.benjamini_hochberg_q <= 0.05)
        ),
        "parameter_impact_ranking": parameter_ranking.to_dict("records"),
        "top_correlations": correlations.head(12).to_dict("records"),
        "interpretation_limit": (
            "All driver correlations are post-confirmation and based on ten systems. False-discovery "
            "rate adjustments and leave-one-system-out ranges are reported; they generate hypotheses "
            "and cannot select a confirmed formula."
        ),
    }
    (RESULTS / "driver_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
