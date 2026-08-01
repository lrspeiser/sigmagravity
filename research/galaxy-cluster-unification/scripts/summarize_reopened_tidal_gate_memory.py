#!/usr/bin/env python3
"""Summarize the balanced tidal-gate-memory experiments and robust lensing."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/reopened_hybrid_tidal_gate_memory_protocol.json"
REPORT = ROOT / "results/reopened_hybrid_tidal_gate_memory/report.json"
SCORES = ROOT / "results/reopened_hybrid_tidal_gate_memory/scores.csv"
ROBUST_PROTOCOL = (
    ROOT
    / "configs/reopened_hybrid_tidal_gate_memory_raw_robustness_protocol.json"
)
ROBUST_REPORT = (
    ROOT
    / "results/reopened_hybrid_tidal_gate_memory_raw_robustness/report.json"
)
OUTPUT = ROOT / "results/reopened_hybrid_tidal_gate_memory_analysis"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def balanced_effects(
    frame: pd.DataFrame, metric: str, factors: list[str]
) -> dict:
    data = frame[[*factors, metric]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    expected = math.prod(data[factor].nunique() for factor in factors)
    if len(data) != expected or data.groupby(factors).size().max() != 1:
        raise RuntimeError(
            f"{metric} design is not one complete balanced factorial: "
            f"rows={len(data)} expected={expected}"
        )
    values = data[metric].to_numpy(float)
    grand = float(np.mean(values))
    total = float(np.sum(np.square(values - grand)))
    main = []
    level_means = {}
    for factor in factors:
        means = data.groupby(factor)[metric].mean()
        counts = data.groupby(factor)[metric].size()
        ss = float(
            sum(
                counts.loc[level] * (means.loc[level] - grand) ** 2
                for level in means.index
            )
        )
        level_means[factor] = means.to_dict()
        main.append(
            {
                "factor": factor,
                "sum_squares": ss,
                "fraction_of_total_variation": ss / total if total else 0.0,
                "level_mean_span": float(means.max() - means.min()),
                "level_means": means.to_dict(),
            }
        )
    pairwise = []
    for first, second in combinations(factors, 2):
        grouped = data.groupby([first, second])[metric].mean()
        counts = data.groupby([first, second])[metric].size()
        residuals = []
        ss = 0.0
        for levels, cell_mean in grouped.items():
            residual = (
                cell_mean
                - level_means[first][levels[0]]
                - level_means[second][levels[1]]
                + grand
            )
            residuals.append(float(residual))
            ss += counts.loc[levels] * residual**2
        pairwise.append(
            {
                "factors": [first, second],
                "sum_squares": float(ss),
                "fraction_of_total_variation": ss / total if total else 0.0,
                "maximum_absolute_interaction_residual": float(
                    np.max(np.abs(residuals))
                ),
            }
        )
    assigned = sum(row["sum_squares"] for row in main + pairwise)
    return {
        "metric": metric,
        "rows": len(data),
        "grand_mean": grand,
        "total_sum_squares": total,
        "main_effects_ranked": sorted(
            main,
            key=lambda row: row["sum_squares"],
            reverse=True,
        ),
        "pairwise_interactions_ranked": sorted(
            pairwise,
            key=lambda row: row["sum_squares"],
            reverse=True,
        ),
        "unassigned_higher_order_fraction": (
            max(0.0, 1.0 - assigned / total) if total else 0.0
        ),
    }


def repeated_prediction_ranges(
    scores: pd.DataFrame, report: dict, orientation: str
) -> dict:
    block = scores[
        (scores.orientation == orientation) & (scores.memory_strength == 0.0)
    ].copy()
    parameters = []
    for name in block.variant:
        parameters.append(report["results"][name]["full_fit_parameters"])
    parameter_frame = pd.DataFrame(parameters)
    return {
        "copies": len(block),
        "stable_root_complete_copies": int(block.raw_eight_start_all_roots.sum()),
        "SPARC_span_km_s": float(block.SPARC_outer_RMSE_km_s.max() - block.SPARC_outer_RMSE_km_s.min()),
        "bridge_span_dex": float(block.bridge_RMSE_dex.max() - block.bridge_RMSE_dex.min()),
        "raw_eight_start_span_arcsec": float(block.raw_eight_start_RMS_arcsec.max() - block.raw_eight_start_RMS_arcsec.min()),
        "median_cross_domain_reference_ratio": float(block.robust_cross_domain_reference_ratio.median()),
        "universal_parameter_ranges": {
            column: {
                "minimum": float(parameter_frame[column].min()),
                "median": float(parameter_frame[column].median()),
                "maximum": float(parameter_frame[column].max()),
                "span": float(parameter_frame[column].max() - parameter_frame[column].min()),
            }
            for column in parameter_frame
        },
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    robustness = json.loads(ROBUST_REPORT.read_text(encoding="utf-8"))
    scores = pd.read_csv(SCORES)
    settings = {
        name: row["variant"]["settings"]
        for name, row in report["results"].items()
    }
    scores["orientation"] = scores.variant.map(
        lambda name: "cluster_high"
        if bool(settings[name]["channel_gate_cluster_high"])
        else "cluster_low"
    )
    scores["memory_direction"] = scores.variant.map(
        lambda name: "outer_to_inner"
        if bool(settings[name]["channel_gate_memory_outer_to_inner"])
        else "inner_to_outer"
    )
    scores["memory_log_scale"] = scores.variant.map(
        lambda name: float(settings[name]["channel_gate_memory_log_scale"])
    )
    scores["memory_strength"] = scores.variant.map(
        lambda name: float(settings[name]["channel_gate_memory_strength"])
    )
    scores["raw_eight_start_RMS_arcsec"] = scores.variant.map(
        lambda name: robustness["comparisons"][name]["eight_start"][
            "equal_system_radial_RMS_arcsec"
        ]
    )
    scores["raw_eight_start_all_roots"] = scores.variant.map(
        lambda name: bool(
            robustness["comparisons"][name]["eight_start"][
                "all_roots_converged"
            ]
        )
    )
    scores["raw_eight_start_pooled_reduced_chi2"] = scores.variant.map(
        lambda name: robustness["comparisons"][name]["eight_start"][
            "pooled_reduced_chi2"
        ]
    )
    references = report["references"]
    scores["robust_cross_domain_reference_ratio"] = np.maximum(
        scores.SPARC_outer_RMSE_km_s
        / references["SPARC_fixed_RAR_outer_RMSE_km_s"],
        scores.raw_eight_start_RMS_arcsec
        / references["raw_compact_halo_RMS_arcsec"],
    )
    stable = scores[scores.raw_eight_start_all_roots].copy()
    best = stable.sort_values(
        ["robust_cross_domain_reference_ratio", "bridge_RMSE_dex"]
    ).iloc[0]
    best_raw = stable.sort_values(
        ["raw_eight_start_RMS_arcsec", "SPARC_outer_RMSE_km_s"]
    ).iloc[0]
    metrics = [
        "SPARC_outer_RMSE_km_s",
        "bridge_RMSE_dex",
        "raw_eight_start_RMS_arcsec",
        "solar_maximum_fractional_change",
    ]
    inner = scores[scores.memory_direction == "inner_to_outer"].copy()
    direction = scores[scores.memory_log_scale == 0.35].copy()
    inner_effects = {
        metric: balanced_effects(
            inner, metric, ["orientation", "memory_log_scale", "memory_strength"]
        )
        for metric in metrics
    }
    direction_effects = {
        metric: balanced_effects(
            direction, metric, ["orientation", "memory_direction", "memory_strength"]
        )
        for metric in metrics
    }
    reversals = [
        {
            "variant": name,
            "two_start_all_roots": bool(
                comparison["two_start"]["all_roots_converged"]
            ),
            "eight_start_all_roots": bool(
                comparison["eight_start"]["all_roots_converged"]
            ),
        }
        for name, comparison in robustness["comparisons"].items()
        if comparison["two_start"]["all_roots_converged"]
        != comparison["eight_start"]["all_roots_converged"]
    ]
    output = {
        "status": "completed tidal-gate-memory interaction analysis",
        "formula": {
            "local_gate": "w=logistic[o*k*(tidal_middle_to_max-pivot)]",
            "gate_memory": "M_i=exp(-Delta ln r/ell) M_(i-1)+(1-exp(-Delta ln r/ell)) w_i",
            "effective_gate": "w_eff=(1-mu) w+mu M",
            "placement": "w_eff places existing RG/Sigma ceilings before channel combination and Solar screening",
        },
        "coverage": {
            "universal_refits": len(scores),
            "inner_to_outer_factorial_cells": len(inner),
            "direction_factorial_cells": len(direction),
            "eight_start_raw_replays": len(robustness["comparisons"]),
            "stable_root_complete_replays": int(stable.shape[0]),
        },
        "references": references,
        "best_stable_observed": best.to_dict(),
        "best_stable_raw_case": best_raw.to_dict(),
        "inner_to_outer_balanced_effects": inner_effects,
        "direction_balanced_effects_at_log_scale_0p35": direction_effects,
        "exact_local_repeatability": {
            orientation: repeated_prediction_ranges(scores, report, orientation)
            for orientation in ["cluster_high", "cluster_low"]
        },
        "root_reversals": reversals,
        "claim_boundary": protocol["claim_boundary"],
        "input_hashes": {
            "protocol": sha256(PROTOCOL),
            "report": sha256(REPORT),
            "scores": sha256(SCORES),
            "robust_protocol": sha256(ROBUST_PROTOCOL),
            "robust_report": sha256(ROBUST_REPORT),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(output), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Tidal-gate memory analysis",
        "",
        "The remembered quantity is the bounded tidal/channel classification, not force amplitude.",
        "",
        "## Best stable cross-domain row",
        "",
        f"- `{best.variant}`",
        f"- SPARC outer RMSE: {best.SPARC_outer_RMSE_km_s:.3f} km/s",
        f"- Bridge RMSE: {best.bridge_RMSE_dex:.4f} dex",
        f"- Eight-start raw RMS: {best.raw_eight_start_RMS_arcsec:.3f} arcsec",
        f"- Worse-reference ratio: {best.robust_cross_domain_reference_ratio:.3f}",
        "",
        "## Best stable raw-lensing row",
        "",
        f"- `{best_raw.variant}`",
        f"- Raw RMS: {best_raw.raw_eight_start_RMS_arcsec:.3f} arcsec",
        f"- SPARC outer RMSE: {best_raw.SPARC_outer_RMSE_km_s:.3f} km/s",
        "",
        "## Main-effect ranking: inner-to-outer factorial",
        "",
        "| metric | first | fraction | second | fraction |",
        "|---|---|---:|---|---:|",
    ]
    for metric, analysis in inner_effects.items():
        ranked = analysis["main_effects_ranked"]
        lines.append(
            f"| {metric} | {ranked[0]['factor']} | {ranked[0]['fraction_of_total_variation']:.1%} | "
            f"{ranked[1]['factor']} | {ranked[1]['fraction_of_total_variation']:.1%} |"
        )
    lines.extend(
        [
            "",
            "Variance fractions apply only to the frozen sampled grids. The cluster-side tidal coordinate is spherical and therefore not independent directional information.",
        ]
    )
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe({
        "best": output["best_stable_observed"],
        "best_raw": output["best_stable_raw_case"],
        "coverage": output["coverage"],
    }), indent=2))


if __name__ == "__main__":
    main()
