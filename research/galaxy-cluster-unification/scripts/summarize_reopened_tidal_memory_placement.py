#!/usr/bin/env python3
"""Summarize where a measured tidal gate has the most cross-domain leverage."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/reopened_hybrid_tidal_memory_placement_protocol.json"
REPORT = ROOT / "results/reopened_hybrid_tidal_memory_placement/report.json"
SCORES = ROOT / "results/reopened_hybrid_tidal_memory_placement/scores.csv"
ROBUST_PROTOCOL = ROOT / (
    "configs/reopened_hybrid_tidal_memory_placement_raw_robustness_protocol.json"
)
ROBUST_REPORT = ROOT / (
    "results/reopened_hybrid_tidal_memory_placement_raw_robustness/report.json"
)
OUTPUT = ROOT / "results/reopened_hybrid_tidal_memory_placement_analysis"
PRIOR_GLOBAL_BEST = {
    "variant": "factorial_q_9_e_6_a_1:radial_memory_log_scale=0.35",
    "stage": "endpoint_interaction_factorial",
    "SPARC_outer_RMSE_km_s": 37.12039332000713,
    "bridge_RMSE_dex": 0.1913917697605262,
    "raw_eight_start_RMS_arcsec": 27.938543909348322,
    "cross_domain_reference_ratio": 3.475197964710249,
}


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
            f"{metric} is not one complete balanced factorial: "
            f"rows={len(data)} expected={expected}"
        )
    values = data[metric].to_numpy(float)
    grand = float(np.mean(values))
    total = float(np.sum(np.square(values - grand)))
    level_means = {}
    main = []
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
            main, key=lambda row: row["sum_squares"], reverse=True
        ),
        "pairwise_interactions_ranked": sorted(
            pairwise, key=lambda row: row["sum_squares"], reverse=True
        ),
        "unassigned_higher_order_fraction": (
            max(0.0, 1.0 - assigned / total) if total else 0.0
        ),
    }


def row_dict(row: pd.Series) -> dict:
    return {name: row[name] for name in row.index}


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    robustness = json.loads(ROBUST_REPORT.read_text(encoding="utf-8"))
    scores = pd.read_csv(SCORES)
    settings = {
        name: result["variant"]["settings"]
        for name, result in report["results"].items()
    }

    def placement(setting: dict) -> str:
        cap = bool(setting.get("rg_cap_gate_enabled", True))
        memory = str(setting.get("radial_memory_gate_mode", "none")) != "none"
        if cap and memory:
            return "both"
        if cap:
            return "cap_only"
        if memory:
            return "memory_only"
        return "global_control"

    scores["placement"] = scores.variant.map(
        lambda name: placement(settings[name])
    )
    scores["cap_orientation"] = scores.variant.map(
        lambda name: (
            "high"
            if bool(settings[name].get("rg_cap_cluster_weight", True))
            else "low"
        )
    )
    scores["memory_orientation"] = scores.variant.map(
        lambda name: {
            "channel": "high",
            "complement": "low",
            "none": "global",
        }[str(settings[name].get("radial_memory_gate_mode", "none"))]
    )
    scores["sharpness"] = scores.variant.map(
        lambda name: float(settings[name].get("channel_gate_sharpness", 5.0))
    )
    scores["maximum_memory_strength"] = scores.variant.map(
        lambda name: float(settings[name].get("radial_memory_strength", 0.0))
    )
    scores["locked_baseline_SPARC_RMSE_km_s"] = scores.variant.map(
        lambda name: report["results"][name][
            "locked_baseline_parameter_sensitivity"
        ]["SPARC"]["RMSE_km_s"]
    )
    scores["locked_baseline_bridge_RMSE_dex"] = scores.variant.map(
        lambda name: report["results"][name][
            "locked_baseline_parameter_sensitivity"
        ]["bridge"]["equal_domain_RMSE_dex"]
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
    both = scores[scores.placement == "both"].copy()
    factors = [
        "cap_orientation",
        "memory_orientation",
        "sharpness",
        "maximum_memory_strength",
    ]
    metrics = [
        "SPARC_outer_RMSE_km_s",
        "bridge_RMSE_dex",
        "raw_eight_start_RMS_arcsec",
        "solar_maximum_fractional_change",
    ]
    effects = {
        metric: balanced_effects(both, metric, factors) for metric in metrics
    }
    locked_effects = {
        metric: balanced_effects(both, metric, factors)
        for metric in [
            "locked_baseline_SPARC_RMSE_km_s",
            "locked_baseline_bridge_RMSE_dex",
        ]
    }

    global_controls = scores[
        scores.placement == "global_control"
    ].set_index("maximum_memory_strength")
    global_full = global_controls.loc[1.0]
    matched_contrasts = []
    for _, row in scores[scores.placement != "global_control"].iterrows():
        reference = global_controls.loc[row.maximum_memory_strength]
        matched_contrasts.append(
            {
                "variant": row.variant,
                "placement": row.placement,
                "cap_orientation": row.cap_orientation,
                "memory_orientation": row.memory_orientation,
                "sharpness": float(row.sharpness),
                "maximum_memory_strength": float(
                    row.maximum_memory_strength
                ),
                "reference_variant": reference.variant,
                "delta_SPARC_RMSE_km_s": float(
                    row.SPARC_outer_RMSE_km_s
                    - reference.SPARC_outer_RMSE_km_s
                ),
                "delta_bridge_RMSE_dex": float(
                    row.bridge_RMSE_dex - reference.bridge_RMSE_dex
                ),
                "delta_raw_eight_start_RMS_arcsec": float(
                    row.raw_eight_start_RMS_arcsec
                    - reference.raw_eight_start_RMS_arcsec
                ),
            }
        )

    prior_best = PRIOR_GLOBAL_BEST
    reconciliation = []
    for _, row in stable.iterrows():
        reconciliation.append(
            {
                "variant": row.variant,
                "placement": row.placement,
                "improves_global_control_galaxy": bool(
                    row.SPARC_outer_RMSE_km_s
                    < global_full.SPARC_outer_RMSE_km_s
                ),
                "improves_global_control_raw": bool(
                    row.raw_eight_start_RMS_arcsec
                    < global_full.raw_eight_start_RMS_arcsec
                ),
                "improves_both_global_control_domains": bool(
                    row.SPARC_outer_RMSE_km_s
                    < global_full.SPARC_outer_RMSE_km_s
                    and row.raw_eight_start_RMS_arcsec
                    < global_full.raw_eight_start_RMS_arcsec
                ),
                "beats_prior_global_ratio": bool(
                    row.robust_cross_domain_reference_ratio
                    < prior_best["cross_domain_reference_ratio"]
                ),
                "meets_both_external_references": bool(
                    row.SPARC_outer_RMSE_km_s
                    <= references["SPARC_fixed_RAR_outer_RMSE_km_s"]
                    and row.raw_eight_start_RMS_arcsec
                    <= references["raw_compact_halo_RMS_arcsec"]
                ),
            }
        )
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

    parameter_names = protocol["universal_parameters"]["names"]
    parameter_rows = []
    for _, row in scores.iterrows():
        result = report["results"][row.variant]
        parameter_rows.append(
            {
                "placement": row.placement,
                "variant": row.variant,
                **{
                    name: float(result["full_fit_parameters"][name])
                    for name in parameter_names
                },
                "any_boundary": any(result["full_fit_at_boundary"].values()),
            }
        )
    parameter_frame = pd.DataFrame(parameter_rows)
    parameter_summaries = {}
    for name, block in parameter_frame.groupby("placement"):
        parameter_summaries[name] = {
            "rows": len(block),
            "boundary_rows": int(block.any_boundary.sum()),
            "parameters": {
                parameter: {
                    "minimum": float(block[parameter].min()),
                    "median": float(block[parameter].median()),
                    "maximum": float(block[parameter].max()),
                }
                for parameter in parameter_names
            },
        }

    output = {
        "status": "completed tidal memory-placement analysis",
        "formula": {
            "gate": "w=sigmoid[k(x-0.685)]",
            "cap_high": "RG cap weight = w",
            "cap_low": "RG cap weight = 1-w",
            "memory_high": "mu_eff = mu_max w",
            "memory_low": "mu_eff = mu_max (1-w)",
            "carrier": "X=F (g_N/g_ref)^1.927395 (r/kpc)^9, memory length=0.35",
        },
        "coverage": {
            "universal_refits": len(scores),
            "both_placement_factorial_cells": len(both),
            "matched_control_rows": len(scores) - len(both),
            "eight_start_raw_replays": len(robustness["comparisons"]),
            "stable_root_complete_replays": int(stable.shape[0]),
        },
        "references": references,
        "prior_global_best": prior_best,
        "best_stable_observed": row_dict(best),
        "best_stable_raw_case": row_dict(best_raw),
        "global_memory_controls": {
            str(strength): row_dict(row)
            for strength, row in global_controls.iterrows()
        },
        "balanced_effects": effects,
        "locked_baseline_parameter_effects": locked_effects,
        "matched_control_contrasts": matched_contrasts,
        "reconciliation": reconciliation,
        "rows_improving_both_global_control_domains": sum(
            row["improves_both_global_control_domains"]
            for row in reconciliation
        ),
        "rows_beating_prior_global_ratio": sum(
            row["beats_prior_global_ratio"] for row in reconciliation
        ),
        "rows_meeting_both_external_references": sum(
            row["meets_both_external_references"] for row in reconciliation
        ),
        "placement_parameter_summaries": parameter_summaries,
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
        "# Tidal gate-placement analysis",
        "",
        "The same measured coordinate was applied to the RG ceiling, radial-memory strength, both, or neither.",
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
        f"- Rows improving both domains over the global-memory control: {output['rows_improving_both_global_control_domains']}",
        f"- Rows beating the prior global ratio: {output['rows_beating_prior_global_ratio']}",
        f"- Rows meeting both external references: {output['rows_meeting_both_external_references']}",
        "",
        "## Main-effect ranking: both-placement factorial",
        "",
        "| metric | first | fraction | second | fraction |",
        "|---|---|---:|---|---:|",
    ]
    for metric, analysis in effects.items():
        ranked = analysis["main_effects_ranked"]
        lines.append(
            f"| {metric} | {ranked[0]['factor']} | {ranked[0]['fraction_of_total_variation']:.1%} | "
            f"{ranked[1]['factor']} | {ranked[1]['fraction_of_total_variation']:.1%} |"
        )
    lines.extend(
        [
            "",
            "Variance fractions apply only to the frozen 2x2x2x2 grid. The cluster-side coordinate remains a spherical density-ratio proxy.",
        ]
    )
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            json_safe(
                {
                    "best": output["best_stable_observed"],
                    "best_raw": output["best_stable_raw_case"],
                    "coverage": output["coverage"],
                    "improves_both": output[
                        "rows_improving_both_global_control_domains"
                    ],
                    "beats_prior": output["rows_beating_prior_global_ratio"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
