#!/usr/bin/env python3
"""Summarize nonmonotonic tidal-gate topology and robust raw lensing."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/reopened_hybrid_tidal_gate_topology_protocol.json"
REPORT = ROOT / "results/reopened_hybrid_tidal_gate_topology/report.json"
SCORES = ROOT / "results/reopened_hybrid_tidal_gate_topology/scores.csv"
SYSTEM_INDICATORS = (
    ROOT
    / "results/reopened_tidal_shape_indicator_audit/system_tidal_indicator_medians.csv"
)
ROBUST_PROTOCOL = (
    ROOT
    / "configs/reopened_hybrid_tidal_gate_topology_raw_robustness_protocol.json"
)
ROBUST_REPORT = (
    ROOT
    / "results/reopened_hybrid_tidal_gate_topology_raw_robustness/report.json"
)
OUTPUT = ROOT / "results/reopened_hybrid_tidal_gate_topology_analysis"


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


def logistic(value) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -700.0, 700.0)))


def gate_weight(coordinate: np.ndarray, settings: dict) -> np.ndarray:
    topology = str(settings.get("channel_gate_topology", "monotonic"))
    sharpness = float(settings.get("channel_gate_sharpness", 1.0))
    if topology == "monotonic":
        pivot = float(settings.get("channel_gate_pivot", 0.5))
        orientation = 1.0 if settings.get("channel_gate_cluster_high", True) else -1.0
        return logistic(orientation * sharpness * (coordinate - pivot))
    if topology in {"band", "tails"}:
        lower = float(settings["channel_gate_lower_pivot"])
        upper = float(settings["channel_gate_upper_pivot"])
        band = logistic(sharpness * (coordinate - lower)) * logistic(
            sharpness * (upper - coordinate)
        )
        return band if topology == "band" else 1.0 - band
    if topology == "constant":
        return np.full_like(
            coordinate, float(settings["channel_gate_constant_weight"])
        )
    raise ValueError(f"unknown topology {topology}")


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
            main, key=lambda row: row["sum_squares"], reverse=True
        ),
        "pairwise_interactions_ranked": sorted(
            pairwise, key=lambda row: row["sum_squares"], reverse=True
        ),
        "unassigned_higher_order_fraction": (
            max(0.0, 1.0 - assigned / total) if total else 0.0
        ),
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
    scores["topology"] = scores.variant.map(
        lambda name: str(settings[name].get("channel_gate_topology", "monotonic"))
    )
    scores["lower_pivot"] = scores.variant.map(
        lambda name: float(settings[name].get("channel_gate_lower_pivot", np.nan))
    )
    scores["upper_pivot"] = scores.variant.map(
        lambda name: float(settings[name].get("channel_gate_upper_pivot", np.nan))
    )
    scores["sharpness"] = scores.variant.map(
        lambda name: float(settings[name].get("channel_gate_sharpness", np.nan))
    )
    scores["orientation"] = scores.variant.map(
        lambda name: (
            "high"
            if settings[name].get("channel_gate_topology", "monotonic")
            == "monotonic"
            and bool(settings[name].get("channel_gate_cluster_high", True))
            else "low"
            if settings[name].get("channel_gate_topology", "monotonic")
            == "monotonic"
            else "not_applicable"
        )
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
    nonmonotonic = scores[scores.topology.isin(["band", "tails"])].copy()
    factors = ["topology", "lower_pivot", "upper_pivot", "sharpness"]
    metrics = [
        "SPARC_outer_RMSE_km_s",
        "bridge_RMSE_dex",
        "raw_eight_start_RMS_arcsec",
        "solar_maximum_fractional_change",
    ]
    effects = {
        metric: balanced_effects(nonmonotonic, metric, factors)
        for metric in metrics
    }
    locked_baseline_effects = {
        metric: balanced_effects(nonmonotonic, metric, factors)
        for metric in [
            "locked_baseline_SPARC_RMSE_km_s",
            "locked_baseline_bridge_RMSE_dex",
        ]
    }
    complement_pairs = []
    for (lower, upper, sharpness), block in nonmonotonic.groupby(
        ["lower_pivot", "upper_pivot", "sharpness"]
    ):
        by_topology = block.set_index("topology")
        if set(by_topology.index) != {"band", "tails"}:
            raise RuntimeError("missing exact band/tails complement pair")
        complement_pairs.append(
            {
                "lower_pivot": float(lower),
                "upper_pivot": float(upper),
                "sharpness": float(sharpness),
                "tails_minus_band": {
                    metric: float(
                        by_topology.loc["tails", metric]
                        - by_topology.loc["band", metric]
                    )
                    for metric in metrics
                },
            }
        )
    reconciliation = []
    monotonic = scores[scores.topology == "monotonic"].copy()
    for sharpness, block in nonmonotonic.groupby("sharpness"):
        controls = monotonic[monotonic.sharpness == sharpness].set_index(
            "orientation"
        )
        galaxy_control = controls.loc["low"]
        raw_control = controls.loc["high"]
        for _, row in block.iterrows():
            reconciliation.append(
                {
                    "variant": row.variant,
                    "sharpness": float(sharpness),
                    "beats_low_orientation_galaxy": bool(
                        row.SPARC_outer_RMSE_km_s
                        < galaxy_control.SPARC_outer_RMSE_km_s
                    ),
                    "beats_high_orientation_raw": bool(
                        row.raw_eight_start_all_roots
                        and row.raw_eight_start_RMS_arcsec
                        < raw_control.raw_eight_start_RMS_arcsec
                    ),
                    "reconciles_both_endpoints": bool(
                        row.SPARC_outer_RMSE_km_s
                        < galaxy_control.SPARC_outer_RMSE_km_s
                        and row.raw_eight_start_all_roots
                        and row.raw_eight_start_RMS_arcsec
                        < raw_control.raw_eight_start_RMS_arcsec
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
    parameter_rows = []
    parameter_names = protocol["universal_parameters"]["names"]
    for _, row in scores.iterrows():
        result = report["results"][row.variant]
        parameter_rows.append(
            {
                "topology": row.topology,
                "variant": row.variant,
                **{
                    name: float(result["full_fit_parameters"][name])
                    for name in parameter_names
                },
                "any_boundary": any(
                    result["full_fit_at_boundary"].values()
                ),
            }
        )
    parameter_frame = pd.DataFrame(parameter_rows)
    parameter_summaries = {}
    for topology, block in parameter_frame.groupby("topology"):
        parameter_summaries[topology] = {
            "rows": len(block),
            "boundary_rows": int(block.any_boundary.sum()),
            "parameters": {
                name: {
                    "minimum": float(block[name].min()),
                    "median": float(block[name].median()),
                    "maximum": float(block[name].max()),
                }
                for name in parameter_names
            },
        }
    indicators = pd.read_csv(SYSTEM_INDICATORS)
    coordinate = indicators.tidal_middle_to_max.to_numpy(float)
    gate_exposure = {}
    for name in scores.variant:
        weights = gate_weight(coordinate, settings[name])
        exposure = indicators[["domain"]].copy()
        exposure["weight"] = weights
        gate_exposure[name] = {
            domain: {
                "systems": len(block),
                "mean": float(block.weight.mean()),
                "median": float(block.weight.median()),
                "p10": float(block.weight.quantile(0.1)),
                "p90": float(block.weight.quantile(0.9)),
            }
            for domain, block in exposure.groupby("domain")
        }
    output = {
        "status": "completed nonmonotonic tidal-gate topology analysis",
        "formula": {
            "band": "w_band=sigmoid[k(x-lower)] sigmoid[k(upper-x)]",
            "tails": "w_tails=1-w_band",
            "placement": "w places the existing RG ceiling before channel combination and Solar screening",
        },
        "coverage": {
            "universal_refits": len(scores),
            "nonmonotonic_factorial_cells": len(nonmonotonic),
            "exact_complement_pairs": len(complement_pairs),
            "eight_start_raw_replays": len(robustness["comparisons"]),
            "stable_root_complete_replays": int(stable.shape[0]),
            "stable_nonmonotonic_replays": int(
                nonmonotonic.raw_eight_start_all_roots.sum()
            ),
        },
        "references": references,
        "best_stable_observed": best.to_dict(),
        "best_stable_raw_case": best_raw.to_dict(),
        "balanced_effects": effects,
        "locked_baseline_parameter_effects": locked_baseline_effects,
        "exact_complement_pairs": complement_pairs,
        "endpoint_reconciliation": reconciliation,
        "reconciliation_count": sum(
            row["reconciles_both_endpoints"] for row in reconciliation
        ),
        "topology_parameter_summaries": parameter_summaries,
        "gate_exposure_at_system_medians": gate_exposure,
        "root_reversals": reversals,
        "claim_boundary": protocol["claim_boundary"],
        "input_hashes": {
            "protocol": sha256(PROTOCOL),
            "report": sha256(REPORT),
            "scores": sha256(SCORES),
            "robust_protocol": sha256(ROBUST_PROTOCOL),
            "robust_report": sha256(ROBUST_REPORT),
            "system_tidal_indicators": sha256(SYSTEM_INDICATORS),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(output), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Nonmonotonic tidal-gate topology analysis",
        "",
        "Band and two-tail gates are exact complements on the same measured tidal coordinate.",
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
        f"- Nonmonotonic rows beating both monotonic endpoints: {output['reconciliation_count']}",
        "",
        "## Main-effect ranking: nonmonotonic factorial",
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
            "Variance fractions apply only to the frozen 2x2x2x2 grid. The cluster-side tidal coordinate remains a spherical density-ratio proxy.",
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
                    "reconciliation_count": output["reconciliation_count"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
