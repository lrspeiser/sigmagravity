#!/usr/bin/env python3
"""Analyze the balanced endpoint exponent-memory interaction factorial."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_reopened_hybrid_sensitivity import expand_variants, json_safe  # noqa: E402


PROTOCOL = ROOT / "configs/reopened_hybrid_endpoint_interaction_factorial_protocol.json"
SCORES = ROOT / "results/reopened_hybrid_endpoint_interaction_factorial/scores.csv"
REPORT = ROOT / "results/reopened_hybrid_endpoint_interaction_factorial/report.json"
ROBUST = ROOT / "results/reopened_hybrid_endpoint_interaction_factorial_raw_robustness/report.json"
PRIOR = ROOT / "results/reopened_hybrid_endpoint_high_q_ridge_analysis/report.json"
OUTPUT = ROOT / "results/reopened_hybrid_endpoint_interaction_factorial_analysis"
REFERENCES = {
    "SPARC_fixed_RAR_outer_RMSE_km_s": 10.681519066526649,
    "bridge_target_RMSE_dex": 0.139,
    "raw_baryons_RMS_arcsec": 27.43864684589079,
    "raw_simple_MOND_RMS_arcsec": 28.188493432162694,
    "raw_compact_halo_RMS_arcsec": 9.048410306058654,
}
SPARC_SLOPE = -1.556505303070826
FACTORS = ["q", "SPARC_effective_power", "memory_log_scale", "memory_strength"]
METRICS = [
    "SPARC_outer_RMSE_km_s",
    "bridge_RMSE_dex",
    "raw_eight_start_RMS_arcsec",
    "solar_maximum_fractional_change",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def span(values: pd.Series) -> float:
    clean = values.astype(float)
    return float(clean.max() - clean.min())


def compact(row: pd.Series) -> dict[str, object]:
    return {
        "variant": row.variant,
        "p": float(row.p),
        "q": float(row.q),
        "SPARC_effective_power": float(row.SPARC_effective_power),
        "memory_log_scale": float(row.memory_log_scale),
        "memory_strength": float(row.memory_strength),
        "SPARC_RMSE_km_s": float(row.SPARC_outer_RMSE_km_s),
        "bridge_RMSE_dex": float(row.bridge_RMSE_dex),
        "raw_eight_start_RMS_arcsec": float(row.raw_eight_start_RMS_arcsec),
        "raw_eight_start_all_roots": bool(row.raw_eight_start_all_roots),
        "robust_cross_domain_reference_ratio": float(
            row.robust_cross_domain_reference_ratio
        ),
    }


def balanced_effect_decomposition(
    frame: pd.DataFrame, metric: str
) -> dict[str, object]:
    """Classical orthogonal main/pair SS for a complete balanced factorial."""
    values = frame[metric].astype(float)
    grand = float(values.mean())
    total_ss = float(np.sum(np.square(values - grand)))
    main = []
    marginal_means: dict[str, pd.Series] = {}
    for factor in FACTORS:
        means = frame.groupby(factor, sort=True)[metric].mean()
        marginal_means[factor] = means
        observations_per_level = len(frame) / len(means)
        effect_ss = float(
            observations_per_level * np.sum(np.square(means - grand))
        )
        main.append(
            {
                "factor": factor,
                "sum_squares": effect_ss,
                "fraction_of_total_variation": (
                    effect_ss / total_ss if total_ss > 0.0 else 0.0
                ),
                "level_mean_span": span(means),
                "level_means": {
                    str(level): float(value) for level, value in means.items()
                },
            }
        )
    pairwise = []
    for left, right in itertools.combinations(FACTORS, 2):
        means = frame.groupby([left, right], sort=True)[metric].mean()
        residuals = []
        for (left_level, right_level), value in means.items():
            residuals.append(
                float(value)
                - float(marginal_means[left].loc[left_level])
                - float(marginal_means[right].loc[right_level])
                + grand
            )
        observations_per_cell = len(frame) / len(means)
        effect_ss = float(
            observations_per_cell * np.sum(np.square(residuals))
        )
        pairwise.append(
            {
                "factors": [left, right],
                "sum_squares": effect_ss,
                "fraction_of_total_variation": (
                    effect_ss / total_ss if total_ss > 0.0 else 0.0
                ),
                "maximum_absolute_interaction_residual": float(
                    np.max(np.abs(residuals))
                ),
            }
        )
    return {
        "metric": metric,
        "grand_mean": grand,
        "total_sum_squares": total_ss,
        "main_effects_ranked": sorted(
            main,
            key=lambda row: row["fraction_of_total_variation"],
            reverse=True,
        ),
        "pairwise_interactions_ranked": sorted(
            pairwise,
            key=lambda row: row["fraction_of_total_variation"],
            reverse=True,
        ),
        "unassigned_higher_order_fraction": float(
            max(
                0.0,
                1.0
                - sum(row["fraction_of_total_variation"] for row in main)
                - sum(
                    row["fraction_of_total_variation"] for row in pairwise
                ),
            )
        ),
        "raw_metric_note": (
            "Raw-lensing decomposition uses every finite diagnostic RMS to preserve the balanced design; root-incomplete cells are not valid ranked fits."
            if metric == "raw_eight_start_RMS_arcsec"
            else None
        ),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    robust = json.loads(ROBUST.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    variants = {row["name"]: row for row in expand_variants(protocol)}
    scores = pd.read_csv(SCORES)
    if set(scores.variant) != set(variants):
        raise RuntimeError("factorial scores do not match the frozen protocol")
    if set(robust["selected_variants"]) != set(scores.variant):
        raise RuntimeError("factorial robustness does not cover every row")

    settings = scores.variant.map(lambda name: variants[name]["settings"])
    scores["p"] = settings.map(
        lambda row: float(row["radial_memory_gbar_power"])
    )
    scores["q"] = settings.map(
        lambda row: float(row["radial_memory_radius_power"])
    )
    scores["memory_log_scale"] = settings.map(
        lambda row: float(row["radial_memory_log_scale"])
    )
    scores["memory_strength"] = settings.map(
        lambda row: float(row["radial_memory_strength"])
    )
    scores["SPARC_effective_power_raw"] = scores.q + SPARC_SLOPE * scores.p
    scores["SPARC_effective_power"] = scores.SPARC_effective_power_raw.map(
        lambda value: min([5.3, 5.65, 6.0], key=lambda level: abs(level - value))
    )
    scores["settings_json"] = settings.map(
        lambda row: json.dumps(row, sort_keys=True)
    )
    scores["raw_eight_start_RMS_arcsec"] = np.nan
    scores["raw_eight_start_all_roots"] = pd.Series(
        pd.NA, index=scores.index, dtype="boolean"
    )
    scores["raw_eight_start_pooled_reduced_chi2"] = np.nan
    root_reversals = []
    for name, comparison in robust["comparisons"].items():
        mask = scores.variant.eq(name)
        two = comparison["two_start"]
        eight = comparison["eight_start"]
        scores.loc[mask, "raw_eight_start_RMS_arcsec"] = eight[
            "equal_system_radial_RMS_arcsec"
        ]
        scores.loc[mask, "raw_eight_start_all_roots"] = eight[
            "all_roots_converged"
        ]
        scores.loc[mask, "raw_eight_start_pooled_reduced_chi2"] = eight[
            "pooled_reduced_chi2"
        ]
        if bool(two["all_roots_converged"]) != bool(
            eight["all_roots_converged"]
        ):
            root_reversals.append(
                {
                    "variant": name,
                    "two_start_all_roots": bool(two["all_roots_converged"]),
                    "eight_start_all_roots": bool(eight["all_roots_converged"]),
                }
            )
    scores["solar_all_pass"] = (
        scores.Cassini_proxy_pass.astype(bool)
        & scores.Earth_pass.astype(bool)
        & scores.Mercury_pass.astype(bool)
    )
    stable = (
        scores.solar_all_pass
        & scores.raw_eight_start_all_roots.fillna(False).astype(bool)
    )
    scores["robust_cross_domain_reference_ratio"] = np.nan
    scores.loc[stable, "robust_cross_domain_reference_ratio"] = np.maximum(
        scores.loc[stable, "SPARC_outer_RMSE_km_s"]
        / REFERENCES["SPARC_fixed_RAR_outer_RMSE_km_s"],
        scores.loc[stable, "raw_eight_start_RMS_arcsec"]
        / REFERENCES["raw_compact_halo_RMS_arcsec"],
    )
    stable_rows = scores[stable].sort_values(
        ["robust_cross_domain_reference_ratio", "bridge_RMSE_dex"]
    )
    if stable_rows.empty:
        raise RuntimeError("no Solar-valid, root-complete factorial row")
    best = stable_rows.iloc[0]
    best_raw = stable_rows.sort_values("raw_eight_start_RMS_arcsec").iloc[0]

    factorial = scores[scores.variant.str.startswith("factorial_")].copy()
    if len(factorial) != 81:
        raise RuntimeError(f"expected 81 factorial rows, found {len(factorial)}")
    if factorial.groupby(FACTORS).size().ne(1).any():
        raise RuntimeError("factorial cells are not unique and balanced")
    decompositions = {
        metric: balanced_effect_decomposition(factorial, metric)
        for metric in METRICS
    }
    root_completion = {
        "overall_fraction": float(
            factorial.raw_eight_start_all_roots.astype(bool).mean()
        ),
        "by_factor": {
            factor: {
                str(level): float(value)
                for level, value in factorial.groupby(factor)[
                    "raw_eight_start_all_roots"
                ].mean().items()
            }
            for factor in FACTORS
        },
        "incomplete_cells": factorial.loc[
            ~factorial.raw_eight_start_all_roots.astype(bool),
            [
                "variant",
                "q",
                "SPARC_effective_power",
                "memory_log_scale",
                "memory_strength",
                "SPARC_outer_RMSE_km_s",
                "raw_eight_start_RMS_arcsec",
            ],
        ].to_dict(orient="records"),
    }

    baseline_json = json.dumps(
        variants[protocol["baseline_variant_name"]]["settings"], sort_keys=True
    )
    repeats = scores[scores.settings_json.eq(baseline_json)].copy()
    if len(repeats) != 5:
        raise RuntimeError(f"expected five exact factorial copies, found {len(repeats)}")
    repeat_parameters = pd.DataFrame(
        [
            {"variant": name, **report["results"][name]["full_fit_parameters"]}
            for name in repeats.variant
        ]
    )
    repeatability = {
        "copies": int(len(repeats)),
        "stable_root_complete_copies": int(
            (
                repeats.solar_all_pass
                & repeats.raw_eight_start_all_roots.astype(bool)
            ).sum()
        ),
        "SPARC_span_km_s": span(repeats.SPARC_outer_RMSE_km_s),
        "SPARC_median_km_s": float(repeats.SPARC_outer_RMSE_km_s.median()),
        "bridge_span_dex": span(repeats.bridge_RMSE_dex),
        "raw_eight_start_span_arcsec": span(
            repeats.raw_eight_start_RMS_arcsec
        ),
        "raw_eight_start_median_arcsec": float(
            repeats.raw_eight_start_RMS_arcsec.median()
        ),
        "median_robust_reference_ratio": float(
            repeats.robust_cross_domain_reference_ratio.median()
        ),
        "universal_parameter_ranges": {
            parameter: {
                "minimum": float(repeat_parameters[parameter].min()),
                "median": float(repeat_parameters[parameter].median()),
                "maximum": float(repeat_parameters[parameter].max()),
                "span": span(repeat_parameters[parameter]),
            }
            for parameter in protocol["universal_parameters"]["names"]
        },
    }

    best_factorial = factorial[
        factorial.solar_all_pass
        & factorial.raw_eight_start_all_roots.astype(bool)
    ].sort_values("robust_cross_domain_reference_ratio").iloc[0]
    threshold = float(best_factorial.robust_cross_domain_reference_ratio * 1.01)
    near_optimal = factorial[
        factorial.robust_cross_domain_reference_ratio.le(threshold)
    ]
    plateau = {
        "within_one_percent_of_best_factorial_ratio": int(len(near_optimal)),
        "ratio_threshold": threshold,
        "factor_ranges": {
            factor: {
                "minimum": float(near_optimal[factor].min()),
                "maximum": float(near_optimal[factor].max()),
            }
            for factor in FACTORS
        },
    }

    summary = {
        "status": "completed endpoint interaction-factorial analysis",
        "coverage": {
            "factorial_cells": int(len(factorial)),
            "universal_refits": int(len(scores)),
            "eight_start_raw_replays": int(len(robust["selected_variants"])),
            "stable_root_complete_rows": int(len(stable_rows)),
            "exact_repeat_refits": int(len(repeats)),
        },
        "references": REFERENCES,
        "formula": prior["formula"],
        "factor_levels": {
            factor: sorted(float(value) for value in factorial[factor].unique())
            for factor in FACTORS
        },
        "best_stable_observed": {
            **compact(best),
            "improvement_vs_prior_stage_best_percent": float(
                100.0
                * (
                    1.0
                    - best.robust_cross_domain_reference_ratio
                    / prior["best_stable_observed"][
                        "robust_cross_domain_reference_ratio"
                    ]
                )
            ),
        },
        "best_stable_factorial_cell": compact(best_factorial),
        "best_stable_raw_case": {
            **compact(best_raw),
            "raw_ratio_to_baryons": float(
                best_raw.raw_eight_start_RMS_arcsec
                / REFERENCES["raw_baryons_RMS_arcsec"]
            ),
            "raw_ratio_to_simple_MOND": float(
                best_raw.raw_eight_start_RMS_arcsec
                / REFERENCES["raw_simple_MOND_RMS_arcsec"]
            ),
            "raw_ratio_to_compact_halo": float(
                best_raw.raw_eight_start_RMS_arcsec
                / REFERENCES["raw_compact_halo_RMS_arcsec"]
            ),
        },
        "balanced_effect_decompositions": decompositions,
        "root_completion": root_completion,
        "near_optimal_plateau": plateau,
        "exact_formula_repeatability": repeatability,
        "root_reversals": root_reversals,
        "solar_failures": scores.loc[~scores.solar_all_pass, "variant"].tolist(),
        "empirical_interpretation": [
            "A balanced factorial distinguishes local main effects from exponent-memory interactions rather than attributing a coupled response to one coordinate.",
            "Effective power is used as an empirical coordinate derived from the measured median SPARC slope; it is not assumed to be a fundamental constant.",
            "Raw-lensing RMS effects over incomplete-root cells are diagnostic only; stable ranking requires complete eight-start roots.",
            "Only this local factorial region is ranked; no parent nonlocal, history, void, or slope mechanism is rejected.",
        ],
        "claim_boundary": [
            "This is sequential development-data analysis, not a preregistered external holdout.",
            "The bridge parameters remain bounded and partly nonidentified.",
            "Raw lensing is a zero-slip pseudo-elliptical transfer and Solar checks are weak-field proxies.",
            "Factorial variance fractions measure this sampled grid and are not population-wide physical variance components.",
        ],
        "input_hashes": {
            "protocol": sha256(PROTOCOL),
            "scores": sha256(SCORES),
            "report": sha256(REPORT),
            "robustness": sha256(ROBUST),
            "prior": sha256(PRIOR),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    repeat_parameters.to_csv(OUTPUT / "repeat_parameters.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Endpoint interaction-factorial results",
        "",
        f"Coverage: **{len(factorial)}** balanced cells, **{len(scores)}** universal refits, **{len(robust['selected_variants'])}** eight-start replays, and **{len(repeats)}** exact copies.",
        "",
        f"Best stable branch: `{best.variant}` - {best.SPARC_outer_RMSE_km_s:.3f} km/s on SPARC, {best.raw_eight_start_RMS_arcsec:.3f} arcsec on raw lensing, ratio {best.robust_cross_domain_reference_ratio:.3f}.",
        "",
        f"Exact-repeat median ratio: **{repeatability['median_robust_reference_ratio']:.3f}**.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
