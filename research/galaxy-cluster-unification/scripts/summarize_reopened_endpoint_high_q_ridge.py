#!/usr/bin/env python3
"""Consolidate the bracketed high-q endpoint ridge and matched-path controls."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_reopened_hybrid_sensitivity import expand_variants, json_safe  # noqa: E402


PROTOCOL = ROOT / "configs/reopened_hybrid_endpoint_high_q_ridge_protocol.json"
SCORES = ROOT / "results/reopened_hybrid_endpoint_high_q_ridge/scores.csv"
REPORT = ROOT / "results/reopened_hybrid_endpoint_high_q_ridge/report.json"
ROBUST = ROOT / "results/reopened_hybrid_endpoint_high_q_ridge_raw_robustness/report.json"
PRIOR = ROOT / "results/reopened_hybrid_endpoint_boundary_refinement_analysis/report.json"
AUDITS = {
    "fixed_q_by_p": ROOT / "results/reopened_hybrid_endpoint_high_q_audit/scores.csv",
    "p_bracket_q4p5_to_q5p5": ROOT / "results/reopened_hybrid_endpoint_high_q_p_bracket_audit/scores.csv",
    "moving_ridge_q3p5_to_q8": ROOT / "results/reopened_hybrid_endpoint_high_q_moving_ridge_audit/scores.csv",
    "moving_ridge_q8_to_q16": ROOT / "results/reopened_hybrid_endpoint_high_q_moving_ridge_extension_audit/scores.csv",
    "constant_sparc_power_q8_to_q20": ROOT / "results/reopened_hybrid_endpoint_constant_sparc_power_stress_audit/scores.csv",
}
OUTPUT = ROOT / "results/reopened_hybrid_endpoint_high_q_ridge_analysis"
REFERENCES = {
    "SPARC_fixed_RAR_outer_RMSE_km_s": 10.681519066526649,
    "bridge_target_RMSE_dex": 0.139,
    "raw_baryons_RMS_arcsec": 27.43864684589079,
    "raw_simple_MOND_RMS_arcsec": 28.188493432162694,
    "raw_compact_halo_RMS_arcsec": 9.048410306058654,
}
SLOPES = {
    "SPARC_median_dln_gbar_dln_r": -1.556505303070826,
    "CLASH_median_dln_gbar_dln_r": -0.4478066976644597,
}


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


def path_summary(name: str, block: pd.DataFrame) -> dict[str, object]:
    ordered = block.sort_values(["q", "p"])
    return {
        "path": name,
        "rows": int(len(ordered)),
        "SPARC_span_km_s": span(ordered.SPARC_outer_RMSE_km_s),
        "bridge_span_dex": span(ordered.bridge_RMSE_dex),
        "raw_eight_start_span_arcsec": span(ordered.raw_eight_start_RMS_arcsec),
        "rows_detail": [compact(row) for _, row in ordered.iterrows()],
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    robust = json.loads(ROBUST.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    audit_tables = {name: pd.read_csv(path) for name, path in AUDITS.items()}
    variants = {row["name"]: row for row in expand_variants(protocol)}
    scores = pd.read_csv(SCORES)
    if set(scores.variant) != set(variants):
        raise RuntimeError("high-q scores do not match the frozen protocol")
    if set(robust["selected_variants"]) != set(scores.variant):
        raise RuntimeError("high-q robustness does not cover every frozen row")

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
    scores["settings_json"] = settings.map(
        lambda row: json.dumps(row, sort_keys=True)
    )
    scores["SPARC_effective_power"] = (
        scores.q + scores.p * SLOPES["SPARC_median_dln_gbar_dln_r"]
    )
    scores["CLASH_effective_power"] = (
        scores.q + scores.p * SLOPES["CLASH_median_dln_gbar_dln_r"]
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
        raise RuntimeError("no Solar-valid, root-complete high-q row")
    best = stable_rows.iloc[0]
    best_raw = stable_rows.sort_values("raw_eight_start_RMS_arcsec").iloc[0]

    ridge = scores[scores.family.str.startswith("ridge_q_")].copy()
    ridge_minima = []
    for q, block in ridge.groupby("q", sort=True):
        row = block.sort_values("SPARC_outer_RMSE_km_s").iloc[0]
        ridge_minima.append({"q": float(q), **compact(row)})
    correlations = []
    for coordinate in [
        "p",
        "q",
        "SPARC_effective_power",
        "CLASH_effective_power",
    ]:
        for metric in [
            "SPARC_outer_RMSE_km_s",
            "bridge_RMSE_dex",
            "raw_eight_start_RMS_arcsec",
        ]:
            coefficient, p_value = spearmanr(ridge[coordinate], ridge[metric])
            correlations.append(
                {
                    "coordinate": coordinate,
                    "metric": metric,
                    "spearman_r": float(coefficient),
                    "two_sided_p_value_descriptive": float(p_value),
                }
            )

    paths = {
        "constant_SPARC_e6": scores.family.str.startswith("constant_sparc_e6_"),
        "constant_SPARC_e5p8": scores.family.str.startswith(
            "constant_sparc_e5p8_"
        ),
        "constant_CLASH_e8p849": scores.family.str.startswith(
            "constant_clash_e8p849_"
        ),
    }
    path_summaries = [
        path_summary(name, scores[mask]) for name, mask in paths.items()
    ]

    scale = scores[scores.family.eq("high_q_memory_scale")].sort_values(
        "memory_log_scale"
    )
    strength = scores[scores.family.eq("high_q_memory_strength")].sort_values(
        "memory_strength"
    )
    memory_response = {
        "scale": {
            "SPARC_span_km_s": span(scale.SPARC_outer_RMSE_km_s),
            "bridge_span_dex": span(scale.bridge_RMSE_dex),
            "raw_eight_start_span_arcsec": span(
                scale.raw_eight_start_RMS_arcsec
            ),
            "best_galaxy": compact(
                scale.sort_values("SPARC_outer_RMSE_km_s").iloc[0]
            ),
        },
        "strength": {
            "SPARC_span_km_s": span(strength.SPARC_outer_RMSE_km_s),
            "bridge_span_dex": span(strength.bridge_RMSE_dex),
            "raw_eight_start_span_arcsec": span(
                strength.raw_eight_start_RMS_arcsec
            ),
            "best_galaxy": compact(
                strength.sort_values("SPARC_outer_RMSE_km_s").iloc[0]
            ),
        },
    }

    repeated_json = json.dumps(protocol["common_variant"], sort_keys=True)
    repeats = scores[scores.settings_json.eq(repeated_json)].copy()
    if len(repeats) != 8:
        raise RuntimeError(f"expected eight exact high-q copies, found {len(repeats)}")
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
        "SPARC_minimum_km_s": float(repeats.SPARC_outer_RMSE_km_s.min()),
        "SPARC_median_km_s": float(repeats.SPARC_outer_RMSE_km_s.median()),
        "SPARC_maximum_km_s": float(repeats.SPARC_outer_RMSE_km_s.max()),
        "SPARC_span_km_s": span(repeats.SPARC_outer_RMSE_km_s),
        "bridge_span_dex": span(repeats.bridge_RMSE_dex),
        "raw_eight_start_minimum_arcsec": float(
            repeats.raw_eight_start_RMS_arcsec.min()
        ),
        "raw_eight_start_median_arcsec": float(
            repeats.raw_eight_start_RMS_arcsec.median()
        ),
        "raw_eight_start_maximum_arcsec": float(
            repeats.raw_eight_start_RMS_arcsec.max()
        ),
        "raw_eight_start_span_arcsec": span(
            repeats.raw_eight_start_RMS_arcsec
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

    fixed_audit_summary = {
        name: {
            "rows": int(len(table)),
            "valid_rows": int(table.valid.astype(bool).sum()),
            "solar_valid_rows": int(table.solar_all_pass.astype(bool).sum()),
            "best_galaxy_RMSE_km_s": float(
                table.SPARC_outer_RMSE_km_s.min()
            ),
        }
        for name, table in audit_tables.items()
    }
    summary = {
        "status": "completed high-q endpoint ridge analysis",
        "coverage": {
            "fixed_parameter_rows": int(sum(len(x) for x in audit_tables.values())),
            "universal_refits": int(len(scores)),
            "eight_start_raw_replays": int(len(robust["selected_variants"])),
            "stable_root_complete_rows": int(len(stable_rows)),
            "local_ridge_rows": int(len(ridge)),
            "matched_effective_power_rows": int(sum(mask.sum() for mask in paths.values())),
            "exact_repeat_refits": int(len(repeats)),
        },
        "references": REFERENCES,
        "measured_profile_slopes": SLOPES,
        "formula": prior["formula"],
        "fixed_parameter_audits": fixed_audit_summary,
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
        "ridge_minima_by_q": ridge_minima,
        "ridge_correlations": correlations,
        "matched_effective_power_paths": path_summaries,
        "memory_response": memory_response,
        "exact_formula_repeatability": repeatability,
        "root_reversals": root_reversals,
        "solar_failures": scores.loc[~scores.solar_all_pass, "variant"].tolist(),
        "empirical_interpretation": [
            "The first fixed-p q turnover was a coordinate artifact; moving p exposed a continued ridge that is bracketed only near q=9 to 10.",
            "Constant median-profile effective power is locally useful but fails as a global invariance at extreme exponents because real profiles are not single power laws.",
            "Matched SPARC and CLASH paths isolate whether exponent changes act differentially across domains after universal refitting.",
            "Only the tested high-q endpoint equations are ranked; no parent nonlocal, history, void, or slope mechanism is rejected.",
        ],
        "claim_boundary": [
            "This is sequential development-data analysis, not a preregistered external holdout.",
            "The bridge parameters remain bounded and partly nonidentified.",
            "Raw lensing is a zero-slip pseudo-elliptical transfer and Solar checks are weak-field proxies.",
            "A bracketed empirical exponent ridge is not a microscopic derivation or novelty claim.",
        ],
        "input_hashes": {
            "protocol": sha256(PROTOCOL),
            "scores": sha256(SCORES),
            "report": sha256(REPORT),
            "robustness": sha256(ROBUST),
            "prior": sha256(PRIOR),
            **{f"audit_{name}": sha256(path) for name, path in AUDITS.items()},
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    repeat_parameters.to_csv(OUTPUT / "repeat_parameters.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# High-q endpoint ridge results",
        "",
        f"Coverage: **{summary['coverage']['fixed_parameter_rows']}** fixed rows, **{len(scores)}** universal refits, **{len(robust['selected_variants'])}** eight-start replays, and **{len(repeats)}** exact copies.",
        "",
        f"Best stable observed branch: `{best.variant}` - {best.SPARC_outer_RMSE_km_s:.3f} km/s on SPARC, {best.raw_eight_start_RMS_arcsec:.3f} arcsec on raw lensing, ratio {best.robust_cross_domain_reference_ratio:.3f}.",
        "",
        f"Exact-repeat median ratio: **{repeatability['median_robust_reference_ratio']:.3f}**.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
