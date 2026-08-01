#!/usr/bin/env python3
"""Consolidate the bracketed exact-endpoint boundary refinement."""

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


PROTOCOL = ROOT / "configs/reopened_hybrid_endpoint_boundary_refinement_protocol.json"
SCORES = ROOT / "results/reopened_hybrid_endpoint_boundary_refinement/scores.csv"
REPORT = ROOT / "results/reopened_hybrid_endpoint_boundary_refinement/report.json"
ROBUST = ROOT / "results/reopened_hybrid_endpoint_boundary_refinement_raw_robustness/report.json"
AUDIT = ROOT / "results/reopened_hybrid_endpoint_boundary_refinement_audit/scores.csv"
PRIOR = ROOT / "results/reopened_hybrid_endpoint_power_memory_analysis/report.json"
OUTPUT = ROOT / "results/reopened_hybrid_endpoint_boundary_refinement_analysis"
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
        "SPARC_RMSE_km_s": float(row.SPARC_outer_RMSE_km_s),
        "bridge_RMSE_dex": float(row.bridge_RMSE_dex),
        "raw_eight_start_RMS_arcsec": float(row.raw_eight_start_RMS_arcsec),
        "raw_eight_start_all_roots": bool(row.raw_eight_start_all_roots),
        "robust_cross_domain_reference_ratio": float(
            row.robust_cross_domain_reference_ratio
        ),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    robust = json.loads(ROBUST.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    audit = pd.read_csv(AUDIT)
    variants = {row["name"]: row for row in expand_variants(protocol)}
    scores = pd.read_csv(SCORES)
    if set(scores.variant) != set(variants):
        raise RuntimeError("refinement scores do not match frozen protocol")
    if set(robust["selected_variants"]) != set(scores.variant):
        raise RuntimeError("refinement robustness does not cover every row")

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
        raise RuntimeError("no Solar-valid, root-complete refinement row")
    best = stable_rows.iloc[0]
    best_raw = stable_rows.sort_values("raw_eight_start_RMS_arcsec").iloc[0]

    surface = scores[scores.family.str.startswith("surface_q_")].copy()
    correlations = []
    for coordinate in ["p", "q", "SPARC_effective_power", "CLASH_effective_power"]:
        for metric in [
            "SPARC_outer_RMSE_km_s",
            "bridge_RMSE_dex",
            "raw_eight_start_RMS_arcsec",
        ]:
            coefficient, p_value = spearmanr(surface[coordinate], surface[metric])
            correlations.append(
                {
                    "coordinate": coordinate,
                    "metric": metric,
                    "spearman_r": float(coefficient),
                    "two_sided_p_value_descriptive": float(p_value),
                }
            )
    surface_minima_by_q = []
    for q, block in surface.groupby("q", sort=True):
        row = block.sort_values("SPARC_outer_RMSE_km_s").iloc[0]
        surface_minima_by_q.append({"q": float(q), **compact(row)})
    surface_minima_by_p = []
    for p, block in surface.groupby("p", sort=True):
        row = block.sort_values("SPARC_outer_RMSE_km_s").iloc[0]
        surface_minima_by_p.append({"p": float(p), **compact(row)})

    scale_rows = scores[scores.family.str.startswith("scale_")].copy()
    scale_slices = []
    for family, block in scale_rows.groupby("family", sort=False):
        ordered = block.sort_values("memory_log_scale")
        best_block = ordered.sort_values("SPARC_outer_RMSE_km_s").iloc[0]
        scale_slices.append(
            {
                "family": family,
                "p": float(ordered.p.iloc[0]),
                "q": float(ordered.q.iloc[0]),
                "SPARC_span_km_s": span(ordered.SPARC_outer_RMSE_km_s),
                "bridge_span_dex": span(ordered.bridge_RMSE_dex),
                "raw_eight_start_span_arcsec": span(
                    ordered.raw_eight_start_RMS_arcsec
                ),
                "best_galaxy_row": compact(best_block),
            }
        )

    repeated_settings = variants[protocol["baseline_variant_name"]]["settings"]
    repeated_json = json.dumps(repeated_settings, sort_keys=True)
    repeats = scores[scores.settings_json.eq(repeated_json)].copy()
    if len(repeats) != 5:
        raise RuntimeError(f"expected five exact refinement repeats, found {len(repeats)}")
    repeat_parameters = pd.DataFrame(
        [
            {
                "variant": name,
                **report["results"][name]["full_fit_parameters"],
            }
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

    summary = {
        "status": "completed bracketed endpoint-boundary refinement analysis",
        "coverage": {
            "fixed_parameter_refinement_rows": int(len(audit)),
            "universal_refits": int(len(scores)),
            "eight_start_raw_replays": int(len(robust["selected_variants"])),
            "stable_root_complete_rows": int(len(stable_rows)),
            "surface_rows": int(len(surface)),
            "exact_repeat_refits": int(len(repeats)),
        },
        "references": REFERENCES,
        "measured_profile_slopes": SLOPES,
        "formula": prior["formula"],
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
        "exact_formula_repeatability": repeatability,
        "surface_minima_by_q": surface_minima_by_q,
        "surface_minima_by_p": surface_minima_by_p,
        "surface_correlations": correlations,
        "memory_scale_slices": scale_slices,
        "root_reversals": root_reversals,
        "solar_failures": scores.loc[~scores.solar_all_pass, "variant"].tolist(),
        "empirical_interpretation": [
            "The memory-length optimum is bracketed rather than left at the tested boundary.",
            "The p,q response forms a shallow ridge; neighboring exponent pairs are more informative than a single best pair.",
            "Exact repeats determine whether the leading formula's typical prediction survives bridge-optimizer non-identification.",
            "Only the bracketed exact-endpoint formula is ranked; no parent nonlocal, history, void, or slope mechanism is rejected.",
        ],
        "claim_boundary": [
            "This is sequential development-data analysis, not a preregistered external holdout.",
            "The bridge parameters remain bounded and partly nonidentified.",
            "Raw lensing is a zero-slip pseudo-elliptical transfer and Solar checks are weak-field proxies.",
            "A bracketed empirical ridge is not a microscopic derivation or novelty claim.",
        ],
        "input_hashes": {
            "protocol": sha256(PROTOCOL),
            "scores": sha256(SCORES),
            "report": sha256(REPORT),
            "robustness": sha256(ROBUST),
            "fixed_audit": sha256(AUDIT),
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
        "# Bracketed endpoint-boundary refinement results",
        "",
        f"Coverage: **{len(audit)}** fixed rows, **{len(scores)}** universal refits, **{len(robust['selected_variants'])}** eight-start replays, and **{len(repeats)}** exact repeats.",
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
