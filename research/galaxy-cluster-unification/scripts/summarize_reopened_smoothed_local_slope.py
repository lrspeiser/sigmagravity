#!/usr/bin/env python3
"""Consolidate smoothed-local-slope audits, refits, and robust lensing."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_reopened_hybrid_sensitivity import expand_variants, json_safe  # noqa: E402


PROTOCOL = ROOT / "configs/reopened_hybrid_smoothed_local_slope_protocol.json"
SCORES = ROOT / "results/reopened_hybrid_smoothed_local_slope/scores.csv"
REPORT = ROOT / "results/reopened_hybrid_smoothed_local_slope/report.json"
ROBUST = ROOT / "results/reopened_hybrid_smoothed_local_slope_raw_robustness/report.json"
FIXED_AUDIT = ROOT / "results/reopened_hybrid_smoothed_local_slope_audit/report.json"
FIXED_SCORES = ROOT / "results/reopened_hybrid_smoothed_local_slope_audit/scores.csv"
GEOMETRY = ROOT / "results/reopened_smoothed_slope_geometry_audit/report.json"
PRIOR = ROOT / "results/reopened_hybrid_slope_response_fine_analysis/report.json"
OUTPUT = ROOT / "results/reopened_hybrid_smoothed_local_slope_analysis"
REFERENCES = {
    "SPARC_fixed_RAR_outer_RMSE_km_s": 10.681519066526649,
    "bridge_target_RMSE_dex": 0.139,
    "raw_baryons_RMS_arcsec": 27.43864684589079,
    "raw_simple_MOND_RMS_arcsec": 28.188493432162694,
    "raw_compact_halo_RMS_arcsec": 9.048410306058654,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_summary(row: pd.Series) -> dict[str, object]:
    return {
        "variant": row.variant,
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
    fixed_audit = json.loads(FIXED_AUDIT.read_text(encoding="utf-8"))
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    scores = pd.read_csv(SCORES)
    fixed_scores = pd.read_csv(FIXED_SCORES)
    variants = {row["name"]: row for row in expand_variants(protocol)}
    if set(scores.variant) != set(variants):
        raise RuntimeError("full scores do not match the frozen protocol")
    scores["settings_json"] = scores.variant.map(
        lambda name: json.dumps(variants[name]["settings"], sort_keys=True)
    )
    scores["raw_eight_start_RMS_arcsec"] = np.nan
    scores["raw_eight_start_all_roots"] = pd.Series(
        pd.NA, index=scores.index, dtype="boolean"
    )
    scores["raw_eight_start_reduced_chi2"] = np.nan
    root_reversals = []
    for name, comparison in robust["comparisons"].items():
        mask = scores.variant.eq(name)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"robustness row is missing: {name}")
        two = comparison["two_start"]
        eight = comparison["eight_start"]
        scores.loc[mask, "raw_eight_start_RMS_arcsec"] = eight[
            "equal_system_radial_RMS_arcsec"
        ]
        scores.loc[mask, "raw_eight_start_all_roots"] = eight[
            "all_roots_converged"
        ]
        scores.loc[mask, "raw_eight_start_reduced_chi2"] = eight[
            "pooled_reduced_chi2"
        ]
        if two["all_roots_converged"] != eight["all_roots_converged"]:
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
    best = stable_rows.iloc[0]
    best_raw = stable_rows.sort_values("raw_eight_start_RMS_arcsec").iloc[0]

    family_impacts = []
    for family, block in scores.groupby("family", sort=False):
        row = {
            "family": family,
            "rows": int(len(block)),
            "SPARC_span_km_s": float(block.SPARC_outer_RMSE_km_s.max() - block.SPARC_outer_RMSE_km_s.min()),
            "bridge_span_dex": float(block.bridge_RMSE_dex.max() - block.bridge_RMSE_dex.min()),
            "raw_two_start_span_arcsec": float(block.raw_lensing_RMS_arcsec.max() - block.raw_lensing_RMS_arcsec.min()),
            "two_start_root_states": int(block.raw_all_roots_converged.astype(bool).nunique()),
        }
        replayed = block[block.raw_eight_start_RMS_arcsec.notna()]
        row["raw_eight_start_span_arcsec"] = (
            float(replayed.raw_eight_start_RMS_arcsec.max() - replayed.raw_eight_start_RMS_arcsec.min())
            if len(replayed) > 1
            else np.nan
        )
        row["eight_start_root_states"] = (
            int(replayed.raw_eight_start_all_roots.astype(bool).nunique())
            if len(replayed)
            else 0
        )
        row["maximum_normalized_span"] = max(
            row["SPARC_span_km_s"] / REFERENCES["SPARC_fixed_RAR_outer_RMSE_km_s"],
            row["bridge_span_dex"] / REFERENCES["bridge_target_RMSE_dex"],
            row["raw_two_start_span_arcsec"] / REFERENCES["raw_compact_halo_RMS_arcsec"],
        )
        family_impacts.append(row)
    impacts = pd.DataFrame(family_impacts).sort_values(
        "maximum_normalized_span", ascending=False
    )

    duplicate_groups = []
    for settings, block in scores.groupby("settings_json", sort=False):
        if len(block) <= 1:
            continue
        replayed = block[block.raw_eight_start_RMS_arcsec.notna()]
        duplicate_groups.append(
            {
                "settings": json.loads(settings),
                "variants": block.variant.tolist(),
                "copies": int(len(block)),
                "SPARC_span_km_s": float(np.ptp(block.SPARC_outer_RMSE_km_s)),
                "bridge_span_dex": float(np.ptp(block.bridge_RMSE_dex)),
                "raw_two_start_span_arcsec": float(np.ptp(block.raw_lensing_RMS_arcsec)),
                "eight_start_replayed_copies": int(len(replayed)),
                "raw_eight_start_span_arcsec": (
                    float(np.ptp(replayed.raw_eight_start_RMS_arcsec))
                    if len(replayed) > 1
                    else None
                ),
                "eight_start_all_roots": (
                    bool(replayed.raw_eight_start_all_roots.astype(bool).all())
                    if len(replayed)
                    else None
                ),
            }
        )
    duplicate_groups.sort(key=lambda row: row["copies"], reverse=True)

    local_scales = geometry["by_smoothing_log_scale"]
    range_tradeoff = {
        scale: {
            "SPARC_vs_CLASH_slope_AUC": row["SPARC_vs_CLASH_slope_auc"],
            "raw_cutoff_median_slope_span": row[
                "raw_cutoff_median_fixed_radius_slope_span"
            ],
            "raw_cutoff_maximum_slope_span": row[
                "raw_cutoff_maximum_fixed_radius_slope_span"
            ],
        }
        for scale, row in local_scales.items()
    }
    prior_typical = prior["identical_formula_repeatability"][
        "median_stable_robust_reference_ratio"
    ]
    summary = {
        "status": "completed smoothed-local-slope sensitivity analysis",
        "coverage": {
            "fixed_parameter_variants": int(fixed_audit["rows"]),
            "full_universal_refits": int(len(scores)),
            "eight_start_raw_replays": int(len(robust["selected_variants"])),
            "stable_root_complete_replays": int(len(stable_rows)),
            **geometry["coverage"],
        },
        "references": REFERENCES,
        "formula": {
            "local_slope": "s_ell(r)=weighted local-linear slope of ln(g_N) versus ln(r)",
            "kernel": "K=exp[-0.5(Delta ln(r)/ell)^2]",
            "gate": "w=mu_s logistic[k(s_pivot-s_ell)]",
            "response": "F=(1-w)F_base_completed+w F_steep_completed",
        },
        "best_stable_compromise": {
            **row_summary(best),
            "improvement_vs_prior_repeatable_ratio_percent": float(
                100.0
                * (1.0 - best.robust_cross_domain_reference_ratio / prior_typical)
            ),
            "SPARC_ratio_to_RAR": float(
                best.SPARC_outer_RMSE_km_s
                / REFERENCES["SPARC_fixed_RAR_outer_RMSE_km_s"]
            ),
            "raw_ratio_to_baryons": float(
                best.raw_eight_start_RMS_arcsec
                / REFERENCES["raw_baryons_RMS_arcsec"]
            ),
            "raw_ratio_to_simple_MOND": float(
                best.raw_eight_start_RMS_arcsec
                / REFERENCES["raw_simple_MOND_RMS_arcsec"]
            ),
            "raw_ratio_to_compact_halo": float(
                best.raw_eight_start_RMS_arcsec
                / REFERENCES["raw_compact_halo_RMS_arcsec"]
            ),
        },
        "best_stable_raw_case": {
            **row_summary(best_raw),
            "raw_improvement_vs_baryons_percent": float(
                100.0
                * (1.0 - best_raw.raw_eight_start_RMS_arcsec / REFERENCES["raw_baryons_RMS_arcsec"])
            ),
            "raw_improvement_vs_simple_MOND_percent": float(
                100.0
                * (1.0 - best_raw.raw_eight_start_RMS_arcsec / REFERENCES["raw_simple_MOND_RMS_arcsec"])
            ),
        },
        "parameter_impact_ranking": impacts.to_dict("records"),
        "exact_formula_duplicate_groups": duplicate_groups,
        "smoothing_range_tradeoff": range_tradeoff,
        "root_reversals": root_reversals,
        "solar_failures": scores.loc[~scores.solar_all_pass, "variant"].tolist(),
        "full_fit_parameter_boundary_rows": scores.loc[
            scores.any_universal_parameter_at_boundary.astype(bool), "variant"
        ].tolist(),
        "empirical_findings": [
            "Gate strength is the largest joint amplitude lever in this grid, while memory length is the largest bridge-shape lever and also crosses raw-lens root branches.",
            "The smoothing bandwidth has very small fixed-parameter galaxy and bridge leverage. Narrow bandwidths preserve local cutoff invariance; broad bandwidths improve descriptive galaxy/cluster separation but inherit outer-cutoff dependence.",
            "Moving the pivot upward produces the best observed stable galaxy/lensing compromise, and the best tested point lies at the upper pivot boundary 1.25, so the transition scale is not yet bracketed.",
            "The eight-start replay recovers five formulas that lacked all roots at two starts and confirms that no two-start root loss is sufficient evidence against a formula.",
            "Exact duplicate settings still occupy multiple bridge-optimizer branches. A structural winner must be repeated before its best score is treated as representative.",
            "All tested settings pass the current Solar proxies; Solar suppression is controlled upstream by the common screen and is insensitive to these profile-only changes in the tested range.",
        ],
        "claim_boundary": [
            "These are development-data sensitivity results, not a frozen-formula external holdout.",
            "The best pivot is a boundary result and one optimizer branch, so it is a follow-up target rather than an identified universal constant.",
            "The raw lensing calculation remains a zero-slip pseudo-elliptical transfer, and the Solar checks remain weak-field proxies.",
            "Only the frozen formula and parameter ranges are ranked; no parent new-physics idea is rejected.",
        ],
        "input_hashes": {
            "protocol": sha256(PROTOCOL),
            "scores": sha256(SCORES),
            "report": sha256(REPORT),
            "robustness": sha256(ROBUST),
            "fixed_audit": sha256(FIXED_AUDIT),
            "fixed_scores": sha256(FIXED_SCORES),
            "geometry": sha256(GEOMETRY),
            "prior": sha256(PRIOR),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    impacts.to_csv(OUTPUT / "parameter_impacts.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Smoothed-local-slope sensitivity results",
        "",
        f"Coverage: **{fixed_audit['rows']}** fixed variants, **{len(scores)}** universal refits, and **{len(robust['selected_variants'])}** eight-start raw-lens replays.",
        "",
        f"Best stable observed branch: `{best.variant}` — {best.SPARC_outer_RMSE_km_s:.3f} km/s on SPARC, {best.raw_eight_start_RMS_arcsec:.3f} arcsec on raw lensing, ratio {best.robust_cross_domain_reference_ratio:.3f}.",
        "",
        f"Best stable raw case: `{best_raw.variant}` — {best_raw.raw_eight_start_RMS_arcsec:.3f} arcsec with all roots and {best_raw.SPARC_outer_RMSE_km_s:.3f} km/s on SPARC.",
        "",
        "The pivot is still improving at the tested boundary, while bandwidth itself is low leverage below the range-dependent global regime. Repeat and extend the pivot before claiming an optimum.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
