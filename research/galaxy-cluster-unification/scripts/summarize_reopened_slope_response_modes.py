#!/usr/bin/env python3
"""Consolidate the derivative-safe slope-response investigation."""

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


PROTOCOL = ROOT / "configs/reopened_hybrid_slope_response_modes_protocol.json"
SCORES = ROOT / "results/reopened_hybrid_slope_response_modes/scores.csv"
REPORT = ROOT / "results/reopened_hybrid_slope_response_modes/report.json"
ROBUST = ROOT / "results/reopened_hybrid_slope_response_modes_raw_robustness/report.json"
AUDIT = ROOT / "results/reopened_hybrid_slope_response_modes_audit/report.json"
AUDIT_SCORES = ROOT / "results/reopened_hybrid_slope_response_modes_audit/scores.csv"
RANGE = ROOT / "results/reopened_slope_response_range_audit/report.json"
PREVIOUS = ROOT / "results/reopened_hybrid_memory_carrier_analysis/report.json"
OUTPUT = ROOT / "results/reopened_hybrid_slope_response_modes_analysis"

REFERENCES = {
    "SPARC_fixed_RAR_outer_RMSE_km_s": 10.681519066526649,
    "raw_baryons_RMS_arcsec": 27.43864684589079,
    "raw_simple_MOND_RMS_arcsec": 28.188493432162694,
    "raw_compact_halo_RMS_arcsec": 9.048410306058654,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    robust = json.loads(ROBUST.read_text(encoding="utf-8"))
    audit_report = json.loads(AUDIT.read_text(encoding="utf-8"))
    range_report = json.loads(RANGE.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    scores = pd.read_csv(SCORES)
    audit_scores = pd.read_csv(AUDIT_SCORES)
    variants = {row["name"]: row for row in expand_variants(protocol)}
    if set(scores.variant) != set(variants):
        raise RuntimeError("full scores do not match the frozen protocol")

    scores["raw_eight_start_RMS_arcsec"] = np.nan
    scores["raw_eight_start_all_roots"] = pd.Series(
        pd.NA, index=scores.index, dtype="boolean"
    )
    scores["raw_eight_start_reduced_chi2"] = np.nan
    root_reversals = []
    for name, comparison in robust["comparisons"].items():
        mask = scores.variant.eq(name)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"missing robustness score for {name}")
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
                    "two_start_all_roots": two["all_roots_converged"],
                    "eight_start_all_roots": eight["all_roots_converged"],
                    "two_start_RMS_arcsec": two[
                        "equal_system_radial_RMS_arcsec"
                    ],
                    "eight_start_RMS_arcsec": eight[
                        "equal_system_radial_RMS_arcsec"
                    ],
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
    best_overall = stable_rows.iloc[0]
    scores["slope_gate_strength"] = scores.variant.map(
        lambda name: float(
            variants[name]["settings"].get(
                "radial_memory_slope_gate_strength", 0.0
            )
        )
    )
    scores["slope_gate_mode"] = scores.variant.map(
        lambda name: int(
            variants[name]["settings"].get("radial_memory_slope_gate_mode", 0)
        )
    )
    derivative_safe = (
        stable
        & scores.slope_gate_strength.gt(0.0)
        & scores.slope_gate_mode.isin([1, 2, 3])
    )
    best = scores[derivative_safe].sort_values(
        ["robust_cross_domain_reference_ratio", "bridge_RMSE_dex"]
    ).iloc[0]
    best_raw = stable_rows.sort_values("raw_eight_start_RMS_arcsec").iloc[0]

    mode_rows = audit_scores[
        audit_scores.family.str.endswith("mode_comparison")
    ].sort_values("fixed_value")
    modes = []
    mode_labels = {
        0: "pointwise exponent interpolation",
        1: "profile-constant exponent interpolation",
        2: "profile-constant completed-response blend",
        3: "pointwise completed-response blend",
    }
    for _, row in mode_rows.iterrows():
        mode = int(row.fixed_value)
        modes.append(
            {
                "mode": mode,
                "description": mode_labels[mode],
                "SPARC_fixed_parameter_RMSE_km_s": float(
                    row.SPARC_outer_RMSE_km_s
                ),
                "bridge_fixed_parameter_RMSE_dex": float(row.bridge_RMSE_dex),
            }
        )
    pointwise_error = modes[0]["SPARC_fixed_parameter_RMSE_km_s"]
    derivative_safe_error = min(
        row["SPARC_fixed_parameter_RMSE_km_s"] for row in modes[1:]
    )
    impact_frame = pd.DataFrame(report["family_impacts"])
    largest_impacts = {}
    for metric, block in impact_frame.groupby("metric"):
        valid = block[np.isfinite(block.absolute_span)]
        largest_impacts[metric] = (
            valid.sort_values("absolute_span", ascending=False).iloc[0].family
            if len(valid)
            else None
        )
    prior = previous["best_eight_start_stable_compromise"]
    raw = range_report["raw_lensing_range_dependence"]
    summary = {
        "status": "completed derivative-safe slope-response analysis",
        "coverage": {
            "fixed_parameter_variants": audit_report["rows"],
            "full_universal_refits": len(scores),
            "eight_start_raw_replays": len(robust["selected_variants"]),
            "SPARC_profile_slopes": range_report["distributions"][
                "SPARC_measured"
            ]["slope"]["systems"],
            "CLASH_profile_slopes": range_report["distributions"][
                "CLASH_measured"
            ]["slope"]["systems"],
            "raw_clusters_in_range_audit": raw["systems"],
        },
        "references": REFERENCES,
        "formula": {
            "profile_slope": "s=least-squares slope of ln(g_N) versus ln(r)",
            "local_slope": "s(r)=d ln(g_N)/d ln(r)",
            "gate": "w=mu_s logistic[k(s_pivot-s)]",
            "base_carrier": "X_0=F(g_N/g_ref)^p0(r/kpc)^q0",
            "steep_carrier": "X_1=F(g_N/g_ref)^p1(r/kpc)^q1",
            "completed_response_blend": "F_memory=(1-w)T[X_0]/C_0+wT[X_1]/C_1",
            "interpretation": "T is the radial-memory operation and C converts the transported quantity back to fractional force; blending after T keeps the answer between two completed responses.",
        },
        "mode_comparison_fixed_parameters": modes,
        "pointwise_exponent_failure_removed": {
            "pointwise_exponent_SPARC_RMSE_km_s": pointwise_error,
            "best_derivative_safe_SPARC_RMSE_km_s": derivative_safe_error,
            "error_reduction_percent": float(
                100.0 * (1.0 - derivative_safe_error / pointwise_error)
            ),
        },
        "best_stable_overall_including_fixed_controls": {
            "variant": best_overall.variant,
            "SPARC_RMSE_km_s": float(best_overall.SPARC_outer_RMSE_km_s),
            "bridge_RMSE_dex": float(best_overall.bridge_RMSE_dex),
            "raw_eight_start_RMS_arcsec": float(
                best_overall.raw_eight_start_RMS_arcsec
            ),
            "robust_cross_domain_reference_ratio": float(
                best_overall.robust_cross_domain_reference_ratio
            ),
        },
        "best_stable_derivative_safe_compromise": {
            "variant": best.variant,
            "SPARC_RMSE_km_s": float(best.SPARC_outer_RMSE_km_s),
            "bridge_RMSE_dex": float(best.bridge_RMSE_dex),
            "raw_eight_start_RMS_arcsec": float(
                best.raw_eight_start_RMS_arcsec
            ),
            "robust_cross_domain_reference_ratio": float(
                best.robust_cross_domain_reference_ratio
            ),
            "relative_change_vs_prior_best_percent": float(
                100.0
                * (
                    best.robust_cross_domain_reference_ratio
                    / prior["robust_cross_domain_reference_ratio"]
                    - 1.0
                )
            ),
            "raw_change_vs_prior_best_percent": float(
                100.0
                * (
                    best.raw_eight_start_RMS_arcsec
                    / prior["raw_eight_start_RMS_arcsec"]
                    - 1.0
                )
            ),
        },
        "best_stable_raw_stage_case": {
            "variant": best_raw.variant,
            "SPARC_RMSE_km_s": float(best_raw.SPARC_outer_RMSE_km_s),
            "bridge_RMSE_dex": float(best_raw.bridge_RMSE_dex),
            "raw_eight_start_RMS_arcsec": float(
                best_raw.raw_eight_start_RMS_arcsec
            ),
            "raw_improvement_vs_baryons_percent": float(
                100.0
                * (
                    1.0
                    - best_raw.raw_eight_start_RMS_arcsec
                    / REFERENCES["raw_baryons_RMS_arcsec"]
                )
            ),
            "raw_improvement_vs_MOND_percent": float(
                100.0
                * (
                    1.0
                    - best_raw.raw_eight_start_RMS_arcsec
                    / REFERENCES["raw_simple_MOND_RMS_arcsec"]
                )
            ),
            "raw_ratio_to_compact_halo": float(
                best_raw.raw_eight_start_RMS_arcsec
                / REFERENCES["raw_compact_halo_RMS_arcsec"]
            ),
            "SPARC_ratio_to_RAR": float(
                best_raw.SPARC_outer_RMSE_km_s
                / REFERENCES["SPARC_fixed_RAR_outer_RMSE_km_s"]
            ),
        },
        "radial_range_sensitivity": {
            "SPARC_median_measured_slope": range_report["distributions"][
                "SPARC_measured"
            ]["slope"]["median"],
            "CLASH_median_measured_slope": range_report["distributions"][
                "CLASH_measured"
            ]["slope"]["median"],
            **raw,
        },
        "largest_full_stage_formula_impacts": largest_impacts,
        "full_fit_parameter_boundary_rows": scores.loc[
            scores.any_universal_parameter_at_boundary.astype(bool), "variant"
        ].tolist(),
        "lens_root_reversals": root_reversals,
        "solar_failures": scores.loc[~scores.solar_all_pass, "variant"].tolist(),
        "empirical_findings": [
            "The slope idea survives when pointwise exponent derivatives are removed: three derivative-safe modes reduce the fixed-parameter galaxy error by about three quarters relative to the sharp pointwise-exponent control.",
            "Blending completed responses is bounded and moves raw lensing without recreating the extreme galaxy penalty.",
            "The best balanced derivative-safe case improves robust raw lensing substantially over the prior fixed carrier while changing its galaxy error by less than one percent, leaving the max cross-domain ratio nearly tied but slightly worse.",
            "A half-strength profile response gives the best robust raw score in this stage, beating baryons and fixed MOND on this four-cluster test but remaining far behind the object-specific compact halo and RAR on galaxies.",
            "The whole-profile slope is sensitive to the raw lensing extrapolation range, while the pointwise completed-response mode avoids that particular global-range dependence.",
            "All tested settings pass the present Solar proxies; cross-domain profile shape and raw-lens numerical branches remain the limiting tests.",
        ],
        "claim_boundary": [
            "These results disfavor only pointwise exponent interpolation in the tested range; they do not reject slope-dependent new physics.",
            "The raw coordinate data are a same-system observable transfer from clusters that contributed derived bridge profiles, not an external cluster holdout.",
            "The Solar test remains a weak-field phenomenological proxy rather than a derived PPN calculation.",
            "Every full-fit row reaches at least one gravity-parameter boundary, so numerical improvements remain phenomenological and not parameter-identified.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    pd.DataFrame(modes).to_csv(OUTPUT / "mode_comparison.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Derivative-safe slope-response results",
        "",
        f"Coverage: **{audit_report['rows']}** fixed variants, **{len(scores)}** universal refits, and **{len(robust['selected_variants'])}** eight-start raw-lens replays.",
        "",
        "| mode | fixed SPARC RMSE | fixed bridge RMSE |",
        "|---|---:|---:|",
    ]
    for row in modes:
        lines.append(
            f"| {row['description']} | {row['SPARC_fixed_parameter_RMSE_km_s']:.3f} km/s | {row['bridge_fixed_parameter_RMSE_dex']:.3f} dex |"
        )
    lines.extend(
        [
            "",
            f"Best derivative-safe compromise: `{best.variant}` — {best.SPARC_outer_RMSE_km_s:.3f} km/s on SPARC and {best.raw_eight_start_RMS_arcsec:.3f} arcsec on raw lensing.",
            "",
            f"Best stable raw case: `{best_raw.variant}` — {best_raw.raw_eight_start_RMS_arcsec:.3f} arcsec with all roots, versus {REFERENCES['raw_baryons_RMS_arcsec']:.3f} for baryons and {REFERENCES['raw_compact_halo_RMS_arcsec']:.3f} for the compact halo.",
            "",
            "The data support continuing bounded response blends and reject only the tested pointwise exponent implementation.",
        ]
    )
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
