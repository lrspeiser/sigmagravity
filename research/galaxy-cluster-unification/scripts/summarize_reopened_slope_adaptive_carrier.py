#!/usr/bin/env python3
"""Consolidate the local-slope adaptive carrier investigation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_reopened_hybrid_sensitivity import expand_variants, json_safe  # noqa: E402


PROTOCOL_PATH = ROOT / "configs/reopened_hybrid_slope_adaptive_carrier_protocol.json"
SCORES_PATH = ROOT / "results/reopened_hybrid_slope_adaptive_carrier/scores.csv"
REPORT_PATH = ROOT / "results/reopened_hybrid_slope_adaptive_carrier/report.json"
ROBUST_PATH = ROOT / "results/reopened_hybrid_slope_adaptive_carrier_raw_robustness/report.json"
AUDIT_PATH = ROOT / "results/reopened_hybrid_slope_adaptive_carrier_audit/report.json"
GEOMETRY_PATH = ROOT / "results/reopened_slope_gate_geometry_audit/report.json"
FAILURE_PATH = ROOT / "results/reopened_slope_gate_failure_modes/report.json"
PREVIOUS_PATH = ROOT / "results/reopened_hybrid_memory_carrier_analysis/report.json"
OUTPUT = ROOT / "results/reopened_hybrid_slope_adaptive_carrier_analysis"

REFERENCES = {
    "SPARC_fixed_RAR_outer_RMSE_km_s": 10.681519066526649,
    "raw_baryons_RMS_arcsec": 27.43864684589079,
    "raw_simple_MOND_RMS_arcsec": 28.188493432162694,
    "raw_compact_halo_RMS_arcsec": 9.048410306058654,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setting_signature(settings: dict) -> str:
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    robust = json.loads(ROBUST_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
    failure = json.loads(FAILURE_PATH.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS_PATH.read_text(encoding="utf-8"))
    scores = pd.read_csv(SCORES_PATH)
    variants = {row["name"]: row for row in expand_variants(protocol)}
    if set(scores.variant) != set(variants):
        raise RuntimeError("full scores do not match the frozen protocol")

    setting_rows = []
    for name in scores.variant:
        settings = variants[name]["settings"]
        setting_rows.append(
            {
                "variant": name,
                "settings_json": setting_signature(settings),
                "gate_strength": float(
                    settings.get("radial_memory_slope_gate_strength", 0.0)
                ),
                "gate_pivot": float(
                    settings.get("radial_memory_slope_gate_pivot", -1.0)
                ),
                "gate_sharpness": float(
                    settings.get("radial_memory_slope_gate_sharpness", 1.0)
                ),
                "base_p": float(
                    settings.get("radial_memory_gbar_power", 0.0)
                ),
                "base_q": float(
                    settings.get("radial_memory_radius_power", 0.0)
                ),
                "steep_p": float(
                    settings.get(
                        "radial_memory_steep_gbar_power",
                        settings.get("radial_memory_gbar_power", 0.0),
                    )
                ),
                "steep_q": float(
                    settings.get(
                        "radial_memory_steep_radius_power",
                        settings.get("radial_memory_radius_power", 0.0),
                    )
                ),
            }
        )
    augmented = scores.merge(
        pd.DataFrame(setting_rows), on="variant", validate="one_to_one"
    )
    augmented["solar_all_pass"] = (
        augmented.Cassini_proxy_pass.astype(bool)
        & augmented.Earth_pass.astype(bool)
        & augmented.Mercury_pass.astype(bool)
    )
    augmented["raw_eight_start_RMS_arcsec"] = np.nan
    augmented["raw_eight_start_all_roots"] = pd.Series(
        pd.NA, index=augmented.index, dtype="boolean"
    )
    augmented["raw_eight_start_pooled_reduced_chi2"] = np.nan
    for name, comparison in robust["comparisons"].items():
        mask = augmented.variant.eq(name)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"missing robustness score for {name}")
        eight = comparison["eight_start"]
        augmented.loc[mask, "raw_eight_start_RMS_arcsec"] = eight[
            "equal_system_radial_RMS_arcsec"
        ]
        augmented.loc[mask, "raw_eight_start_all_roots"] = eight[
            "all_roots_converged"
        ]
        augmented.loc[mask, "raw_eight_start_pooled_reduced_chi2"] = eight[
            "pooled_reduced_chi2"
        ]
    stable = (
        augmented.solar_all_pass
        & augmented.raw_eight_start_all_roots.fillna(False).astype(bool)
    )
    augmented["robust_cross_domain_reference_ratio"] = np.nan
    augmented.loc[stable, "robust_cross_domain_reference_ratio"] = np.maximum(
        augmented.loc[stable, "SPARC_outer_RMSE_km_s"]
        / REFERENCES["SPARC_fixed_RAR_outer_RMSE_km_s"],
        augmented.loc[stable, "raw_eight_start_RMS_arcsec"]
        / REFERENCES["raw_compact_halo_RMS_arcsec"],
    )
    stable_rows = augmented[stable].sort_values(
        ["robust_cross_domain_reference_ratio", "bridge_RMSE_dex"]
    )
    best = stable_rows.iloc[0]
    best_raw = stable_rows.sort_values("raw_eight_start_RMS_arcsec").iloc[0]

    root_reversals = []
    for name, comparison in robust["comparisons"].items():
        two = comparison["two_start"]
        eight = comparison["eight_start"]
        if two["all_roots_converged"] != eight["all_roots_converged"]:
            root_reversals.append(
                {
                    "variant": name,
                    "two_start_all_roots": two["all_roots_converged"],
                    "eight_start_all_roots": eight[
                        "all_roots_converged"
                    ],
                    "two_start_RMS_arcsec": two[
                        "equal_system_radial_RMS_arcsec"
                    ],
                    "eight_start_RMS_arcsec": eight[
                        "equal_system_radial_RMS_arcsec"
                    ],
                }
            )

    duplicate_groups = []
    for signature, block in augmented.groupby("settings_json"):
        if len(block) < 2:
            continue
        robust_block = block[block.raw_eight_start_RMS_arcsec.notna()]
        parameters = [
            report["results"][name]["full_fit_parameters"]
            for name in block.variant
        ]
        parameter_ranges = {
            parameter: float(
                max(item[parameter] for item in parameters)
                - min(item[parameter] for item in parameters)
            )
            for parameter in protocol["universal_parameters"]["names"]
        }
        duplicate_groups.append(
            {
                "settings": json.loads(signature),
                "variants": block.variant.tolist(),
                "runs": len(block),
                "bridge_RMSE_span_dex": float(
                    block.bridge_RMSE_dex.max() - block.bridge_RMSE_dex.min()
                ),
                "SPARC_RMSE_span_km_s": float(
                    block.SPARC_outer_RMSE_km_s.max()
                    - block.SPARC_outer_RMSE_km_s.min()
                ),
                "two_start_RMS_span_arcsec": float(
                    block.raw_lensing_RMS_arcsec.max()
                    - block.raw_lensing_RMS_arcsec.min()
                ),
                "two_start_root_states": sorted(
                    set(bool(value) for value in block.raw_all_roots_converged)
                ),
                "eight_start_runs": len(robust_block),
                "eight_start_RMS_min_arcsec": (
                    float(robust_block.raw_eight_start_RMS_arcsec.min())
                    if len(robust_block)
                    else None
                ),
                "eight_start_RMS_max_arcsec": (
                    float(robust_block.raw_eight_start_RMS_arcsec.max())
                    if len(robust_block)
                    else None
                ),
                "eight_start_root_states": sorted(
                    set(
                        bool(value)
                        for value in robust_block.raw_eight_start_all_roots.dropna()
                    )
                ),
                "full_fit_parameter_ranges": parameter_ranges,
            }
        )

    sharpness = augmented[augmented.family.eq("gate_sharpness")].sort_values(
        "fixed_value"
    )
    sharpness_table = []
    for _, row in sharpness.iterrows():
        sharpness_table.append(
            {
                "sharpness": float(row.fixed_value),
                "SPARC_RMSE_km_s": float(row.SPARC_outer_RMSE_km_s),
                "bridge_RMSE_dex": float(row.bridge_RMSE_dex),
                "raw_two_start_RMS_arcsec": float(
                    row.raw_lensing_RMS_arcsec
                ),
                "raw_two_start_all_roots": bool(
                    row.raw_all_roots_converged
                ),
                "raw_eight_start_RMS_arcsec": (
                    float(row.raw_eight_start_RMS_arcsec)
                    if np.isfinite(row.raw_eight_start_RMS_arcsec)
                    else None
                ),
                "raw_eight_start_all_roots": (
                    bool(row.raw_eight_start_all_roots)
                    if pd.notna(row.raw_eight_start_all_roots)
                    else None
                ),
            }
        )

    prior_best = previous["best_eight_start_stable_compromise"]
    summary = {
        "status": "completed slope-adaptive carrier analysis",
        "coverage": {
            "fixed_parameter_variants": audit["rows"],
            "full_variants": len(augmented),
            "unique_full_settings": int(augmented.settings_json.nunique()),
            "eight_start_raw_replays": len(robust["selected_variants"]),
            "SPARC_gate_geometry_systems": geometry["distribution"]["SPARC"][
                "systems"
            ],
            "CLASH_gate_geometry_systems": geometry["distribution"]["CLASH"][
                "systems"
            ],
        },
        "references": REFERENCES,
        "formula": {
            "local_slope": "s=d ln(g_N)/d ln(r)",
            "gate": "w=mu_s logistic[k(s_pivot-s)]",
            "effective_p": "p_base+w(p_steep-p_base)",
            "effective_q": "q_base+w(q_steep-q_base)",
            "source": "X=F(g_N/g_ref)^p_eff(r/kpc)^q_eff",
        },
        "best_stable_stage_compromise": {
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
                    / prior_best["robust_cross_domain_reference_ratio"]
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
            "SPARC_ratio_to_RAR": float(
                best_raw.SPARC_outer_RMSE_km_s
                / REFERENCES["SPARC_fixed_RAR_outer_RMSE_km_s"]
            ),
        },
        "sharpness_transfer": sharpness_table,
        "geometry_mechanism": {
            "SPARC_fixed_endpoint_median_p90_abs_source_derivative": (
                geometry["distribution"]["SPARC"][
                    "fixed_steep_endpoint_p90_abs_log_factor_derivative"
                ]["median"]
            ),
            "SPARC_k0p25_median_p90_abs_source_derivative": (
                geometry["distribution"]["SPARC"]["gates"]["k_0p25"][
                    "p90_abs_log_factor_derivative"
                ]["median"]
            ),
            "SPARC_k4_median_p90_abs_source_derivative": (
                geometry["distribution"]["SPARC"]["gates"]["k_4"][
                    "p90_abs_log_factor_derivative"
                ]["median"]
            ),
            "SPARC_k4_p90_p90_abs_source_derivative": (
                geometry["distribution"]["SPARC"]["gates"]["k_4"][
                    "p90_abs_log_factor_derivative"
                ]["p90"]
            ),
            "CLASH_k4_median_p90_abs_source_derivative": (
                geometry["distribution"]["CLASH"]["gates"]["k_4"][
                    "p90_abs_log_factor_derivative"
                ]["median"]
            ),
        },
        "failure_mode_correlations": {
            "sharpness_16_top": failure[
                "strongest_per_galaxy_correlations"
            ]["sharpness_16"][:5],
            "sharpness_4_top": failure[
                "strongest_per_galaxy_correlations"
            ]["sharpness_4"][:5],
        },
        "duplicate_setting_groups": duplicate_groups,
        "lens_root_reversals": root_reversals,
        "solar_failures": augmented.loc[
            ~augmented.solar_all_pass, "variant"
        ].tolist(),
        "empirical_findings": [
            "A smooth local-slope gate mostly reduces to a global blended carrier and does not improve the prior fixed-carrier benchmark.",
            "Gate sharpness is a high-leverage anti-galaxy/lensing parameter: hard switching improves the stable raw-lens score while catastrophically worsening SPARC.",
            "Point-dependent exponents create radial source-factor gradients larger than either fixed endpoint, especially in SPARC; CLASH profiles remain smoother under the same formula.",
            "The hard-gate galaxy penalty is strongest in gas-rich, low-stellar-mass systems rather than being isolated to bulge-dominated systems.",
            "Exact formula duplicates can occupy different bridge-fit parameter branches; deeper lens optimization does not restore their missing roots.",
            "All tested slope-gate settings pass the current Solar proxies, so the dominant failure is cross-domain shape rather than high-acceleration leakage.",
        ],
        "claim_boundary": [
            "This disfavours pointwise exponent interpolation over the tested endpoints and ranges; it does not reject every slope-dependent or nonlocal law.",
            "The local derivative contains baryonic structure, measurement noise, and interpolation choices that are not separated here.",
            "Only 18 CLASH systems have multiple bridge radii; one-point bridge systems reduce exactly to the local law.",
            "Raw lensing remains a zero-slip pseudo-elliptical closure rather than covariant ray tracing.",
            "Solar results are force-fraction and first-order Mercury proxies, not a complete ephemeris or PPN fit.",
        ],
        "input_hashes": {
            "protocol": sha256(PROTOCOL_PATH),
            "scores": sha256(SCORES_PATH),
            "full_report": sha256(REPORT_PATH),
            "raw_robustness": sha256(ROBUST_PATH),
            "fixed_audit": sha256(AUDIT_PATH),
            "gate_geometry": sha256(GEOMETRY_PATH),
            "failure_modes": sha256(FAILURE_PATH),
            "previous_carrier_analysis": sha256(PREVIOUS_PATH),
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Local-slope adaptive carrier analysis",
        "",
        f"- Fixed-parameter variants: **{audit['rows']}**",
        f"- Full cross-domain variants: **{len(augmented)}**",
        f"- Eight-start raw-lens replays: **{len(robust['selected_variants'])}**",
        "",
        "## Outcome",
        "",
        f"The best stable adaptive setting is `{best.variant}`: {best.SPARC_outer_RMSE_km_s:.2f} km/s on SPARC and {best.raw_eight_start_RMS_arcsec:.2f} arcsec on raw lensing. Its worse reference ratio is {best.robust_cross_domain_reference_ratio:.3f}, so it does not beat the preceding fixed-carrier ratio of {prior_best['robust_cross_domain_reference_ratio']:.3f}.",
        "",
        f"The strongest stable raw-lens case is `{best_raw.variant}` at {best_raw.raw_eight_start_RMS_arcsec:.2f} arcsec, but its galaxy error is {best_raw.SPARC_outer_RMSE_km_s:.2f} km/s.",
        "",
        "Sharp pointwise exponent switching is therefore a high-impact lensing-versus-galaxy lever, not a unifying mechanism in the tested form. Smooth switching behaves mainly like another fixed blended carrier.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["coverage"], indent=2))
    print(json.dumps(summary["best_stable_stage_compromise"], indent=2))


if __name__ == "__main__":
    main()
