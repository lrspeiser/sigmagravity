#!/usr/bin/env python3
"""Consolidate the generalized radial-memory carrier experiment."""

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


PROTOCOL_PATH = ROOT / "configs/reopened_hybrid_memory_carrier_protocol.json"
SCORES_PATH = ROOT / "results/reopened_hybrid_memory_carrier/scores.csv"
REPORT_PATH = ROOT / "results/reopened_hybrid_memory_carrier/report.json"
ROBUST_PATH = (
    ROOT / "results/reopened_hybrid_memory_carrier_raw_robustness/report.json"
)
SLOPE_NEUTRAL_ROBUST_PATH = (
    ROOT
    / "results/reopened_hybrid_memory_carrier_slope_neutral_raw_robustness/report.json"
)
FIXED_AUDIT_PATH = (
    ROOT / "results/reopened_hybrid_memory_carrier_audit/report.json"
)
SLOPE_NEUTRAL_AUDIT_PATH = (
    ROOT / "results/reopened_hybrid_memory_carrier_slope_neutral_audit/report.json"
)
SLOPE_PATH = ROOT / "results/reopened_profile_slope_audit/report.json"
OUTPUT = ROOT / "results/reopened_hybrid_memory_carrier_analysis"

REFERENCES = {
    "SPARC_fixed_RAR_outer_RMSE_km_s": 10.681519066526649,
    "raw_baryons_RMS_arcsec": 27.43864684589079,
    "raw_simple_MOND_RMS_arcsec": 28.188493432162694,
    "raw_compact_halo_RMS_arcsec": 9.048410306058654,
    "previous_stable_cross_domain_ratio": 5.429313,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def settings_signature(settings: dict) -> str:
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def get_row(frame: pd.DataFrame, variant: str) -> pd.Series:
    block = frame[frame.variant.eq(variant)]
    if len(block) != 1:
        raise RuntimeError(f"expected one score row for {variant}")
    return block.iloc[0]


def spearman(frame: pd.DataFrame, x: str, y: str) -> float:
    block = frame[[x, y]].dropna()
    return float(block[x].corr(block[y], method="spearman"))


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    robust = json.loads(ROBUST_PATH.read_text(encoding="utf-8"))
    slope_neutral_robust = json.loads(
        SLOPE_NEUTRAL_ROBUST_PATH.read_text(encoding="utf-8")
    )
    robustness_reports = [robust, slope_neutral_robust]
    fixed_audit = json.loads(FIXED_AUDIT_PATH.read_text(encoding="utf-8"))
    slope_neutral_audit = json.loads(
        SLOPE_NEUTRAL_AUDIT_PATH.read_text(encoding="utf-8")
    )
    slope = json.loads(SLOPE_PATH.read_text(encoding="utf-8"))
    scores = pd.read_csv(SCORES_PATH)

    variants = {row["name"]: row for row in expand_variants(protocol)}
    if set(scores.variant) != set(variants):
        raise RuntimeError("score variants do not match the frozen protocol")

    sparc_slope = float(slope["SPARC"]["median"])
    clash_slope = float(slope["CLASH"]["median"])
    setting_rows = []
    for variant in scores.variant:
        settings = variants[variant]["settings"]
        p = float(settings.get("radial_memory_gbar_power", 0.0))
        q = float(settings.get("radial_memory_radius_power", 0.0))
        setting_rows.append(
            {
                "variant": variant,
                "settings_json": settings_signature(settings),
                "memory_gbar_power_p": p,
                "memory_radius_power_q": q,
                "memory_channel_code": int(
                    settings.get("radial_memory_channel_code", 0)
                ),
                "memory_strength": float(
                    settings.get("radial_memory_strength", 0.0)
                ),
                "memory_log_scale": float(
                    settings.get("radial_memory_log_scale", 1.0)
                ),
                "memory_outer_to_inner": bool(
                    settings.get("radial_memory_outer_to_inner", False)
                ),
                "memory_pre_screen": bool(
                    settings.get("radial_memory_pre_screen", False)
                ),
                "SPARC_effective_radial_power": q + p * sparc_slope,
                "CLASH_effective_radial_power": q + p * clash_slope,
            }
        )
    settings_frame = pd.DataFrame(setting_rows)
    augmented = scores.merge(settings_frame, on="variant", validate="one_to_one")
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
    for robustness in robustness_reports:
        for variant, comparison in robustness["comparisons"].items():
            mask = augmented.variant.eq(variant)
            if int(mask.sum()) != 1:
                raise RuntimeError(f"missing robustness target {variant}")
            if augmented.loc[mask, "raw_eight_start_RMS_arcsec"].notna().any():
                raise RuntimeError(f"duplicate robustness target {variant}")
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
    augmented["robust_cross_domain_reference_ratio"] = np.nan
    stable_mask = (
        augmented.solar_all_pass
        & augmented.raw_eight_start_all_roots.fillna(False).astype(bool)
    )
    augmented.loc[stable_mask, "robust_cross_domain_reference_ratio"] = np.maximum(
        augmented.loc[stable_mask, "SPARC_outer_RMSE_km_s"]
        / REFERENCES["SPARC_fixed_RAR_outer_RMSE_km_s"],
        augmented.loc[stable_mask, "raw_eight_start_RMS_arcsec"]
        / REFERENCES["raw_compact_halo_RMS_arcsec"],
    )

    # Isolate the two-dimensional transport-exponent surface. Exact duplicate
    # settings from independent families are averaged so they do not receive
    # extra statistical weight.
    surface_mask = (
        augmented.memory_channel_code.eq(0)
        & augmented.memory_strength.eq(1.0)
        & augmented.memory_log_scale.eq(2.0)
        & ~augmented.memory_outer_to_inner
        & ~augmented.memory_pre_screen
    )
    surface = (
        augmented[surface_mask]
        .groupby(
            [
                "memory_gbar_power_p",
                "memory_radius_power_q",
                "SPARC_effective_radial_power",
                "CLASH_effective_radial_power",
            ],
            as_index=False,
        )
        .agg(
            formula_repeats=("variant", "size"),
            SPARC_outer_RMSE_km_s=("SPARC_outer_RMSE_km_s", "mean"),
            bridge_RMSE_dex=("bridge_RMSE_dex", "mean"),
            raw_two_start_RMS_arcsec=("raw_lensing_RMS_arcsec", "mean"),
            raw_two_start_root_fraction=("raw_all_roots_converged", "mean"),
        )
        .sort_values(["memory_gbar_power_p", "memory_radius_power_q"])
        .reset_index(drop=True)
    )
    correlations = {
        "p_vs_SPARC_RMSE": spearman(
            surface, "memory_gbar_power_p", "SPARC_outer_RMSE_km_s"
        ),
        "q_vs_SPARC_RMSE": spearman(
            surface, "memory_radius_power_q", "SPARC_outer_RMSE_km_s"
        ),
        "p_vs_bridge_RMSE": spearman(
            surface, "memory_gbar_power_p", "bridge_RMSE_dex"
        ),
        "q_vs_bridge_RMSE": spearman(
            surface, "memory_radius_power_q", "bridge_RMSE_dex"
        ),
        "SPARC_effective_power_vs_SPARC_RMSE": spearman(
            surface,
            "SPARC_effective_radial_power",
            "SPARC_outer_RMSE_km_s",
        ),
        "SPARC_effective_power_vs_bridge_RMSE": spearman(
            surface,
            "SPARC_effective_radial_power",
            "bridge_RMSE_dex",
        ),
        "CLASH_effective_power_vs_SPARC_RMSE": spearman(
            surface,
            "CLASH_effective_radial_power",
            "SPARC_outer_RMSE_km_s",
        ),
        "CLASH_effective_power_vs_bridge_RMSE": spearman(
            surface,
            "CLASH_effective_radial_power",
            "bridge_RMSE_dex",
        ),
    }

    robust_candidates = augmented[stable_mask].sort_values(
        ["robust_cross_domain_reference_ratio", "bridge_RMSE_dex"]
    )
    best = robust_candidates.iloc[0]
    best_raw = robust_candidates.sort_values("raw_eight_start_RMS_arcsec").iloc[0]
    previous_ratio = REFERENCES["previous_stable_cross_domain_ratio"]
    improvement = 100.0 * (
        1.0 - best.robust_cross_domain_reference_ratio / previous_ratio
    )

    duplicates = []
    robust_rows = augmented[augmented.raw_eight_start_RMS_arcsec.notna()].copy()
    for signature, block in robust_rows.groupby("settings_json"):
        if len(block) < 2:
            continue
        duplicates.append(
            {
                "settings": json.loads(signature),
                "variants": block.variant.tolist(),
                "runs": len(block),
                "eight_start_RMS_min_arcsec": float(
                    block.raw_eight_start_RMS_arcsec.min()
                ),
                "eight_start_RMS_max_arcsec": float(
                    block.raw_eight_start_RMS_arcsec.max()
                ),
                "eight_start_RMS_span_arcsec": float(
                    block.raw_eight_start_RMS_arcsec.max()
                    - block.raw_eight_start_RMS_arcsec.min()
                ),
                "all_root_states": sorted(
                    set(
                        bool(value)
                        for value in block.raw_eight_start_all_roots.dropna()
                    )
                ),
            }
        )

    root_reversals = []
    for robustness in robustness_reports:
        for variant, comparison in robustness["comparisons"].items():
            two = comparison["two_start"]
            eight = comparison["eight_start"]
            if two["all_roots_converged"] != eight["all_roots_converged"]:
                root_reversals.append(
                    {
                        "variant": variant,
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

    channel_comparisons = {}
    for family in [
        "dual_fractional_channel",
        "dual_acceleration_channel",
        "dual_speed_squared_channel",
    ]:
        rows = augmented[augmented.family.eq(family)].sort_values("fixed_value")
        channel_comparisons[family] = [
            {
                "channel_code": int(row.memory_channel_code),
                "SPARC_RMSE_km_s": float(row.SPARC_outer_RMSE_km_s),
                "bridge_RMSE_dex": float(row.bridge_RMSE_dex),
                "raw_two_start_RMS_arcsec": float(
                    row.raw_lensing_RMS_arcsec
                ),
                "raw_two_start_all_roots": bool(
                    row.raw_all_roots_converged
                ),
                "solar_all_pass": bool(row.solar_all_pass),
            }
            for _, row in rows.iterrows()
        ]

    solar_failures = augmented.loc[
        ~augmented.solar_all_pass,
        [
            "variant",
            "memory_gbar_power_p",
            "memory_radius_power_q",
            "memory_channel_code",
            "solar_maximum_fractional_change",
            "Mercury_precession_mas_per_century",
        ],
    ].to_dict(orient="records")

    summary = {
        "status": "completed generalized radial-memory carrier analysis",
        "coverage": {
            "full_formula_variants": len(augmented),
            "unique_full_formula_settings": int(
                augmented.settings_json.nunique()
            ),
            "fixed_parameter_audit_variants": fixed_audit["rows"],
            "fixed_parameter_audit_valid": fixed_audit["valid_rows"],
            "fixed_parameter_audit_solar_valid": fixed_audit[
                "solar_valid_rows"
            ],
            "slope_neutral_fixed_parameter_variants": slope_neutral_audit[
                "rows"
            ],
            "slope_neutral_fixed_parameter_solar_valid": (
                slope_neutral_audit["solar_valid_rows"]
            ),
            "eight_start_raw_replays": sum(
                len(item["selected_variants"])
                for item in robustness_reports
            ),
            "transport_power_surface_points": len(surface),
            "SPARC_profile_slopes": slope["SPARC"]["systems"],
            "CLASH_profile_slopes": slope["CLASH"]["systems"],
        },
        "references": REFERENCES,
        "formula": {
            "transported_quantity": "X = F (g_N/g_ref)^p (r/1 kpc)^q",
            "effective_radial_power": "q + p s when g_N is proportional to r^s",
            "channel_codes": {
                "0": "combined screened response",
                "1": "RG response before recombination",
                "2": "Sigma response before recombination",
                "3": "RG and Sigma independently before recombination",
            },
        },
        "measured_profile_slopes": {
            "SPARC_median_dln_gbar_dln_r": sparc_slope,
            "CLASH_median_dln_gbar_dln_r": clash_slope,
            "AUC_CLASH_slope_higher_than_SPARC": slope[
                "AUC_CLASH_slope_higher_than_SPARC"
            ],
        },
        "transport_power_surface_spearman": correlations,
        "best_eight_start_stable_compromise": {
            "variant": best.variant,
            "p": float(best.memory_gbar_power_p),
            "q": float(best.memory_radius_power_q),
            "SPARC_RMSE_km_s": float(best.SPARC_outer_RMSE_km_s),
            "bridge_RMSE_dex": float(best.bridge_RMSE_dex),
            "raw_eight_start_RMS_arcsec": float(
                best.raw_eight_start_RMS_arcsec
            ),
            "all_roots": bool(best.raw_eight_start_all_roots),
            "robust_cross_domain_reference_ratio": float(
                best.robust_cross_domain_reference_ratio
            ),
            "improvement_vs_previous_stable_ratio_percent": improvement,
            "any_universal_parameter_at_boundary": bool(
                best.any_universal_parameter_at_boundary
            ),
        },
        "best_eight_start_stable_raw_case_in_stage": {
            "variant": best_raw.variant,
            "p": float(best_raw.memory_gbar_power_p),
            "q": float(best_raw.memory_radius_power_q),
            "SPARC_RMSE_km_s": float(best_raw.SPARC_outer_RMSE_km_s),
            "raw_eight_start_RMS_arcsec": float(
                best_raw.raw_eight_start_RMS_arcsec
            ),
            "robust_cross_domain_reference_ratio": float(
                best_raw.robust_cross_domain_reference_ratio
            ),
        },
        "slope_neutral_balanced_candidate": {
            "variant": "dual_pminus1_radius:radial_memory_radius_power=-0.5",
            "p": -1.0,
            "q": -0.5,
            "SPARC_RMSE_km_s": float(
                get_row(
                    augmented,
                    "dual_pminus1_radius:radial_memory_radius_power=-0.5",
                ).SPARC_outer_RMSE_km_s
            ),
            "bridge_RMSE_dex": float(
                get_row(
                    augmented,
                    "dual_pminus1_radius:radial_memory_radius_power=-0.5",
                ).bridge_RMSE_dex
            ),
            "raw_eight_start_RMS_arcsec": float(
                get_row(
                    augmented,
                    "dual_pminus1_radius:radial_memory_radius_power=-0.5",
                ).raw_eight_start_RMS_arcsec
            ),
            "all_roots": bool(
                get_row(
                    augmented,
                    "dual_pminus1_radius:radial_memory_radius_power=-0.5",
                ).raw_eight_start_all_roots
            ),
            "robust_cross_domain_reference_ratio": float(
                get_row(
                    augmented,
                    "dual_pminus1_radius:radial_memory_radius_power=-0.5",
                ).robust_cross_domain_reference_ratio
            ),
            "raw_improvement_vs_baryons_percent": float(
                100.0
                * (
                    1.0
                    - get_row(
                        augmented,
                        "dual_pminus1_radius:radial_memory_radius_power=-0.5",
                    ).raw_eight_start_RMS_arcsec
                    / REFERENCES["raw_baryons_RMS_arcsec"]
                )
            ),
        },
        "channel_comparisons": channel_comparisons,
        "solar_failures": solar_failures,
        "duplicate_setting_replays": duplicates,
        "lens_root_reversals": root_reversals,
        "empirical_findings": [
            "Negative p and positive q emphasize galaxy outskirts and improve SPARC transfer over the tested surface, while degrading bridge and usually raw-lensing performance.",
            "Positive p carries added acceleration more directly: it improves raw lensing but sharply worsens galaxy rotation, making p an anti-correlated galaxy/lensing lever.",
            "The effective radial power q+p*s tracks the responses more strongly than p alone; CLASH bridge RMSE is especially ordered by the CLASH effective power.",
            "The galaxy benefit occurs when memory acts on the combined screened response. Applying fractional memory to RG or Sigma separately returns close to the local control.",
            "Channel-specific acceleration or speed-squared memory can violate Solar proxies even though combined-channel versions pass, so channel placement is physically consequential rather than a relabeling.",
            "Lens-root existence is nonmonotonic and optimizer-sensitive; eight-start replay reversed the apparent root status of several candidates.",
        ],
        "claim_boundary": [
            "These results disfavor only the tested carrier equations and ranges; they do not reject nonlocal or history-dependent gravity as a parent idea.",
            "The bridge cluster accelerations are derived profiles and only 18 CLASH systems contain multiple radii.",
            "The raw lens test uses a zero-slip pseudo-elliptical closure, not a covariant ray trace.",
            "The Solar tests are force-fraction and first-order Mercury proxies, not a full ephemeris or PPN fit.",
            "Correlations summarize a frozen exploratory exponent surface and are not an independent discovery significance.",
            "The stable best candidate reaches universal fit boundaries, so its parameter values are not identified within the current bounds.",
        ],
        "input_hashes": {
            "protocol": sha256(PROTOCOL_PATH),
            "scores": sha256(SCORES_PATH),
            "full_report": sha256(REPORT_PATH),
            "raw_robustness": sha256(ROBUST_PATH),
            "slope_neutral_raw_robustness": sha256(
                SLOPE_NEUTRAL_ROBUST_PATH
            ),
            "fixed_audit": sha256(FIXED_AUDIT_PATH),
            "slope_neutral_fixed_audit": sha256(
                SLOPE_NEUTRAL_AUDIT_PATH
            ),
            "profile_slopes": sha256(SLOPE_PATH),
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    surface.to_csv(OUTPUT / "power_surface.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Generalized radial-memory carrier analysis",
        "",
        f"- Full variants: **{len(augmented)}**",
        f"- Fixed-parameter screening variants: **{fixed_audit['rows']}**",
        f"- Eight-start lens replays: **{sum(len(item['selected_variants']) for item in robustness_reports)}**",
        f"- Unique exponent-surface points: **{len(surface)}**",
        "",
        "## Strongest stable compromise in this stage",
        "",
        f"`{best.variant}`",
        "",
        f"- Transport powers: p={best.memory_gbar_power_p:g}, q={best.memory_radius_power_q:g}",
        f"- Galaxy RMSE: {best.SPARC_outer_RMSE_km_s:.3f} km/s",
        f"- Bridge RMSE: {best.bridge_RMSE_dex:.3f} dex",
        f"- Eight-start raw lens RMS: {best.raw_eight_start_RMS_arcsec:.3f} arcsec; all roots recovered",
        f"- Worse reference ratio: {best.robust_cross_domain_reference_ratio:.3f}",
        f"- Improvement over the preceding stable ratio: {improvement:.1f}%",
        "",
        "## Data-derived leverage",
        "",
        "For a local baryonic slope g_N proportional to r^s, the transported source has effective radial power q+p*s. SPARC has a much steeper median slope than CLASH, so the same universal p changes the two domains differently without using an object label.",
        "",
        f"- SPARC median s: **{sparc_slope:.3f}**",
        f"- CLASH median s: **{clash_slope:.3f}**",
        f"- Spearman(effective SPARC power, galaxy RMSE): **{correlations['SPARC_effective_power_vs_SPARC_RMSE']:.3f}**",
        f"- Spearman(effective CLASH power, bridge RMSE): **{correlations['CLASH_effective_power_vs_bridge_RMSE']:.3f}**",
        "",
        "Negative p or positive q improves galaxy rotation over the tested surface but generally worsens cluster fitting. Positive p moves lensing in the useful direction while over-accelerating galaxies. The most impactful new knob is therefore the radial quantity transported, not merely memory strength.",
        "",
        "The result is a sensitivity map, not a solved theory. The best stable candidate remains 4.11 times its worse benchmark and its fitted universal parameters touch bounds.",
        "",
        "The approximately CLASH-neutral setting p=-1, q=-0.5 is a more balanced alternative: 47.70 km/s on galaxies, 0.204 dex on the bridge, and 25.50 arcsec with every lens root recovered. It improves raw lensing 7.1% over baryons while keeping most of the galaxy gain, but its worse reference ratio is 4.47 rather than 4.11.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["coverage"], indent=2))
    print(json.dumps(summary["best_eight_start_stable_compromise"], indent=2))


if __name__ == "__main__":
    main()
