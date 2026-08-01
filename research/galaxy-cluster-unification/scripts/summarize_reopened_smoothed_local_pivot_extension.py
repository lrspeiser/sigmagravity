#!/usr/bin/env python3
"""Consolidate the finite-pivot versus exact-endpoint experiment."""

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


PROTOCOL = ROOT / "configs/reopened_hybrid_smoothed_local_pivot_extension_protocol.json"
SCORES = ROOT / "results/reopened_hybrid_smoothed_local_pivot_extension/scores.csv"
REPORT = ROOT / "results/reopened_hybrid_smoothed_local_pivot_extension/report.json"
ROBUST = ROOT / "results/reopened_hybrid_smoothed_local_pivot_extension_raw_robustness/report.json"
PRIOR = ROOT / "results/reopened_hybrid_smoothed_local_slope_analysis/report.json"
OUTPUT = ROOT / "results/reopened_hybrid_smoothed_local_pivot_extension_analysis"
REFERENCES = {
    "SPARC_fixed_RAR_outer_RMSE_km_s": 10.681519066526649,
    "bridge_target_RMSE_dex": 0.139,
    "raw_baryons_RMS_arcsec": 27.43864684589079,
    "raw_simple_MOND_RMS_arcsec": 28.188493432162694,
    "raw_compact_halo_RMS_arcsec": 9.048410306058654,
}
EXACT_ENDPOINT = [
    "steep_endpoint_scale:radial_memory_log_scale=0.8",
    "endpoint_repeat_1:radial_memory_log_scale=0.8",
    "endpoint_repeat_2:radial_memory_log_scale=0.8",
    "endpoint_repeat_3:radial_memory_log_scale=0.8",
    "endpoint_repeat_4:radial_memory_log_scale=0.8",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_row(row: pd.Series) -> dict[str, object]:
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


def range_summary(values: pd.Series) -> dict[str, float]:
    clean = values.astype(float)
    return {
        "minimum": float(clean.min()),
        "median": float(clean.median()),
        "maximum": float(clean.max()),
        "span": float(clean.max() - clean.min()),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    robust = json.loads(ROBUST.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    scores = pd.read_csv(SCORES)
    variants = {row["name"]: row for row in expand_variants(protocol)}
    if set(scores.variant) != set(variants):
        raise RuntimeError("score rows do not match the frozen extension protocol")
    if set(robust["selected_variants"]) != set(scores.variant):
        raise RuntimeError("eight-start replay does not cover every extension row")

    scores["settings_json"] = scores.variant.map(
        lambda name: json.dumps(variants[name]["settings"], sort_keys=True)
    )
    scores["raw_eight_start_RMS_arcsec"] = np.nan
    scores["raw_eight_start_all_roots"] = pd.Series(
        pd.NA, index=scores.index, dtype="boolean"
    )
    scores["raw_eight_start_pooled_reduced_chi2"] = np.nan
    root_reversals = []
    for name, comparison in robust["comparisons"].items():
        mask = scores.variant.eq(name)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"missing score row: {name}")
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
        raise RuntimeError("no Solar-valid, root-complete extension row")
    best = stable_rows.iloc[0]
    best_raw = stable_rows.sort_values("raw_eight_start_RMS_arcsec").iloc[0]

    endpoint = scores[scores.variant.isin(EXACT_ENDPOINT)].copy()
    if len(endpoint) != len(EXACT_ENDPOINT):
        raise RuntimeError("the five exact endpoint repeats are incomplete")
    endpoint_stable = endpoint[
        endpoint.solar_all_pass
        & endpoint.raw_eight_start_all_roots.fillna(False).astype(bool)
    ]
    parameter_rows = []
    for name in EXACT_ENDPOINT:
        params = report["results"][name]["full_fit_parameters"]
        parameter_rows.append({"variant": name, **params})
    endpoint_parameters = pd.DataFrame(parameter_rows)
    parameter_spans = {
        column: range_summary(endpoint_parameters[column])
        for column in protocol["universal_parameters"]["names"]
    }
    endpoint_repeatability = {
        "copies": int(len(endpoint)),
        "stable_root_complete_copies": int(len(endpoint_stable)),
        "SPARC_RMSE_km_s": range_summary(endpoint.SPARC_outer_RMSE_km_s),
        "bridge_RMSE_dex": range_summary(endpoint.bridge_RMSE_dex),
        "raw_two_start_RMS_arcsec": range_summary(endpoint.raw_lensing_RMS_arcsec),
        "raw_eight_start_RMS_arcsec": range_summary(
            endpoint.raw_eight_start_RMS_arcsec
        ),
        "universal_parameter_ranges": parameter_spans,
        "all_parameters_at_boundary": bool(
            endpoint.any_universal_parameter_at_boundary.astype(bool).all()
        ),
    }

    pivot = scores[scores.family.eq("pivot_extension")].copy()
    pivot = pivot.sort_values("fixed_value")
    pivot_rows = [
        {
            "pivot": float(row.fixed_value),
            **compact_row(row),
        }
        for _, row in pivot.iterrows()
    ]
    pivot6 = pivot[np.isclose(pivot.fixed_value.astype(float), 6.0)].iloc[0]
    endpoint_median = {
        "SPARC_RMSE_km_s": float(endpoint.SPARC_outer_RMSE_km_s.median()),
        "bridge_RMSE_dex": float(endpoint.bridge_RMSE_dex.median()),
        "raw_eight_start_RMS_arcsec": float(
            endpoint.raw_eight_start_RMS_arcsec.median()
        ),
    }
    pivot6_vs_endpoint = {
        "finite_pivot_variant": pivot6.variant,
        "finite_pivot_minus_endpoint_median_SPARC_km_s": float(
            pivot6.SPARC_outer_RMSE_km_s - endpoint_median["SPARC_RMSE_km_s"]
        ),
        "finite_pivot_minus_endpoint_median_bridge_dex": float(
            pivot6.bridge_RMSE_dex - endpoint_median["bridge_RMSE_dex"]
        ),
        "finite_pivot_minus_endpoint_median_raw_arcsec": float(
            pivot6.raw_eight_start_RMS_arcsec
            - endpoint_median["raw_eight_start_RMS_arcsec"]
        ),
    }

    family_impacts = []
    for family, block in scores.groupby("family", sort=False):
        family_impacts.append(
            {
                "family": family,
                "rows": int(len(block)),
                "SPARC_span_km_s": float(np.ptp(block.SPARC_outer_RMSE_km_s)),
                "bridge_span_dex": float(np.ptp(block.bridge_RMSE_dex)),
                "raw_eight_start_span_arcsec": float(
                    np.ptp(block.raw_eight_start_RMS_arcsec)
                ),
                "eight_start_root_states": int(
                    block.raw_eight_start_all_roots.astype(bool).nunique()
                ),
            }
        )
    impacts = pd.DataFrame(family_impacts)
    impacts["maximum_normalized_span"] = np.maximum.reduce(
        [
            impacts.SPARC_span_km_s
            / REFERENCES["SPARC_fixed_RAR_outer_RMSE_km_s"],
            impacts.bridge_span_dex / REFERENCES["bridge_target_RMSE_dex"],
            impacts.raw_eight_start_span_arcsec
            / REFERENCES["raw_compact_halo_RMS_arcsec"],
        ]
    )
    impacts = impacts.sort_values("maximum_normalized_span", ascending=False)

    summary = {
        "status": "completed finite-pivot versus exact-endpoint analysis",
        "coverage": {
            "universal_refits": int(len(scores)),
            "eight_start_raw_replays": int(len(robust["selected_variants"])),
            "stable_root_complete_rows": int(len(stable_rows)),
            "exact_endpoint_independent_refits": int(len(endpoint)),
        },
        "references": REFERENCES,
        "formula_tested": {
            "finite_pivot": "smoothed local slope selects a bounded blend of two completed radial-memory responses",
            "exact_endpoint": "the steep completed response is selected everywhere; no slope coordinate remains",
        },
        "best_stable_compromise": {
            **compact_row(best),
            "improvement_vs_prior_smoothed_local_best_percent": float(
                100.0
                * (
                    1.0
                    - best.robust_cross_domain_reference_ratio
                    / prior["best_stable_compromise"][
                        "robust_cross_domain_reference_ratio"
                    ]
                )
            ),
        },
        "best_stable_raw_case": {
            **compact_row(best_raw),
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
        "finite_pivot_sequence": pivot_rows,
        "exact_endpoint_repeatability": endpoint_repeatability,
        "pivot6_vs_exact_endpoint_median": pivot6_vs_endpoint,
        "parameter_impact_ranking": impacts.to_dict("records"),
        "root_reversals": root_reversals,
        "solar_failures": scores.loc[~scores.solar_all_pass, "variant"].tolist(),
        "empirical_interpretation": [
            "The exact endpoint control determines whether the apparent high-pivot gain requires a slope coordinate at all.",
            "Agreement between saturated finite pivots and the exact endpoint is evidence that the present data prefer the endpoint response, not evidence for slope-dependent physics.",
            "Disagreement among exact duplicate refits measures bridge-optimizer non-identification and is not structural formula leverage.",
            "Only the tested pivot, bandwidth, memory, and endpoint equations are compared; no parent nonlocal or profile-dependent mechanism is rejected.",
        ],
        "claim_boundary": [
            "This remains a sequential development-data experiment rather than a preregistered external holdout.",
            "The raw calculation is a zero-slip pseudo-elliptical transfer and the Solar checks are weak-field proxies.",
            "The bridge parameters remain bounded and partly nonidentified; prediction repeatability is more informative than any single fitted constant.",
            "An endpoint preference cannot identify a microscopic cause without a field equation and mechanism-specific data.",
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
    endpoint_parameters.to_csv(OUTPUT / "endpoint_parameters.csv", index=False)
    impacts.to_csv(OUTPUT / "parameter_impacts.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Finite-pivot versus exact-endpoint results",
        "",
        f"Coverage: **{len(scores)}** universal refits and **{len(robust['selected_variants'])}** eight-start raw-lens replays.",
        "",
        f"Best stable observed row: `{best.variant}` - {best.SPARC_outer_RMSE_km_s:.3f} km/s on SPARC, {best.raw_eight_start_RMS_arcsec:.3f} arcsec on raw lensing, ratio {best.robust_cross_domain_reference_ratio:.3f}.",
        "",
        f"The exact endpoint was independently refitted {len(endpoint)} times; {len(endpoint_stable)} copies retained every raw-lens root.",
        "",
        "Interpret the saturated-pivot comparison as a mechanism check: matching endpoint performance means the slope coordinate is unnecessary in this tested formula, not that all slope-dependent physics fails.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
