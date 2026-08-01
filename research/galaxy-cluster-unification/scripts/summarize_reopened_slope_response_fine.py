#!/usr/bin/env python3
"""Consolidate fine slope-response, pivot, and repeatability evidence."""

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


STAGES = {
    "fine": {
        "protocol": ROOT / "configs/reopened_hybrid_slope_response_fine_protocol.json",
        "scores": ROOT / "results/reopened_hybrid_slope_response_fine/scores.csv",
        "report": ROOT / "results/reopened_hybrid_slope_response_fine/report.json",
        "robust": ROOT / "results/reopened_hybrid_slope_response_fine_raw_robustness/report.json",
    },
    "pivot_extension": {
        "protocol": ROOT / "configs/reopened_hybrid_slope_response_pivot_extension_protocol.json",
        "scores": ROOT / "results/reopened_hybrid_slope_response_pivot_extension/scores.csv",
        "report": ROOT / "results/reopened_hybrid_slope_response_pivot_extension/report.json",
        "robust": ROOT / "results/reopened_hybrid_slope_response_pivot_extension_raw_robustness/report.json",
    },
    "repeatability": {
        "protocol": ROOT / "configs/reopened_hybrid_slope_response_best_repeatability_protocol.json",
        "scores": ROOT / "results/reopened_hybrid_slope_response_best_repeatability/scores.csv",
        "report": ROOT / "results/reopened_hybrid_slope_response_best_repeatability/report.json",
        "robust": ROOT / "results/reopened_hybrid_slope_response_best_repeatability_raw/report.json",
    },
}
OUTPUT = ROOT / "results/reopened_hybrid_slope_response_fine_analysis"
REFERENCES = {
    "SPARC_fixed_RAR_outer_RMSE_km_s": 10.681519066526649,
    "raw_baryons_RMS_arcsec": 27.43864684589079,
    "raw_simple_MOND_RMS_arcsec": 28.188493432162694,
    "raw_compact_halo_RMS_arcsec": 9.048410306058654,
    "prior_fixed_carrier_robust_ratio": 4.109246,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_row(row: pd.Series) -> dict[str, object]:
    return {
        "stage": row.stage,
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
    tables = []
    input_hashes: dict[str, str] = {}
    reports = {}
    robust_reports = {}
    settings_by_stage = {}
    for stage, paths in STAGES.items():
        protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        robust = json.loads(paths["robust"].read_text(encoding="utf-8"))
        scores = pd.read_csv(paths["scores"])
        variants = {row["name"]: row for row in expand_variants(protocol)}
        if set(scores.variant) != set(variants):
            raise RuntimeError(f"{stage} scores do not match its frozen protocol")
        scores.insert(0, "stage", stage)
        scores["settings_json"] = scores.variant.map(
            lambda name: json.dumps(variants[name]["settings"], sort_keys=True)
        )
        scores["raw_eight_start_RMS_arcsec"] = np.nan
        scores["raw_eight_start_all_roots"] = pd.Series(
            pd.NA, index=scores.index, dtype="boolean"
        )
        for name, comparison in robust["comparisons"].items():
            mask = scores.variant.eq(name)
            if int(mask.sum()) != 1:
                raise RuntimeError(f"{stage} robustness row is missing: {name}")
            replay = comparison["eight_start"]
            scores.loc[mask, "raw_eight_start_RMS_arcsec"] = replay[
                "equal_system_radial_RMS_arcsec"
            ]
            scores.loc[mask, "raw_eight_start_all_roots"] = replay[
                "all_roots_converged"
            ]
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
        tables.append(scores)
        reports[stage] = report
        robust_reports[stage] = robust
        settings_by_stage[stage] = variants
        for kind, path in paths.items():
            if kind != "scores":
                input_hashes[f"{stage}_{kind}"] = sha256(path)
        input_hashes[f"{stage}_scores"] = sha256(paths["scores"])

    combined = pd.concat(tables, ignore_index=True)
    stable = combined[np.isfinite(combined.robust_cross_domain_reference_ratio)]
    development = stable[stable.stage.isin(["fine", "pivot_extension"])]
    best_development = development.sort_values(
        ["robust_cross_domain_reference_ratio", "bridge_RMSE_dex"]
    ).iloc[0]

    repeat = combined[combined.stage.eq("repeatability")].copy()
    if repeat.settings_json.nunique() != 1:
        raise RuntimeError("repeatability rows are not the same structural formula")
    repeat_stable = repeat[
        np.isfinite(repeat.robust_cross_domain_reference_ratio)
    ]
    typical_ratio = float(repeat_stable.robust_cross_domain_reference_ratio.median())
    parameter_rows = []
    for name, result in reports["repeatability"]["results"].items():
        parameter_rows.append({"variant": name, **result["full_fit_parameters"]})
    parameters = pd.DataFrame(parameter_rows)
    parameter_ranges = {
        column: {
            "minimum": float(parameters[column].min()),
            "median": float(parameters[column].median()),
            "maximum": float(parameters[column].max()),
            "span": float(parameters[column].max() - parameters[column].min()),
        }
        for column in ["epsilon_0", "log10_rho_c_g_cm3", "Q", "B"]
    }

    root_reversals = []
    for stage, robust in robust_reports.items():
        for name, comparison in robust["comparisons"].items():
            two = comparison["two_start"]["all_roots_converged"]
            eight = comparison["eight_start"]["all_roots_converged"]
            if two != eight:
                root_reversals.append(
                    {
                        "stage": stage,
                        "variant": name,
                        "two_start_all_roots": bool(two),
                        "eight_start_all_roots": bool(eight),
                    }
                )

    summary = {
        "status": "completed fine slope-response and repeatability analysis",
        "coverage": {
            "full_universal_refits": int(len(combined)),
            "eight_start_raw_replays": int(
                sum(len(row["selected_variants"]) for row in robust_reports.values())
            ),
            "repeatability_refits_of_identical_formula": int(len(repeat)),
            "repeatability_stable_root_fraction": float(len(repeat_stable) / len(repeat)),
        },
        "references": REFERENCES,
        "best_observed_development_branch": compact_row(best_development),
        "identical_formula_repeatability": {
            "formula_settings": json.loads(repeat.settings_json.iloc[0]),
            "stable_refits": int(len(repeat_stable)),
            "total_refits": int(len(repeat)),
            "median_stable_SPARC_RMSE_km_s": float(
                repeat_stable.SPARC_outer_RMSE_km_s.median()
            ),
            "median_stable_raw_eight_start_RMS_arcsec": float(
                repeat_stable.raw_eight_start_RMS_arcsec.median()
            ),
            "median_stable_robust_reference_ratio": typical_ratio,
            "typical_improvement_vs_prior_fixed_carrier_percent": float(
                100.0
                * (1.0 - typical_ratio / REFERENCES["prior_fixed_carrier_robust_ratio"])
            ),
            "parameter_ranges": parameter_ranges,
        },
        "optimistic_branch_improvement_vs_prior_fixed_carrier_percent": float(
            100.0
            * (
                1.0
                - best_development.robust_cross_domain_reference_ratio
                / REFERENCES["prior_fixed_carrier_robust_ratio"]
            )
        ),
        "root_reversals": root_reversals,
        "solar_failures": combined.loc[
            ~combined.solar_all_pass.astype(bool), ["stage", "variant"]
        ].to_dict("records"),
        "empirical_findings": [
            "Moving the profile-response pivot from -1 toward 0 and shortening memory from two to about 0.8 natural-log radius units are the strongest stable local changes in this neighborhood.",
            "The best single development branch reaches a 3.962 worse-reference ratio, but exact structural repeats place the typical stable value at 3.982; the more conservative repeatability-adjusted improvement is about three percent.",
            "Four of five independent bridge fits of the same formula recover every raw-lens root. The fifth does not, so root existence is still branch-sensitive rather than structurally guaranteed.",
            "Galaxy and raw-lens scores are much more repeatable than the fitted bridge parameters. The observable response is partly identified, while epsilon, density threshold, Q, and the boundary-hitting Sigma amplitude are not uniquely identified.",
            "Every tested row passes the current Solar proxies; the important controls remain galaxy profile shape, raw lens-root topology, and optimizer branch selection.",
        ],
        "claim_boundary": [
            "The fine grid and pivot extension were selected after earlier development results and are not untouched holdouts.",
            "Repeatability measures numerical and bridge-parameter branch sensitivity for one structural formula; it is not a new-data replication.",
            "Only the tested pivot, scale, strength, and response-blend ranges are compared. Slope-dependent or nonlocal gravity as parent ideas remain open.",
            "Solar results remain force and first-order precession proxies rather than a complete ephemeris or PPN fit.",
        ],
        "input_hashes": input_hashes,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    parameters.to_csv(OUTPUT / "repeat_fit_parameters.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fine slope-response and repeatability results",
        "",
        f"Coverage: **{len(combined)}** universal refits and **{summary['coverage']['eight_start_raw_replays']}** eight-start raw-lens replays.",
        "",
        f"Best observed development branch: `{best_development.variant}` — {best_development.SPARC_outer_RMSE_km_s:.3f} km/s on SPARC, {best_development.raw_eight_start_RMS_arcsec:.3f} arcsec on raw lensing, and a {best_development.robust_cross_domain_reference_ratio:.3f} worse-reference ratio.",
        "",
        f"The exact-formula repeat gives **{len(repeat_stable)}/{len(repeat)}** stable root-complete fits. Its median stable ratio is **{typical_ratio:.3f}**, a **{summary['identical_formula_repeatability']['typical_improvement_vs_prior_fixed_carrier_percent']:.1f}%** improvement over the prior fixed carrier.",
        "",
        "The data support the pivot/shorter-memory direction, but the fitted gravity parameters and one lens-root branch are not uniquely identified.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
