#!/usr/bin/env python3
"""Consolidate exact-endpoint source-power and memory sensitivities."""

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


PROTOCOL = ROOT / "configs/reopened_hybrid_endpoint_power_memory_protocol.json"
SCORES = ROOT / "results/reopened_hybrid_endpoint_power_memory/scores.csv"
REPORT = ROOT / "results/reopened_hybrid_endpoint_power_memory/report.json"
ROBUST = ROOT / "results/reopened_hybrid_endpoint_power_memory_raw_robustness/report.json"
AUDIT_LOCAL = ROOT / "results/reopened_hybrid_endpoint_power_memory_audit/scores.csv"
AUDIT_EXTENSION = ROOT / "results/reopened_hybrid_endpoint_power_extension_audit/scores.csv"
PRIOR = ROOT / "results/reopened_hybrid_smoothed_local_pivot_extension_analysis/report.json"
OUTPUT = ROOT / "results/reopened_hybrid_endpoint_power_memory_analysis"
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


def setting_columns(frame: pd.DataFrame, lookup: dict[str, dict]) -> pd.DataFrame:
    result = frame.copy()
    settings = result.variant.map(lambda name: lookup[name]["settings"])
    result["p"] = settings.map(lambda row: float(row["radial_memory_gbar_power"]))
    result["q"] = settings.map(lambda row: float(row["radial_memory_radius_power"]))
    result["memory_log_scale"] = settings.map(
        lambda row: float(row["radial_memory_log_scale"])
    )
    result["memory_strength"] = settings.map(
        lambda row: float(row["radial_memory_strength"])
    )
    result["settings_json"] = settings.map(
        lambda row: json.dumps(row, sort_keys=True)
    )
    result["SPARC_effective_power"] = (
        result.q + result.p * SLOPES["SPARC_median_dln_gbar_dln_r"]
    )
    result["CLASH_effective_power"] = (
        result.q + result.p * SLOPES["CLASH_median_dln_gbar_dln_r"]
    )
    return result


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


def grouped_spans(
    frame: pd.DataFrame,
    group_column: str,
    metric: str,
) -> dict[str, float]:
    values = frame.groupby(group_column, sort=True)[metric].agg(span).astype(float)
    return {
        "groups": int(len(values)),
        "median_span": float(values.median()),
        "maximum_span": float(values.max()),
    }


def path_summary(frame: pd.DataFrame) -> dict[str, object]:
    ordered = frame.sort_values("p")
    return {
        "rows": int(len(ordered)),
        "p_range": [float(ordered.p.min()), float(ordered.p.max())],
        "q_range": [float(ordered.q.min()), float(ordered.q.max())],
        "SPARC_span_km_s": span(ordered.SPARC_outer_RMSE_km_s),
        "bridge_span_dex": span(ordered.bridge_RMSE_dex),
        "raw_eight_start_span_arcsec": span(ordered.raw_eight_start_RMS_arcsec),
        "SPARC_effective_power_span": span(ordered.SPARC_effective_power),
        "CLASH_effective_power_span": span(ordered.CLASH_effective_power),
        "sequence": [compact(row) for _, row in ordered.iterrows()],
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    robust = json.loads(ROBUST.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    variants = {row["name"]: row for row in expand_variants(protocol)}
    scores = pd.read_csv(SCORES)
    if set(scores.variant) != set(variants):
        raise RuntimeError("score rows do not match the frozen endpoint protocol")
    if set(robust["selected_variants"]) != set(scores.variant):
        raise RuntimeError("eight-start replay does not cover every endpoint row")
    scores = setting_columns(scores, variants)
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
        raise RuntimeError("no Solar-valid, root-complete endpoint row")
    best = stable_rows.iloc[0]
    best_raw = stable_rows.sort_values("raw_eight_start_RMS_arcsec").iloc[0]

    grid = scores[scores.family.str.startswith("grid_q_")].copy()
    correlations = []
    for coordinate in ["p", "q", "SPARC_effective_power", "CLASH_effective_power"]:
        for metric in [
            "SPARC_outer_RMSE_km_s",
            "bridge_RMSE_dex",
            "raw_eight_start_RMS_arcsec",
        ]:
            coefficient, p_value = spearmanr(grid[coordinate], grid[metric])
            correlations.append(
                {
                    "coordinate": coordinate,
                    "metric": metric,
                    "spearman_r": float(coefficient),
                    "two_sided_p_value_descriptive": float(p_value),
                }
            )

    p_axis = {
        metric: grouped_spans(grid, "q", metric)
        for metric in [
            "SPARC_outer_RMSE_km_s",
            "bridge_RMSE_dex",
            "raw_eight_start_RMS_arcsec",
        ]
    }
    q_axis = {
        metric: grouped_spans(grid, "p", metric)
        for metric in [
            "SPARC_outer_RMSE_km_s",
            "bridge_RMSE_dex",
            "raw_eight_start_RMS_arcsec",
        ]
    }
    scale_rows = scores[scores.family.str.startswith("scale_")]
    strength_rows = scores[scores.family.str.startswith("strength_")]
    scale_axis = {
        metric: grouped_spans(scale_rows, "family", metric)
        for metric in [
            "SPARC_outer_RMSE_km_s",
            "bridge_RMSE_dex",
            "raw_eight_start_RMS_arcsec",
        ]
    }
    strength_axis = {
        metric: {
            "groups": 1,
            "median_span": span(strength_rows[metric]),
            "maximum_span": span(strength_rows[metric]),
        }
        for metric in [
            "SPARC_outer_RMSE_km_s",
            "bridge_RMSE_dex",
            "raw_eight_start_RMS_arcsec",
        ]
    }
    axes = {
        "acceleration_power_p": p_axis,
        "radius_power_q": q_axis,
        "memory_log_scale": scale_axis,
        "memory_strength": strength_axis,
    }
    impact_rows = []
    normalizers = {
        "SPARC_outer_RMSE_km_s": REFERENCES["SPARC_fixed_RAR_outer_RMSE_km_s"],
        "bridge_RMSE_dex": REFERENCES["bridge_target_RMSE_dex"],
        "raw_eight_start_RMS_arcsec": REFERENCES["raw_compact_halo_RMS_arcsec"],
    }
    for name, axis in axes.items():
        normalized = {
            metric: axis[metric]["median_span"] / normalizers[metric]
            for metric in normalizers
        }
        impact_rows.append(
            {
                "coordinate": name,
                "normalized_median_spans": normalized,
                "maximum_normalized_median_span": float(max(normalized.values())),
            }
        )
    impact_rows.sort(
        key=lambda row: row["maximum_normalized_median_span"], reverse=True
    )

    constant_clash = scores[scores.family.str.startswith("constant_clash_")]
    constant_sparc = scores[scores.family.str.startswith("constant_sparc_")]
    duplicate_groups = []
    for settings_json, block in scores.groupby("settings_json", sort=False):
        if len(block) <= 1:
            continue
        duplicate_groups.append(
            {
                "settings": json.loads(settings_json),
                "copies": int(len(block)),
                "variants": block.variant.tolist(),
                "stable_copies": int(
                    (
                        block.solar_all_pass
                        & block.raw_eight_start_all_roots.astype(bool)
                    ).sum()
                ),
                "SPARC_span_km_s": span(block.SPARC_outer_RMSE_km_s),
                "bridge_span_dex": span(block.bridge_RMSE_dex),
                "raw_eight_start_span_arcsec": span(
                    block.raw_eight_start_RMS_arcsec
                ),
                "median_robust_reference_ratio": float(
                    block.robust_cross_domain_reference_ratio.median()
                ),
            }
        )
    duplicate_groups.sort(key=lambda row: row["copies"], reverse=True)

    parameter_ranges = {}
    for parameter in protocol["universal_parameters"]["names"]:
        values = np.array(
            [
                report["results"][name]["full_fit_parameters"][parameter]
                for name in scores.variant
            ],
            dtype=float,
        )
        parameter_ranges[parameter] = {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
            "span": float(np.ptp(values)),
        }

    audits = []
    for label, path in [
        ("local", AUDIT_LOCAL),
        ("extension", AUDIT_EXTENSION),
    ]:
        audit = pd.read_csv(path)
        audits.append(
            {
                "stage": label,
                "rows": int(len(audit)),
                "valid_rows": int(audit.valid.astype(bool).sum()),
                "solar_valid_rows": int(audit.solar_all_pass.astype(bool).sum()),
                "best_fixed_parameter_ratio": float(
                    audit.audit_worst_reference_ratio.min()
                ),
            }
        )

    summary = {
        "status": "completed exact-endpoint power-memory sensitivity analysis",
        "coverage": {
            "fixed_parameter_audit_rows": int(sum(row["rows"] for row in audits)),
            "universal_refits": int(len(scores)),
            "eight_start_raw_replays": int(len(robust["selected_variants"])),
            "stable_root_complete_rows": int(len(stable_rows)),
            "grid_rows": int(len(grid)),
        },
        "references": REFERENCES,
        "measured_profile_slopes": SLOPES,
        "formula": {
            "transported_source": "X=F (g_N/g_ref)^p (r/kpc)^q",
            "effective_power": "e=q+p*dln(g_N)/dln(r)",
            "memory": "inner-to-outer exponential running average in ln(r)",
            "slope_gate": "absent in every row",
        },
        "fixed_parameter_audits": audits,
        "best_stable_observed": {
            **compact(best),
            "improvement_vs_prior_best_observed_percent": float(
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
        "coordinate_impact_ranking": impact_rows,
        "coordinate_spans": axes,
        "grid_correlations": correlations,
        "constant_CLASH_effective_power_path": path_summary(constant_clash),
        "constant_SPARC_effective_power_path": path_summary(constant_sparc),
        "duplicate_setting_groups": duplicate_groups,
        "universal_parameter_ranges": parameter_ranges,
        "root_reversals": root_reversals,
        "solar_failures": scores.loc[~scores.solar_all_pass, "variant"].tolist(),
        "empirical_interpretation": [
            "The p coordinate is a differential lever because measured galaxy profiles are substantially steeper than measured CLASH profiles, while q shifts both effective powers equally.",
            "A constant-CLASH-effective-power path tests whether galaxy transfer can move without materially changing the cluster bridge; the opposite path is its falsifying control.",
            "Memory-scale and strength spans test whether source-power improvements are stable to the averaging kernel rather than tied to one exact kernel setting.",
            "Exact duplicate settings quantify optimizer non-identification; the best single branch is not treated as representative when repeats disagree.",
            "Only the tested endpoint formula and ranges are ranked; no broader nonlocal, history-dependent, void, or slope mechanism is rejected.",
        ],
        "claim_boundary": [
            "The stage remains sequential development-data analysis rather than a preregistered external holdout.",
            "Constant effective power uses median measured slopes and is approximate for individual systems.",
            "Raw lensing remains a zero-slip pseudo-elliptical transfer and Solar checks remain weak-field proxies.",
            "A useful empirical coordinate is not a microscopic derivation or evidence of novelty.",
        ],
        "input_hashes": {
            "protocol": sha256(PROTOCOL),
            "scores": sha256(SCORES),
            "report": sha256(REPORT),
            "robustness": sha256(ROBUST),
            "local_audit": sha256(AUDIT_LOCAL),
            "extension_audit": sha256(AUDIT_EXTENSION),
            "prior": sha256(PRIOR),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT / "augmented_scores.csv", index=False)
    pd.DataFrame(impact_rows).to_json(
        OUTPUT / "coordinate_impacts.json", orient="records", indent=2
    )
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Exact-endpoint power-memory sensitivity results",
        "",
        f"Coverage: **{sum(row['rows'] for row in audits)}** fixed rows, **{len(scores)}** universal refits, and **{len(robust['selected_variants'])}** eight-start raw-lens replays.",
        "",
        f"Best stable observed branch: `{best.variant}` - {best.SPARC_outer_RMSE_km_s:.3f} km/s on SPARC, {best.raw_eight_start_RMS_arcsec:.3f} arcsec on raw lensing, ratio {best.robust_cross_domain_reference_ratio:.3f}.",
        "",
        "The constant-domain paths and duplicate groups determine whether that branch reflects differential source weighting or optimizer variation.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
