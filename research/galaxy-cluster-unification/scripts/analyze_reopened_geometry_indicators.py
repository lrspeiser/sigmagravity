#!/usr/bin/env python3
"""Audit label-free baryonic geometry indicators before using them as gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
G_SI = 6.67430e-11
M_SUN_KG = 1.988409870698051e30
KPC_M = 3.085677581491367e19
AU_M = 149_597_870_700.0
BRIDGE_PATH = ROOT / "results/phenomenology_formula_sweep/sample.csv"
SPARC_PATH = ROOT / "results/sparc_density_transfer/primary_predictions.csv"
OUTPUT = ROOT / "results/reopened_geometry_indicator_audit"

INDICATORS = [
    "log10_gbar_m_s2",
    "log10_local_density_g_cm3",
    "log10_radius_kpc",
    "log10_tidal_curvature_s2",
    "log10_equivalent_enclosed_mass_msun",
    "log10_local_to_mean_density_ratio",
    "source_concentration",
    "equivalent_mass_log_slope",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def add_profile_slope(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["equivalent_mass_log_slope"] = np.nan
    output["source_concentration"] = np.nan
    for _, indices in output.groupby(["domain", "system"]).groups.items():
        block = output.loc[indices].sort_values("radius_kpc")
        radius = block.radius_kpc.to_numpy(float)
        if len(radius) < 3 or len(np.unique(radius)) < 3:
            continue
        log_radius = np.log(radius)
        log_mass = (
            np.log(block.gbar_m_s2.to_numpy(float))
            + 2.0 * log_radius
        )
        slope = np.gradient(log_mass, log_radius, edge_order=2)
        slope = np.clip(slope, 0.0, 3.0)
        output.loc[block.index, "equivalent_mass_log_slope"] = slope
        output.loc[block.index, "source_concentration"] = 1.0 / (1.0 + slope)
    return output


def add_local_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    radius_m = output.radius_kpc.to_numpy(float) * KPC_M
    gbar = output.gbar_m_s2.to_numpy(float)
    density = output.local_density_g_cm3.to_numpy(float)
    mean_density = 3.0 * gbar / (4.0 * math.pi * G_SI * radius_m) * 1.0e-3
    ratio = density / np.maximum(mean_density, np.finfo(float).tiny)
    equivalent_mass = gbar * np.square(radius_m) / (G_SI * M_SUN_KG)
    output["log10_gbar_m_s2"] = np.log10(gbar)
    output["log10_local_density_g_cm3"] = np.log10(density)
    output["log10_radius_kpc"] = np.log10(output.radius_kpc.to_numpy(float))
    output["log10_tidal_curvature_s2"] = np.log10(gbar / radius_m)
    output["log10_equivalent_enclosed_mass_msun"] = np.log10(equivalent_mass)
    output["log10_local_to_mean_density_ratio"] = np.log10(
        np.maximum(ratio, np.finfo(float).tiny)
    )
    return add_profile_slope(output)


def load_profiles() -> pd.DataFrame:
    bridge = pd.read_csv(BRIDGE_PATH)
    bridge_frame = pd.DataFrame(
        {
            "domain": bridge.domain.replace({"cluster": "CLASH"}),
            "system": bridge.system.astype(str),
            "radius_kpc": bridge.radius_kpc.astype(float),
            "gbar_m_s2": np.power(10.0, bridge.log_gbar.astype(float)),
            "gobs_m_s2": np.power(10.0, bridge.log_gobs.astype(float)),
            "local_density_g_cm3": bridge.local_density_g_cm3.astype(float),
        }
    )
    bridge_frame["source"] = "bridge"

    sparc = pd.read_csv(SPARC_PATH)
    radius = sparc.radius_adjusted_kpc.to_numpy(float)
    observed_velocity_m_s = (
        sparc.velocity_observed_adjusted_kms.to_numpy(float) * 1000.0
    )
    sparc_frame = pd.DataFrame(
        {
            "domain": "SPARC",
            "system": sparc.galaxy.astype(str),
            "radius_kpc": radius,
            "gbar_m_s2": sparc.g_bar_m_s2.astype(float),
            "gobs_m_s2": np.square(observed_velocity_m_s) / (radius * KPC_M),
            "local_density_g_cm3": sparc.local_density_g_cm3.astype(float),
            "source": "SPARC outer transfer",
        }
    )
    combined = pd.concat([bridge_frame, sparc_frame], ignore_index=True)
    if np.any(combined.gbar_m_s2 <= 0.0):
        raise RuntimeError("all baryonic accelerations must be positive")
    if np.any(combined.local_density_g_cm3 <= 0.0):
        raise RuntimeError("all local densities must be positive")
    combined["required_log10_enhancement"] = np.log10(
        combined.gobs_m_s2 / combined.gbar_m_s2
    )
    return add_local_indicators(combined)


def auc_and_threshold(
    negative_values: np.ndarray,
    positive_values: np.ndarray,
) -> dict:
    negative = np.asarray(negative_values, dtype=float)
    positive = np.asarray(positive_values, dtype=float)
    negative = negative[np.isfinite(negative)]
    positive = positive[np.isfinite(positive)]
    if len(negative) == 0 or len(positive) == 0:
        return {
            "negative_count": len(negative),
            "positive_count": len(positive),
            "raw_auc_positive_high": math.nan,
            "separation_auc": math.nan,
            "best_direction": None,
            "descriptive_threshold": math.nan,
            "descriptive_balanced_accuracy": math.nan,
        }
    values = np.concatenate([negative, positive])
    ranks = rankdata(values, method="average")
    positive_rank_sum = float(np.sum(ranks[len(negative) :]))
    raw_auc = (
        positive_rank_sum - len(positive) * (len(positive) + 1.0) / 2.0
    ) / (len(negative) * len(positive))
    quantiles = np.unique(np.quantile(values, np.linspace(0.01, 0.99, 199)))
    best = (-math.inf, math.nan, "positive_high")
    for threshold in quantiles:
        for direction in ("positive_high", "positive_low"):
            if direction == "positive_high":
                true_positive = np.mean(positive >= threshold)
                true_negative = np.mean(negative < threshold)
            else:
                true_positive = np.mean(positive <= threshold)
                true_negative = np.mean(negative > threshold)
            balanced = 0.5 * (true_positive + true_negative)
            if balanced > best[0]:
                best = (float(balanced), float(threshold), direction)
    return {
        "negative_count": len(negative),
        "positive_count": len(positive),
        "raw_auc_positive_high": raw_auc,
        "separation_auc": max(raw_auc, 1.0 - raw_auc),
        "best_direction": best[2],
        "descriptive_threshold": best[1],
        "descriptive_balanced_accuracy": best[0],
    }


def separation_table(
    points: pd.DataFrame,
    systems: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    comparisons = [("SPARC", "CLASH"), ("SPARC", "BCG"), ("BCG", "CLASH")]
    for level, frame in (("point", points), ("equal_system_median", systems)):
        for negative, positive in comparisons:
            for indicator in INDICATORS:
                result = auc_and_threshold(
                    frame.loc[frame.domain == negative, indicator].to_numpy(float),
                    frame.loc[frame.domain == positive, indicator].to_numpy(float),
                )
                rows.append(
                    {
                        "level": level,
                        "negative_domain": negative,
                        "positive_domain": positive,
                        "indicator": indicator,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def distribution_table(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    for domain, block in frame.groupby("domain", sort=False):
        for indicator in INDICATORS:
            values = block[indicator].to_numpy(float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "level": level,
                    "domain": domain,
                    "indicator": indicator,
                    "count": len(values),
                    "q10": np.quantile(values, 0.10) if len(values) else math.nan,
                    "median": np.median(values) if len(values) else math.nan,
                    "q90": np.quantile(values, 0.90) if len(values) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def boost_correlations(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, block in points.groupby("domain", sort=False):
        for indicator in INDICATORS:
            valid = block[[indicator, "required_log10_enhancement"]].dropna()
            correlation = (
                float(spearmanr(valid[indicator], valid.required_log10_enhancement).statistic)
                if len(valid) >= 3
                else math.nan
            )
            rows.append(
                {
                    "domain": domain,
                    "indicator": indicator,
                    "points": len(valid),
                    "spearman_r_with_required_log10_enhancement": correlation,
                }
            )
    return pd.DataFrame(rows)


def solar_ranges() -> dict:
    solar_radius_m = 6.957e8
    radius_m = np.geomspace(1.6 * solar_radius_m, 8.43 * AU_M, 1000)
    gbar = G_SI * 1.988409870698051e30 / np.square(radius_m)
    density_g_cm3 = 1.0e-30
    mean_density = 3.0 * gbar / (4.0 * math.pi * G_SI * radius_m) * 1.0e-3
    ratio = density_g_cm3 / mean_density
    return {
        "radius_range_kpc": [
            float(radius_m.min() / KPC_M),
            float(radius_m.max() / KPC_M),
        ],
        "log10_gbar_m_s2_range": [
            float(np.log10(gbar).min()),
            float(np.log10(gbar).max()),
        ],
        "log10_tidal_curvature_s2_range": [
            float(np.log10(gbar / radius_m).min()),
            float(np.log10(gbar / radius_m).max()),
        ],
        "log10_equivalent_enclosed_mass_msun_range": [
            0.0,
            0.0,
        ],
        "log10_local_to_mean_density_ratio_range": [
            float(np.log10(ratio).min()),
            float(np.log10(ratio).max()),
        ],
        "source_concentration_point_mass": 1.0,
        "equivalent_mass_log_slope_point_mass": 0.0,
    }


def main() -> None:
    points = load_profiles()
    systems = (
        points.groupby(["domain", "system"], as_index=False)[
            INDICATORS + ["required_log10_enhancement"]
        ]
        .median(numeric_only=True)
    )
    separations = separation_table(points, systems)
    distributions = pd.concat(
        [
            distribution_table(points, "point"),
            distribution_table(systems, "equal_system_median"),
        ],
        ignore_index=True,
    )
    correlations = boost_correlations(points)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    points.to_csv(OUTPUT / "point_indicators.csv", index=False)
    systems.to_csv(OUTPUT / "system_indicator_medians.csv", index=False)
    separations.to_csv(OUTPUT / "separation_scores.csv", index=False)
    distributions.to_csv(OUTPUT / "indicator_distributions.csv", index=False)
    correlations.to_csv(OUTPUT / "boost_correlations.csv", index=False)

    system_sparc_clash = separations[
        (separations.level == "equal_system_median")
        & (separations.negative_domain == "SPARC")
        & (separations.positive_domain == "CLASH")
    ].sort_values("separation_auc", ascending=False)
    report = {
        "report_version": "REOPENED-GEOMETRY-INDICATOR-AUDIT-0.1.0",
        "status": "completed label-free geometry indicator audit",
        "input_hashes": {
            str(BRIDGE_PATH.relative_to(ROOT)).replace("\\", "/"): sha256(BRIDGE_PATH),
            str(SPARC_PATH.relative_to(ROOT)).replace("\\", "/"): sha256(SPARC_PATH),
        },
        "coverage": {
            "point_rows": len(points),
            "systems": int(systems.system.nunique()),
            "systems_by_domain": systems.groupby("domain").size().to_dict(),
        },
        "solar_ranges": solar_ranges(),
        "system_level_SPARC_vs_CLASH_ranking": system_sparc_clash.to_dict(
            orient="records"
        ),
        "claim_boundary": [
            "Object labels are used only to audit separation and are not inputs to any proposed formula.",
            "Thresholds maximize descriptive balanced accuracy on the same development sample and are not validation claims.",
            "Local density and spherical-equivalent mass indicators inherit the profile approximations used by the existing bridge and SPARC transfer data.",
            "The BCG sample mostly lacks multi-radius profiles, so profile-slope indicators are unavailable for that domain.",
            "A useful separator is only a candidate gate; it must still improve held-out bridge, galaxy, raw-lensing, and Solar tests.",
        ],
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Reopened geometry-indicator audit",
        "",
        "All indicators use baryonic acceleration, local baryonic density, radius, "
        "or radial changes of those quantities. Object labels are used only after "
        "calculation to measure separation.",
        "",
        "## Equal-system SPARC versus CLASH separation",
        "",
        "| rank | indicator | AUC | balanced accuracy | CLASH direction | threshold |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for rank, row in enumerate(system_sparc_clash.itertuples(), 1):
        lines.append(
            f"| {rank} | {row.indicator} | {row.separation_auc:.3f} | "
            f"{row.descriptive_balanced_accuracy:.3f} | "
            f"{row.best_direction} | {row.descriptive_threshold:.4g} |"
        )
    lines += [
        "",
        "These are discovery-sample diagnostics, not evidence that a gravity gate works. "
        "Advancement requires a frozen formula and unchanged cross-domain tests.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "top_system_level_SPARC_vs_CLASH": (
                    system_sparc_clash.iloc[0].to_dict()
                ),
            },
            indent=2,
            default=json_safe,
        )
    )


if __name__ == "__main__":
    main()
