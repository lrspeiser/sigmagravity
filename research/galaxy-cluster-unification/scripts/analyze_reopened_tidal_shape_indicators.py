#!/usr/bin/env python3
"""Audit dimensionless tidal-shape invariants on current baryonic profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_reopened_geometry_indicators import (  # noqa: E402
    auc_and_threshold,
    json_safe,
    load_profiles,
)
from voidscreen.tensor_completion import (  # noqa: E402
    axisymmetric_tidal_eigenvalues,
    spherical_tidal_eigenvalues,
)


INDICATORS = [
    "tidal_traceless_fraction",
    "tidal_trace_fraction",
    "tidal_l1_dominance",
    "tidal_middle_to_max",
    "tidal_minimum_to_max",
    "tidal_positive_fraction",
    "tidal_signed_determinant_shape",
    "tidal_radial_abs_fraction",
    "tidal_third_axis_abs_fraction",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eigenvalues_for_profiles(
    points: pd.DataFrame,
    sparc_method: str = "axisymmetric_midplane_density_closure",
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues = np.full((len(points), 3), np.nan, dtype=float)
    methods = np.full(len(points), "", dtype=object)
    spherical_mask = points.domain.isin(["BCG", "CLASH"]).to_numpy()
    spherical = points.loc[spherical_mask]
    eigenvalues[spherical_mask] = spherical_tidal_eigenvalues(
        spherical.gbar_m_s2.to_numpy(float),
        spherical.radius_kpc.to_numpy(float),
        spherical.local_density_g_cm3.to_numpy(float),
    )
    methods[spherical_mask] = "spherical_density_closure"

    sparc_mask = (points.domain == "SPARC").to_numpy()
    sparc = points.loc[sparc_mask]
    if sparc_method == "spherical_density_closure":
        eigenvalues[sparc_mask] = spherical_tidal_eigenvalues(
            sparc.gbar_m_s2.to_numpy(float),
            sparc.radius_kpc.to_numpy(float),
            sparc.local_density_g_cm3.to_numpy(float),
        )
        methods[sparc_mask] = sparc_method
    elif sparc_method == "axisymmetric_midplane_density_closure":
        for _, indices in sparc.groupby("system").groups.items():
            block = points.loc[indices].sort_values("radius_kpc")
            if (
                len(block) >= 2
                and len(np.unique(block.radius_kpc.to_numpy(float))) == len(block)
            ):
                values = axisymmetric_tidal_eigenvalues(
                    block.gbar_m_s2.to_numpy(float),
                    block.radius_kpc.to_numpy(float),
                    block.local_density_g_cm3.to_numpy(float),
                )
                eigenvalues[block.index] = values
                methods[block.index] = sparc_method
            else:
                values = spherical_tidal_eigenvalues(
                    block.gbar_m_s2.to_numpy(float),
                    block.radius_kpc.to_numpy(float),
                    block.local_density_g_cm3.to_numpy(float),
                )
                eigenvalues[block.index] = values
                methods[block.index] = "spherical_fallback"
    else:
        raise ValueError(f"unknown SPARC tidal geometry method {sparc_method}")
    if np.any(~np.isfinite(eigenvalues)):
        raise RuntimeError("tidal eigenvalues remain missing")
    return eigenvalues, methods


def add_invariants(
    points: pd.DataFrame,
    sparc_method: str = "axisymmetric_midplane_density_closure",
) -> pd.DataFrame:
    output = points.copy()
    eigenvalues, methods = eigenvalues_for_profiles(output, sparc_method)
    absolute = np.abs(eigenvalues)
    l1 = np.sum(absolute, axis=1)
    l2 = np.linalg.norm(eigenvalues, axis=1)
    mean = np.mean(eigenvalues, axis=1, keepdims=True)
    traceless = eigenvalues - mean
    sorted_absolute = np.sort(absolute, axis=1)
    output["tidal_eigenvalue_1_s2"] = eigenvalues[:, 0]
    output["tidal_eigenvalue_2_s2"] = eigenvalues[:, 1]
    output["tidal_eigenvalue_3_s2"] = eigenvalues[:, 2]
    output["tidal_geometry_method"] = methods
    output["tidal_traceless_fraction"] = (
        np.linalg.norm(traceless, axis=1) / np.maximum(l2, np.finfo(float).tiny)
    )
    output["tidal_trace_fraction"] = np.abs(np.sum(eigenvalues, axis=1)) / (
        math.sqrt(3.0) * np.maximum(l2, np.finfo(float).tiny)
    )
    output["tidal_l1_dominance"] = np.max(absolute, axis=1) / np.maximum(
        l1, np.finfo(float).tiny
    )
    output["tidal_middle_to_max"] = sorted_absolute[:, 1] / np.maximum(
        sorted_absolute[:, 2], np.finfo(float).tiny
    )
    output["tidal_minimum_to_max"] = sorted_absolute[:, 0] / np.maximum(
        sorted_absolute[:, 2], np.finfo(float).tiny
    )
    output["tidal_positive_fraction"] = np.sum(
        np.maximum(eigenvalues, 0.0), axis=1
    ) / np.maximum(l1, np.finfo(float).tiny)
    output["tidal_signed_determinant_shape"] = np.prod(
        eigenvalues, axis=1
    ) / np.maximum(np.power(l2, 3.0), np.finfo(float).tiny)
    output["tidal_radial_abs_fraction"] = absolute[:, 0] / np.maximum(
        l1, np.finfo(float).tiny
    )
    output["tidal_third_axis_abs_fraction"] = absolute[:, 2] / np.maximum(
        l1, np.finfo(float).tiny
    )
    return output


def separation_table(points: pd.DataFrame, systems: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [("SPARC", "CLASH"), ("SPARC", "BCG"), ("BCG", "CLASH")]
    for level, frame in (("point", points), ("equal_system_median", systems)):
        for negative, positive in comparisons:
            for indicator in INDICATORS:
                rows.append(
                    {
                        "level": level,
                        "negative_domain": negative,
                        "positive_domain": positive,
                        "indicator": indicator,
                        **auc_and_threshold(
                            frame.loc[
                                frame.domain == negative, indicator
                            ].to_numpy(float),
                            frame.loc[
                                frame.domain == positive, indicator
                            ].to_numpy(float),
                        ),
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
                    "q10": np.quantile(values, 0.1),
                    "median": np.median(values),
                    "q90": np.quantile(values, 0.9),
                }
            )
    return pd.DataFrame(rows)


def correlation_table(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, block in points.groupby("domain", sort=False):
        for indicator in INDICATORS:
            valid = block[[indicator, "required_log10_enhancement"]].dropna()
            rows.append(
                {
                    "domain": domain,
                    "indicator": indicator,
                    "points": len(valid),
                    "spearman_r_with_required_log10_enhancement": float(
                        spearmanr(
                            valid[indicator],
                            valid.required_log10_enhancement,
                        ).statistic
                    ),
                }
            )
    return pd.DataFrame(rows)


def solar_invariants() -> dict:
    eigenvalues = np.array([-2.0, 1.0, 1.0])
    absolute = np.abs(eigenvalues)
    l1 = np.sum(absolute)
    l2 = np.linalg.norm(eigenvalues)
    traceless = eigenvalues - np.mean(eigenvalues)
    sorted_absolute = np.sort(absolute)
    return {
        "tidal_traceless_fraction": np.linalg.norm(traceless) / l2,
        "tidal_trace_fraction": abs(np.sum(eigenvalues)) / (math.sqrt(3.0) * l2),
        "tidal_l1_dominance": np.max(absolute) / l1,
        "tidal_middle_to_max": sorted_absolute[1] / sorted_absolute[2],
        "tidal_minimum_to_max": sorted_absolute[0] / sorted_absolute[2],
        "tidal_positive_fraction": np.sum(np.maximum(eigenvalues, 0.0)) / l1,
        "tidal_signed_determinant_shape": np.prod(eigenvalues) / l2**3,
        "tidal_radial_abs_fraction": absolute[0] / l1,
        "tidal_third_axis_abs_fraction": absolute[2] / l1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sparc-method",
        choices=[
            "axisymmetric_midplane_density_closure",
            "spherical_density_closure",
        ],
        default="axisymmetric_midplane_density_closure",
    )
    parser.add_argument(
        "--output",
        default="results/reopened_tidal_shape_indicator_audit",
    )
    parser.add_argument(
        "--report-version",
        default="REOPENED-TIDAL-SHAPE-INDICATOR-AUDIT-0.1.0",
    )
    arguments = parser.parse_args()
    output = ROOT / arguments.output
    points = add_invariants(load_profiles(), arguments.sparc_method)
    systems = points.groupby(["domain", "system"], as_index=False)[
        INDICATORS + ["required_log10_enhancement"]
    ].median(numeric_only=True)
    separations = separation_table(points, systems)
    distributions = pd.concat(
        [
            distribution_table(points, "point"),
            distribution_table(systems, "equal_system_median"),
        ],
        ignore_index=True,
    )
    correlations = correlation_table(points)
    output.mkdir(parents=True, exist_ok=True)
    points.to_csv(output / "point_tidal_indicators.csv", index=False)
    systems.to_csv(output / "system_tidal_indicator_medians.csv", index=False)
    separations.to_csv(output / "separation_scores.csv", index=False)
    distributions.to_csv(output / "indicator_distributions.csv", index=False)
    correlations.to_csv(output / "boost_correlations.csv", index=False)

    ranking = separations[
        (separations.level == "equal_system_median")
        & (separations.negative_domain == "SPARC")
        & (separations.positive_domain == "CLASH")
    ].sort_values("separation_auc", ascending=False)
    method_counts = points.groupby(["domain", "tidal_geometry_method"]).size()
    report = {
        "report_version": arguments.report_version,
        "status": "completed dimensionless tidal-shape indicator audit",
        "source_geometry_audit_sha256": sha256(
            ROOT / "results/reopened_geometry_indicator_audit/report.json"
        ),
        "coverage": {
            "points": len(points),
            "systems": len(systems),
            "sparc_method": arguments.sparc_method,
            "geometry_methods": {
                f"{domain}/{method}": int(count)
                for (domain, method), count in method_counts.items()
            },
        },
        "solar_point_mass_invariants": solar_invariants(),
        "system_level_SPARC_vs_CLASH_ranking": ranking.to_dict(
            orient="records"
        ),
        "claim_boundary": [
            "The formula-facing quantities are dimensionless tidal eigenvalue invariants; object labels are used only to audit separation.",
            f"SPARC eigenvalues use {arguments.sparc_method}; BCG and CLASH use spherical_density_closure.",
            "The geometry closures are approximations and must be replaced by registered 3-D baryonic maps for a final test.",
            "Descriptive thresholds are selected on the same development sample and are not holdout claims.",
            "A high AUC is only permission to run a gravity test, not evidence that the gate improves gravity predictions.",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Reopened tidal-shape indicator audit",
        "",
        "The formula-facing candidates are dimensionless functions of baryonic "
        "tidal eigenvalues. Labels are used only after calculation.",
        "",
        "| rank | invariant | system AUC | balanced accuracy | CLASH direction |",
        "|---:|---|---:|---:|---|",
    ]
    for rank, row in enumerate(ranking.itertuples(), 1):
        lines.append(
            f"| {rank} | {row.indicator} | {row.separation_auc:.3f} | "
            f"{row.descriptive_balanced_accuracy:.3f} | {row.best_direction} |"
        )
    lines += [
        "",
        f"SPARC uses {arguments.sparc_method}; BCG and CLASH use a spherical "
        "density closure. Advancement still requires the unchanged four-domain gravity tests.",
    ]
    (output / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "top": ranking.iloc[0].to_dict(),
            },
            indent=2,
            default=json_safe,
        )
    )


if __name__ == "__main__":
    main()
