#!/usr/bin/env python3
"""Audit smoothed local slopes in measured and raw-lensing profile ranges."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.raw_lensing import loglog_interpolate_with_tails  # noqa: E402
from voidscreen.reopened_hybrids import _profile_smoothed_log_slope  # noqa: E402


PROTOCOL = ROOT / "configs/reopened_smoothed_slope_geometry_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logistic(value) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    return np.exp(-np.logaddexp(0.0, -values))


def describe(values) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    return {
        "systems": int(len(array)),
        "minimum": float(np.min(array)),
        "p10": float(np.percentile(array, 10.0)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "maximum": float(np.max(array)),
    }


def separation_auc(galaxy_values, cluster_values) -> float:
    galaxies = np.asarray(galaxy_values, dtype=float)
    clusters = np.asarray(cluster_values, dtype=float)
    differences = clusters[:, None] - galaxies[None, :]
    return float(np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0))


def profile_rows(
    frame: pd.DataFrame,
    *,
    group: str,
    radius: str,
    acceleration: str,
    domain: str,
    scales: list[float],
    pivot: float,
    sharpness: float,
) -> list[dict[str, object]]:
    rows = []
    for system, block in frame.groupby(group, sort=False):
        block = block.sort_values(radius)
        if len(block) < 2:
            continue
        radii = block[radius].to_numpy(float)
        values = block[acceleration].to_numpy(float)
        for scale in scales:
            slopes = _profile_smoothed_log_slope(
                radii, values, log_scale=scale
            )
            weights = logistic(sharpness * (pivot - slopes))
            rows.append(
                {
                    "domain": domain,
                    "system": system,
                    "points": int(len(block)),
                    "minimum_radius_kpc": float(np.min(radii)),
                    "maximum_radius_kpc": float(np.max(radii)),
                    "smoothing_log_scale": scale,
                    "median_smoothed_slope": float(np.median(slopes)),
                    "slope_span": float(np.ptp(slopes)),
                    "median_gate_weight": float(np.median(weights)),
                    "gate_weight_span": float(np.ptp(weights)),
                }
            )
    return rows


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_geometry_scores":
        raise RuntimeError("smoothed-slope geometry protocol is not frozen")
    inputs = {name: ROOT / path for name, path in protocol["inputs"].items()}
    scales = [float(value) for value in protocol["smoothing_log_scales"]]
    pivot = float(protocol["gate_pivot"])
    sharpness = float(protocol["gate_sharpness"])
    sparc = pd.read_csv(inputs["sparc_outer_sample"])
    tian = pd.read_csv(
        inputs["tian_baryonic_profiles"],
        sep=r"\s+",
        names=[
            "system",
            "radius_kpc",
            "log_gbar",
            "log_gobs",
            "err_log_gbar",
            "err_log_gobs",
        ],
    )
    tian["gbar_m_s2"] = np.power(10.0, tian.log_gbar.to_numpy(float))
    raw_protocol = json.loads(inputs["raw_lensing_protocol"].read_text(encoding="utf-8"))
    raw_labels = {
        row["label"]: row["system"]
        for row in raw_protocol["systems"]
        if row["system"]
        in {
            "MACS J0329.7-0211",
            "MACS J0429.6-0253",
            "MACS J1115.9+0129",
            "MACS J1931.8-2635",
        }
    }

    rows = profile_rows(
        sparc,
        group="galaxy",
        radius="radius_adjusted_kpc",
        acceleration="g_bar_m_s2",
        domain="SPARC_outer",
        scales=scales,
        pivot=pivot,
        sharpness=sharpness,
    )
    rows.extend(
        profile_rows(
            tian,
            group="system",
            radius="radius_kpc",
            acceleration="gbar_m_s2",
            domain="CLASH_measured",
            scales=scales,
            pivot=pivot,
            sharpness=sharpness,
        )
    )
    profiles = pd.DataFrame(rows)

    cutoff_rows = []
    grid_points = int(protocol["raw_grid_points"])
    for label, full_name in raw_labels.items():
        anchors = tian[tian.system.eq(label)].sort_values("radius_kpc")
        anchor_radius = anchors.radius_kpc.to_numpy(float)
        anchor_gbar = anchors.gbar_m_s2.to_numpy(float)
        for cutoff in protocol["raw_cutoffs_kpc"]:
            radius = np.geomspace(0.1, float(cutoff), grid_points)
            gbar = loglog_interpolate_with_tails(
                radius, anchor_radius, anchor_gbar, outer_slope=-2.0
            )
            for scale in scales:
                slopes = _profile_smoothed_log_slope(
                    radius, gbar, log_scale=scale
                )
                for probe in protocol["raw_probe_radii_kpc"]:
                    slope = float(
                        np.interp(np.log(float(probe)), np.log(radius), slopes)
                    )
                    cutoff_rows.append(
                        {
                            "system": full_name,
                            "label": label,
                            "cutoff_kpc": float(cutoff),
                            "probe_radius_kpc": float(probe),
                            "smoothing_log_scale": scale,
                            "smoothed_slope": slope,
                            "gate_weight": float(
                                logistic(sharpness * (pivot - slope))
                            ),
                        }
                    )
    cutoffs = pd.DataFrame(cutoff_rows)

    by_scale = {}
    for scale in scales:
        block = profiles[profiles.smoothing_log_scale.eq(scale)]
        galaxies = block[block.domain.eq("SPARC_outer")]
        clusters = block[block.domain.eq("CLASH_measured")]
        range_spans = (
            cutoffs[cutoffs.smoothing_log_scale.eq(scale)]
            .groupby(["system", "probe_radius_kpc"])
            .smoothed_slope.agg(lambda values: float(np.ptp(values)))
        )
        by_scale[f"{scale:g}"] = {
            "SPARC_system_median_slope": describe(galaxies.median_smoothed_slope),
            "CLASH_system_median_slope": describe(clusters.median_smoothed_slope),
            "SPARC_vs_CLASH_slope_auc": separation_auc(
                galaxies.median_smoothed_slope,
                clusters.median_smoothed_slope,
            ),
            "median_within_SPARC_slope_span": float(galaxies.slope_span.median()),
            "median_within_CLASH_slope_span": float(clusters.slope_span.median()),
            "raw_cutoff_median_fixed_radius_slope_span": float(range_spans.median()),
            "raw_cutoff_maximum_fixed_radius_slope_span": float(range_spans.max()),
        }

    report = {
        "status": "completed smoothed-local-slope geometry audit",
        "protocol_sha256": sha256(PROTOCOL),
        "input_hashes": {name: sha256(path) for name, path in inputs.items()},
        "coverage": {
            "SPARC_systems": int(
                profiles[profiles.domain.eq("SPARC_outer")].system.nunique()
            ),
            "CLASH_systems": int(
                profiles[profiles.domain.eq("CLASH_measured")].system.nunique()
            ),
            "raw_clusters": int(cutoffs.system.nunique()),
            "smoothing_scales": len(scales),
            "raw_cutoffs": int(cutoffs.cutoff_kpc.nunique()),
            "raw_probe_radii": int(cutoffs.probe_radius_kpc.nunique()),
        },
        "formula": {
            "slope": "weighted local-linear slope of ln(gbar) versus ln(radius)",
            "kernel": "exp[-0.5 (Delta ln(radius)/ell)^2]",
            "gate": f"logistic[{sharpness:g} ({pivot:g}-slope)]",
        },
        "by_smoothing_log_scale": by_scale,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / Path(protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output / "profile_summaries.csv", index=False)
    cutoffs.to_csv(output / "raw_cutoff_sensitivity.csv", index=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Smoothed-local-slope geometry audit",
        "",
        f"Coverage: **{report['coverage']['SPARC_systems']}** SPARC systems, **{report['coverage']['CLASH_systems']}** measured CLASH profiles, and **{report['coverage']['raw_clusters']}** raw-lens clusters.",
        "",
        "| smoothing scale | galaxy median slope | cluster median slope | separation AUC | median raw cutoff span |",
        "|---:|---:|---:|---:|---:|",
    ]
    for scale in scales:
        row = by_scale[f"{scale:g}"]
        lines.append(
            f"| {scale:g} | {row['SPARC_system_median_slope']['median']:.3f} | {row['CLASH_system_median_slope']['median']:.3f} | {row['SPARC_vs_CLASH_slope_auc']:.3f} | {row['raw_cutoff_median_fixed_radius_slope_span']:.4f} |"
        )
    (output / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
