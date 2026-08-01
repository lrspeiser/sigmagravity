#!/usr/bin/env python3
"""Measure profile-slope gates and their dependence on radial extrapolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.raw_lensing import loglog_interpolate_with_tails  # noqa: E402


OUTPUT = ROOT / "results/reopened_slope_response_range_audit"
PIVOT = -1.0
SHARPNESSES = (0.25, 1.0, 4.0, 16.0)
RAW_CUTOFFS_KPC = (3000.0, 10000.0, 100000.0, 1000000.0)


def log_linear_slope(radius, acceleration) -> float:
    x = np.log(np.asarray(radius, dtype=float))
    y = np.log(np.asarray(acceleration, dtype=float))
    centered = x - np.mean(x)
    return float(np.dot(centered, y - np.mean(y)) / np.dot(centered, centered))


def gate(slope: float, sharpness: float) -> float:
    argument = float(sharpness) * (PIVOT - float(slope))
    if argument >= 0.0:
        return float(1.0 / (1.0 + np.exp(-argument)))
    exp_argument = np.exp(argument)
    return float(exp_argument / (1.0 + exp_argument))


def describe(values) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    return {
        "systems": int(len(array)),
        "minimum": float(np.min(array)),
        "p10": float(np.percentile(array, 10.0)),
        "p25": float(np.percentile(array, 25.0)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75.0)),
        "p90": float(np.percentile(array, 90.0)),
        "maximum": float(np.max(array)),
    }


def measured_rows(frame, group_name, radius_name, acceleration_name, domain):
    rows = []
    for system, block in frame.groupby(group_name, sort=False):
        block = block.sort_values(radius_name)
        if len(block) < 2:
            continue
        slope = log_linear_slope(
            block[radius_name].to_numpy(float),
            block[acceleration_name].to_numpy(float),
        )
        row = {
            "domain": domain,
            "system": system,
            "points": int(len(block)),
            "minimum_radius_kpc": float(block[radius_name].min()),
            "maximum_radius_kpc": float(block[radius_name].max()),
            "log_linear_slope": slope,
        }
        for sharpness in SHARPNESSES:
            row[f"gate_weight_k{sharpness:g}"] = gate(slope, sharpness)
        rows.append(row)
    return rows


def main() -> None:
    sparc = pd.read_csv(
        ROOT / "results/sparc_density_transfer/primary_predictions.csv"
    )
    bridge = pd.read_csv(
        ROOT / "results/phenomenology_formula_sweep/sample.csv"
    )
    tian = pd.read_csv(
        ROOT / "data/raw/clash_tian2020/fig2.dat",
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
    raw_protocol = json.loads(
        (ROOT / "configs/unbounded_running_multicluster_raw_protocol.json")
        .read_text(encoding="utf-8")
    )
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

    rows = measured_rows(
        sparc,
        "galaxy",
        "radius_adjusted_kpc",
        "g_bar_m_s2",
        "SPARC_measured",
    )
    cluster = bridge[bridge.domain.eq("cluster")].copy()
    cluster["gbar_m_s2"] = np.power(10.0, cluster.log_gbar.to_numpy(float))
    rows.extend(
        measured_rows(
            cluster,
            "system",
            "radius_kpc",
            "gbar_m_s2",
            "CLASH_measured",
        )
    )
    rows.extend(
        measured_rows(
            tian[tian.system.isin(raw_labels)],
            "system",
            "radius_kpc",
            "gbar_m_s2",
            "raw_cluster_measured_anchors",
        )
    )

    cutoff_rows = []
    for label, full_name in raw_labels.items():
        anchors = tian[tian.system.eq(label)].sort_values("radius_kpc")
        anchor_radius = anchors.radius_kpc.to_numpy(float)
        anchor_gbar = anchors.gbar_m_s2.to_numpy(float)
        measured_slope = log_linear_slope(anchor_radius, anchor_gbar)
        for cutoff in RAW_CUTOFFS_KPC:
            radius = np.geomspace(0.1, cutoff, 1536)
            acceleration = loglog_interpolate_with_tails(
                radius,
                anchor_radius,
                anchor_gbar,
                outer_slope=-2.0,
            )
            slope = log_linear_slope(radius, acceleration)
            row = {
                "system": full_name,
                "label": label,
                "cutoff_kpc": cutoff,
                "measured_anchor_slope": measured_slope,
                "extrapolated_grid_slope": slope,
            }
            for sharpness in SHARPNESSES:
                row[f"measured_gate_weight_k{sharpness:g}"] = gate(
                    measured_slope, sharpness
                )
                row[f"extrapolated_gate_weight_k{sharpness:g}"] = gate(
                    slope, sharpness
                )
            cutoff_rows.append(row)

    systems = pd.DataFrame(rows)
    cutoffs = pd.DataFrame(cutoff_rows)
    distributions = {}
    for domain, block in systems.groupby("domain", sort=False):
        distributions[domain] = {
            "slope": describe(block.log_linear_slope),
            "gate_weight": {
                f"k_{sharpness:g}": describe(
                    block[f"gate_weight_k{sharpness:g}"]
                )
                for sharpness in SHARPNESSES
            },
        }
    isolated_tail = cutoffs[cutoffs.cutoff_kpc.eq(1000000.0)]
    report = {
        "status": "completed profile-slope radial-range audit",
        "formula": {
            "profile_slope": "least-squares slope of ln(gbar) versus ln(radius)",
            "gate": "w=logistic[k(-1-slope)]",
        },
        "distributions": distributions,
        "raw_lensing_range_dependence": {
            "cutoffs_kpc": list(RAW_CUTOFFS_KPC),
            "systems": int(cutoffs.system.nunique()),
            "median_measured_anchor_slope": float(
                isolated_tail.measured_anchor_slope.median()
            ),
            "median_1e6_kpc_grid_slope": float(
                isolated_tail.extrapolated_grid_slope.median()
            ),
            "median_measured_gate_k4": float(
                isolated_tail["measured_gate_weight_k4"].median()
            ),
            "median_1e6_kpc_grid_gate_k4": float(
                isolated_tail["extrapolated_gate_weight_k4"].median()
            ),
            "maximum_slope_span_across_cutoffs": float(
                cutoffs.groupby("system").extrapolated_grid_slope.agg(
                    lambda values: values.max() - values.min()
                ).max()
            ),
        },
        "claim_boundary": [
            "The measured slope separation is descriptive and does not prove a physical cause.",
            "The raw-lens global slope includes the analysis pipeline's required inner and outer extrapolations, so its cutoff dependence is a formula-definition issue rather than new observational information.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    systems.to_csv(OUTPUT / "measured_profile_slopes.csv", index=False)
    cutoffs.to_csv(OUTPUT / "raw_cutoff_sensitivity.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Slope-response radial-range audit",
        "",
        f"SPARC median slope: **{distributions['SPARC_measured']['slope']['median']:.3f}**.",
        f"CLASH median slope: **{distributions['CLASH_measured']['slope']['median']:.3f}**.",
        f"Raw measured-anchor median slope: **{report['raw_lensing_range_dependence']['median_measured_anchor_slope']:.3f}**.",
        f"Raw 1,000,000 kpc-grid median slope: **{report['raw_lensing_range_dependence']['median_1e6_kpc_grid_slope']:.3f}**.",
        "",
        "The whole-profile gate separates measured SPARC and CLASH profiles, but the raw-lensing value also changes when the required extrapolation range changes.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
