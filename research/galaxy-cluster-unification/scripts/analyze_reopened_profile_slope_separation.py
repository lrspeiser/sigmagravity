#!/usr/bin/env python3
"""Measure whether baryonic radial slopes separate galaxy and cluster memory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPARC_PATH = ROOT / "results/sparc_density_transfer/primary_predictions.csv"
BRIDGE_PATH = ROOT / "results/phenomenology_formula_sweep/sample.csv"
OUTPUT = ROOT / "results/reopened_profile_slope_audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slopes(
    frame: pd.DataFrame,
    *,
    domain: str,
    group_name: str,
    radius_name: str,
    gbar_name: str,
) -> list[dict]:
    rows = []
    for system, block in frame.groupby(group_name, sort=False):
        radius = block[radius_name].to_numpy(float)
        gbar = block[gbar_name].to_numpy(float)
        if len(block) < 2 or np.ptp(np.log(radius)) <= 0.0:
            continue
        slope = float(np.polyfit(np.log(radius), np.log(gbar), 1)[0])
        rows.append(
            {
                "domain": domain,
                "system": system,
                "points": len(block),
                "log_radius_span": float(np.ptp(np.log(radius))),
                "d_ln_gbar_d_ln_radius": slope,
            }
        )
    return rows


def pairwise_auc(lower: np.ndarray, higher: np.ndarray) -> float:
    """Probability that a random higher-class value exceeds a lower-class one."""

    differences = higher[:, None] - lower[None, :]
    return float(
        np.mean(differences > 0.0) + 0.5 * np.mean(differences == 0.0)
    )


def distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "systems": len(values),
        "minimum": float(np.min(values)),
        "p10": float(np.percentile(values, 10.0)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    sparc = pd.read_csv(SPARC_PATH)
    bridge = pd.read_csv(BRIDGE_PATH)
    rows = slopes(
        sparc,
        domain="SPARC",
        group_name="galaxy",
        radius_name="radius_adjusted_kpc",
        gbar_name="g_bar_m_s2",
    )
    cluster = bridge[bridge.domain == "cluster"].copy()
    cluster["gbar_m_s2"] = np.power(10.0, cluster.log_gbar.to_numpy(float))
    rows += slopes(
        cluster,
        domain="CLASH",
        group_name="system",
        radius_name="radius_kpc",
        gbar_name="gbar_m_s2",
    )
    table = pd.DataFrame(rows)
    sparc_slopes = table.loc[
        table.domain == "SPARC", "d_ln_gbar_d_ln_radius"
    ].to_numpy(float)
    clash_slopes = table.loc[
        table.domain == "CLASH", "d_ln_gbar_d_ln_radius"
    ].to_numpy(float)
    auc_clash_higher = pairwise_auc(sparc_slopes, clash_slopes)
    candidate_outer_bias = {}
    for name, (gbar_power, radius_power) in {
        "fractional": (0.0, 0.0),
        "low_g_weighted": (-1.0, 0.0),
        "outer_radius_weighted": (0.0, 1.5),
        "combined_low_g_outer": (-1.0, 0.5),
        "added_acceleration": (1.0, 0.0),
        "speed_squared": (1.0, 1.0),
    }.items():
        sparc_bias = radius_power + gbar_power * sparc_slopes
        clash_bias = radius_power + gbar_power * clash_slopes
        candidate_outer_bias[name] = {
            "gbar_power": gbar_power,
            "radius_power": radius_power,
            "SPARC_median_effective_radial_power": float(
                np.median(sparc_bias)
            ),
            "CLASH_median_effective_radial_power": float(
                np.median(clash_bias)
            ),
            "SPARC_minus_CLASH_median": float(
                np.median(sparc_bias) - np.median(clash_bias)
            ),
        }
    report = {
        "report_version": "REOPENED-PROFILE-SLOPE-AUDIT-0.1.0",
        "status": "completed empirical radial-slope audit",
        "SPARC": distribution(sparc_slopes),
        "CLASH": distribution(clash_slopes),
        "AUC_CLASH_slope_higher_than_SPARC": auc_clash_higher,
        "candidate_transport_outer_bias": candidate_outer_bias,
        "interpretation": {
            "source_radial_power": (
                "For X=F(gbar/gref)^p(r/kpc)^q and local "
                "gbar proportional to r^s, the transported source carries "
                "an effective radial factor r^(q+p*s)."
            ),
            "finding": (
                "SPARC baryonic acceleration usually falls much faster with "
                "radius than CLASH baryonic acceleration. Negative p can "
                "therefore weight galaxy outskirts more strongly without an "
                "object label; positive q weights both domains outward."
            ),
        },
        "claim_boundary": [
            "Slopes are log-linear summaries over each available profile, not local derivatives.",
            "Only 18 CLASH systems have at least two bridge radii; the 44 one-point BCG systems cannot enter this audit.",
            "A high separation AUC identifies leverage, not a correct gravity law.",
        ],
        "input_hashes": {
            "SPARC": sha256(SPARC_PATH),
            "bridge": sha256(BRIDGE_PATH),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "system_slopes.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Baryonic radial-slope audit",
        "",
        f"- SPARC median d ln(gbar)/d ln(r): **{np.median(sparc_slopes):.3f}**",
        f"- CLASH median d ln(gbar)/d ln(r): **{np.median(clash_slopes):.3f}**",
        f"- Equal-system slope-separation AUC: **{auc_clash_higher:.3f}**",
        "",
        "Negative gbar power weights the memory source outward much more strongly "
        "in SPARC than in CLASH because galaxy baryonic acceleration falls more "
        "steeply with radius. Positive radius power weights both domains outward.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
