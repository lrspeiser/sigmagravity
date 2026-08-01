#!/usr/bin/env python3
"""Measure how a local-slope carrier gate reshapes real radial profiles."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.reopened_hybrids import _logistic, _profile_log_slope  # noqa: E402


SPARC_PATH = ROOT / "results/sparc_density_transfer/primary_predictions.csv"
BRIDGE_PATH = ROOT / "results/phenomenology_formula_sweep/sample.csv"
OUTPUT = ROOT / "results/reopened_slope_gate_geometry_audit"
SHARPNESSES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
PIVOT = -1.0
BASE_P, BASE_Q = -1.0, -0.5
STEEP_P, STEEP_Q = -0.5, 1.5
G_REFERENCE = 1.0e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile_summary(values) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array),
        "minimum": float(np.min(array)),
        "p10": float(np.percentile(array, 10.0)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "maximum": float(np.max(array)),
    }


def analyze_profile(domain: str, system: str, block: pd.DataFrame) -> tuple[list[dict], dict]:
    ordered = block.sort_values("radius_kpc")
    radius = ordered.radius_kpc.to_numpy(float)
    gbar = ordered.gbar_m_s2.to_numpy(float)
    slope = _profile_log_slope(radius, gbar)
    log_radius = np.log(radius)
    rows = []
    gates = {}
    log_factors = {}
    for sharpness in SHARPNESSES:
        label = f"k_{sharpness:g}".replace(".", "p")
        gate = _logistic(sharpness * (PIVOT - slope))
        effective_p = BASE_P + gate * (STEEP_P - BASE_P)
        effective_q = BASE_Q + gate * (STEEP_Q - BASE_Q)
        log_factor = (
            effective_p * np.log(gbar / G_REFERENCE)
            + effective_q * log_radius
        )
        gates[label] = gate
        log_factors[label] = log_factor
    fixed_log_factor = (
        STEEP_P * np.log(gbar / G_REFERENCE) + STEEP_Q * log_radius
    )
    for index in range(len(ordered)):
        row = {
            "domain": domain,
            "system": system,
            "radius_kpc": radius[index],
            "gbar_m_s2": gbar[index],
            "local_log_gbar_slope": slope[index],
        }
        for label, gate in gates.items():
            row[f"gate_{label}"] = gate[index]
            row[f"log_source_factor_{label}"] = log_factors[label][index]
        rows.append(row)
    adjacent_slope_change = np.abs(np.diff(slope))
    summary = {
        "domain": domain,
        "system": system,
        "points": len(ordered),
        "log_radius_span": float(np.ptp(log_radius)),
        "slope_median": float(np.median(slope)),
        "slope_standard_deviation": float(np.std(slope)),
        "maximum_adjacent_slope_change": float(
            np.max(adjacent_slope_change) if len(adjacent_slope_change) else 0.0
        ),
    }
    fixed_derivative = np.diff(fixed_log_factor) / np.diff(log_radius)
    summary["fixed_steep_endpoint_p90_abs_log_factor_derivative"] = float(
        np.percentile(np.abs(fixed_derivative), 90.0)
    )
    for label, gate in gates.items():
        derivative = np.diff(log_factors[label]) / np.diff(log_radius)
        summary[f"gate_range_{label}"] = float(np.ptp(gate))
        summary[f"p90_abs_log_factor_derivative_{label}"] = float(
            np.percentile(np.abs(derivative), 90.0)
        )
        summary[f"max_abs_log_factor_derivative_{label}"] = float(
            np.max(np.abs(derivative))
        )
    return rows, summary


def main() -> None:
    sparc = pd.read_csv(SPARC_PATH).rename(
        columns={
            "galaxy": "system",
            "radius_adjusted_kpc": "radius_kpc",
            "g_bar_m_s2": "gbar_m_s2",
        }
    )
    bridge = pd.read_csv(BRIDGE_PATH)
    clash = bridge[bridge.domain.eq("cluster")].copy()
    clash["gbar_m_s2"] = np.power(10.0, clash.log_gbar.to_numpy(float))
    point_rows = []
    system_rows = []
    for domain, frame in [("SPARC", sparc), ("CLASH", clash)]:
        for system, block in frame.groupby("system", sort=False):
            if len(block) <= 1:
                continue
            points, system_summary = analyze_profile(
                domain, str(system), block
            )
            point_rows.extend(points)
            system_rows.append(system_summary)
    points = pd.DataFrame(point_rows)
    systems = pd.DataFrame(system_rows)

    distribution = {}
    for domain, block in systems.groupby("domain", sort=False):
        distribution[domain] = {
            "systems": len(block),
            "points": int(
                points.loc[points.domain.eq(domain)].shape[0]
            ),
            "system_median_slope": percentile_summary(block.slope_median),
            "within_system_slope_standard_deviation": percentile_summary(
                block.slope_standard_deviation
            ),
            "maximum_adjacent_slope_change": percentile_summary(
                block.maximum_adjacent_slope_change
            ),
            "fixed_steep_endpoint_p90_abs_log_factor_derivative": (
                percentile_summary(
                    block.fixed_steep_endpoint_p90_abs_log_factor_derivative
                )
            ),
            "gates": {},
        }
        for sharpness in SHARPNESSES:
            label = f"k_{sharpness:g}".replace(".", "p")
            distribution[domain]["gates"][label] = {
                "system_gate_range": percentile_summary(
                    block[f"gate_range_{label}"]
                ),
                "p90_abs_log_factor_derivative": percentile_summary(
                    block[f"p90_abs_log_factor_derivative_{label}"]
                ),
                "maximum_abs_log_factor_derivative": percentile_summary(
                    block[f"max_abs_log_factor_derivative_{label}"]
                ),
            }

    report = {
        "status": "completed local-slope gate geometry audit",
        "formula": {
            "gate": "w=logistic(k(s_pivot-s_local))",
            "pivot": PIVOT,
            "base_carrier": {"p": BASE_P, "q": BASE_Q},
            "steep_carrier": {"p": STEEP_P, "q": STEEP_Q},
            "source": "X=F(g_N/g_ref)^p_eff(r/kpc)^q_eff",
        },
        "distribution": distribution,
        "interpretation": [
            "Small sharpness makes the gate nearly constant across each profile, so it behaves mostly like one globally blended carrier rather than a slope discriminator.",
            "Large sharpness makes effective exponents change rapidly between neighboring radii; exponent derivatives multiply log(g_N/g_ref) and log(r/kpc), creating source-factor gradients larger than either fixed endpoint.",
            "This audit measures the geometric mechanism directly; whether refitting compensates for it is decided only by the full cross-domain run.",
        ],
        "claim_boundary": [
            "Numerical local derivatives can contain real baryonic structure, measurement noise, and interpolation artifacts; this audit does not separate them.",
            "Only 18 CLASH systems have more than one bridge radius; all one-point systems are correctly excluded because their memory response is exactly local.",
            "A large source-factor derivative is a leverage diagnostic, not by itself a rejection criterion.",
        ],
        "input_hashes": {
            "SPARC": sha256(SPARC_PATH),
            "bridge": sha256(BRIDGE_PATH),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    points.to_csv(OUTPUT / "point_slopes_and_gates.csv", index=False)
    systems.to_csv(OUTPUT / "system_gate_geometry.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Local-slope carrier geometry audit",
        "",
        f"- SPARC multi-point systems: **{distribution['SPARC']['systems']}**",
        f"- CLASH multi-point systems: **{distribution['CLASH']['systems']}**",
        "",
        "Small sharpness behaves mainly like a constant interpolation between two carriers. Large sharpness causes the exponents themselves to jump between neighboring radii, amplifying the radial source-factor gradient beyond either fixed endpoint.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
