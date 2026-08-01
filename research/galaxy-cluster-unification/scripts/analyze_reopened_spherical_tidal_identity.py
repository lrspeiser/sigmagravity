#!/usr/bin/env python3
"""Verify that spherical dimensionless tidal shape reduces to density ratio."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "results/reopened_tidal_shape_common_spherical_audit"
    / "point_tidal_indicators.csv"
)
OUTPUT = ROOT / "results/reopened_spherical_tidal_identity"
G_SI = 6.67430e-11
KPC_M = 3.085677581491367e19


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    points = pd.read_csv(INPUT)
    radius_m = points.radius_kpc.to_numpy(float) * KPC_M
    gbar = points.gbar_m_s2.to_numpy(float)
    density_kg_m3 = points.local_density_g_cm3.to_numpy(float) * 1000.0
    mean_density_kg_m3 = 3.0 * gbar / (
        4.0 * math.pi * G_SI * radius_m
    )
    density_ratio = density_kg_m3 / mean_density_kg_m3
    radial_to_tangential = 3.0 * density_ratio - 2.0
    normalized = np.column_stack(
        [
            radial_to_tangential,
            np.ones(len(points)),
            np.ones(len(points)),
        ]
    )
    absolute = np.abs(normalized)
    l1 = np.sum(absolute, axis=1)
    l2 = np.linalg.norm(normalized, axis=1)
    sorted_absolute = np.sort(absolute, axis=1)
    mean = np.mean(normalized, axis=1, keepdims=True)
    expected = {
        "tidal_traceless_fraction": (
            np.linalg.norm(normalized - mean, axis=1) / l2
        ),
        "tidal_trace_fraction": (
            np.abs(np.sum(normalized, axis=1)) / (math.sqrt(3.0) * l2)
        ),
        "tidal_l1_dominance": sorted_absolute[:, 2] / l1,
        "tidal_middle_to_max": (
            sorted_absolute[:, 1] / sorted_absolute[:, 2]
        ),
        "tidal_minimum_to_max": (
            sorted_absolute[:, 0] / sorted_absolute[:, 2]
        ),
        "tidal_positive_fraction": (
            np.sum(np.maximum(normalized, 0.0), axis=1) / l1
        ),
        "tidal_signed_determinant_shape": (
            np.prod(normalized, axis=1) / np.power(l2, 3.0)
        ),
        "tidal_radial_abs_fraction": absolute[:, 0] / l1,
        "tidal_third_axis_abs_fraction": absolute[:, 2] / l1,
    }
    errors = []
    for indicator, predicted in expected.items():
        observed = points[indicator].to_numpy(float)
        errors.append(
            {
                "indicator": indicator,
                "maximum_absolute_error": float(
                    np.max(np.abs(observed - predicted))
                ),
                "RMS_error": float(
                    np.sqrt(np.mean(np.square(observed - predicted)))
                ),
            }
        )
    errors_frame = pd.DataFrame(errors).sort_values("indicator")
    tolerance = 2.0e-14
    verified = bool(
        errors_frame.maximum_absolute_error.max() < tolerance
    )
    report = {
        "report_version": "REOPENED-SPHERICAL-TIDAL-IDENTITY-0.1.0",
        "status": "verified" if verified else "failed",
        "input_sha256": sha256(INPUT),
        "points": len(points),
        "systems": int(points[["domain", "system"]].drop_duplicates().shape[0]),
        "identity": {
            "mean_density": "rho_mean = 3*g/(4*pi*G*r)",
            "local_to_mean_density_ratio": "delta = rho/rho_mean",
            "normalized_tidal_eigenvalues": "lambda/(g/r) = (3*delta-2, 1, 1)",
        },
        "tolerance": tolerance,
        "maximum_error_over_all_invariants": float(
            errors_frame.maximum_absolute_error.max()
        ),
        "implication": (
            "With spherical density closure, every scale-free tidal invariant "
            "tested here is a deterministic function of local-to-mean density "
            "ratio and supplies no independent directional information."
        ),
    }
    if not verified:
        raise RuntimeError("spherical tidal identity did not verify")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    errors_frame.to_csv(OUTPUT / "invariant_identity_errors.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Spherical tidal-shape identity",
        "",
        "For spherical density closure:",
        "",
        r"\[",
        r"\bar\rho=\frac{3g}{4\pi G r},\qquad",
        r"\delta=\frac{\rho}{\bar\rho},",
        r"\]",
        "",
        r"\[",
        r"\frac{(\lambda_r,\lambda_t,\lambda_t)}{g/r}",
        r"=(3\delta-2,1,1).",
        r"\]",
        "",
        f"Verified at **{len(points)} points** with maximum numerical error "
        f"**{report['maximum_error_over_all_invariants']:.3e}**.",
        "",
        "Therefore every dimensionless spherical tidal-shape gate tested in "
        "this stage is a nonlinear reparameterization of the already-tested "
        "local-to-mean density-ratio gate. Independent directional information "
        "requires nonspherical data or spatial derivatives not fixed by this "
        "closure.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
