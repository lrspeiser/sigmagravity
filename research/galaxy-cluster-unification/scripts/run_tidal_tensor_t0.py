#!/usr/bin/env python3
"""Run analytic and numerical viability checks for the tidal response tensor."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.tidal_tensor_response import response_tensor, solar_gate  # noqa: E402

G_SI = 6.67430e-11
SOLAR_MASS_KG = 1.98847e30
AU_M = 149597870700.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260738)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "tidal_tensor_t0" / "report.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    raw = rng.normal(size=(args.draws, 3, 3))
    symmetric = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    trace = np.trace(symmetric, axis1=-2, axis2=-1) / 3.0
    tidal = symmetric - trace[:, np.newaxis, np.newaxis] * np.eye(3)
    acceleration = np.power(10.0, rng.uniform(-14.0, 0.0, args.draws))
    a0 = 1.2e-10
    audits = {}
    for mapping, kappa in [
        ("linear", 0.999999),
        ("reciprocal", 40.0),
        ("exponential", 8.0),
    ]:
        response = response_tensor(
            tidal,
            acceleration,
            kappa=kappa,
            a0_m_s2=a0,
            mapping=mapping,
        )
        eigenvalues = np.linalg.eigvalsh(response)
        audits[mapping] = {
            "stress_kappa": kappa,
            "minimum_eigenvalue": float(np.min(eigenvalues)),
            "maximum_eigenvalue": float(np.max(eigenvalues)),
            "nonpositive_eigenvalues": int(np.sum(eigenvalues <= 0.0)),
            "positive_definite_pass": bool(np.all(eigenvalues > 0.0)),
        }
    audits["linear"][
        "analytic_lower_bound_for_traceless_3D_tidal_tensor"
    ] = 1.0 - (2.0 / 3.0) * audits["linear"]["stress_kappa"]

    earth_acceleration = G_SI * SOLAR_MASS_KG / AU_M**2
    saturn_acceleration = G_SI * SOLAR_MASS_KG / (9.58 * AU_M) ** 2
    earth_gate = float(solar_gate(earth_acceleration, a0_m_s2=a0))
    saturn_gate = float(solar_gate(saturn_acceleration, a0_m_s2=a0))
    cassini_limit = 2.3e-5
    solar_changes = {
        "linear_kappa_0.999999": (2.0 / 3.0) * 0.999999 * saturn_gate,
        "reciprocal_kappa_40": 1.0
        - 1.0 / (1.0 + (2.0 / 3.0) * 40.0 * saturn_gate),
        "exponential_kappa_8": 1.0
        - math.exp(-(2.0 / 3.0) * 8.0 * saturn_gate),
    }
    report = {
        "report_version": "TIDAL-TENSOR-T0-0.1.0",
        "status": "completed positivity and Solar-screening gate",
        "candidate": {
            "field_equation": "div[K grad(Phi)]=4*pi*G*rho_b",
            "response_families": [
                "linear: K=I-kappa*S_a*Q",
                "reciprocal: K=(I+kappa*S_a*Q)^-1",
                "exponential: K=exp(-kappa*S_a*Q)"
            ],
            "kappa_domain": "linear 0<=kappa<1; reciprocal/exponential kappa>=0",
            "a0_m_s2": a0,
        },
        "random_tensor_audit": {
            "draws": args.draws,
            "seed": args.seed,
            "mappings": audits,
            "all_positive_definite_pass": bool(
                all(row["positive_definite_pass"] for row in audits.values())
            ),
        },
        "Solar_System": {
            "Earth_orbit_acceleration_m_s2": earth_acceleration,
            "Earth_gate": earth_gate,
            "Saturn_orbit_acceleration_m_s2": saturn_acceleration,
            "Saturn_gate": saturn_gate,
            "fractional_K_changes_at_Saturn": solar_changes,
            "worst_case_fractional_K_change_at_Saturn": max(
                solar_changes.values()
            ),
            "Cassini_fractional_limit": cassini_limit,
            "screening_pass": bool(
                max(solar_changes.values()) < cassini_limit
            ),
        },
        "verdict": {
            "T0_pass": bool(
                all(row["positive_definite_pass"] for row in audits.values())
                and max(solar_changes.values()) < cassini_limit
            ),
            "remaining_requirement": "Derive a covariant action and solve the anisotropic Poisson equation on residual-blind galaxy baryon maps.",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()
