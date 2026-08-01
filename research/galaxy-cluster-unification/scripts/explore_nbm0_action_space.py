from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.basin_action import (
    KPC_M,
    M_SUN_KG,
    fit_effective_yukawa_from_extras,
    fractional_linear_point_source_scaling,
    point_mass_circular_speed_log_slope,
    point_mass_yukawa_acceleration_m_s2,
    positive_spectral_circular_speed_log_slope,
    reciprocal_dust_couplings,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_decisions(
    protocol: dict[str, object],
    *,
    canonical_maximum_speed_slope: float,
    spectral_maximum_speed_slope: float,
    fractional: dict[str, float | bool],
) -> list[dict[str, object]]:
    decisions = {
        "A0": (
            "reject",
            "Pure conformal response cancels from the Weyl potential, so it cannot supply the missing cluster lensing.",
        ),
        "A1": (
            "reject",
            "A prescribed U supplies algebraic slip but no reciprocal vector equation; general covariance requires the preferred direction to be dynamical.",
        ),
        "A2": (
            "reject_as_unifier_retain_as_null",
            f"The largest point-source circular-speed slope is {canonical_maximum_speed_slope:.6f}; an attractive canonical Yukawa term is never flatter than Keplerian (-0.5).",
        ),
        "A3": (
            "reject",
            "The massless limit only rescales Newtonian gravity, retains v proportional to r^-1/2 outside finite baryons, and has no declared Solar-System screening.",
        ),
        "A4": (
            "reject",
            f"With nonnegative spectral weights the largest tested speed slope is {spectral_maximum_speed_slope:.6f}; analytically every Yukawa contribution decreases with radius.",
        ),
        "A5": (
            "reject",
            "p=3/2 gives a logarithmic potential and flat speed, but linear sourcing fixes v_flat^4 proportional to M_b^2 instead of M_b.",
        ),
        "A6": (
            "excluded_by_research_boundary",
            "Its scale-invariant deep-field equation is the AQUAL/MOND galaxy mechanism already retired by user direction; this is not counted as a scientific falsification.",
        ),
        "A7": (
            "reject_for_broad_rotation_support",
            "A smooth external basin is locally constant plus tidal: the uniform gradient drops out of internal motion and the leading isotropic acceleration is proportional to r, not 1/r.",
        ),
        "A8": (
            "advance_to_action_derivation",
            "Nonlinearity plus causal nonlocality is the first remaining route that can evade the positive-spectral radial theorem without inserting a per-galaxy scale.",
        ),
        "A9": (
            "advance_to_action_derivation",
            "A self-gravitating basin phase could make the same extended field distribution affect dynamics and lensing, but must prove positive energy, void repulsion, and screening.",
        ),
    }
    rows = []
    for candidate in protocol["candidate_families"]:
        status, reason = decisions[candidate["id"]]
        rows.append({**candidate, "decision": status, "reason": reason})
    assert fractional["flat_rotation_curve"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "nbm0_reciprocal_action_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "nbm0_action_space",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))

    x = np.geomspace(1.0e-5, 1.0e3, 20_000)
    canonical_slope = point_mass_circular_speed_log_slope(x, 1.0e6)
    radius = np.geomspace(1.0e-3, 1.0e4, 20_000) * KPC_M
    spectral_slope = positive_spectral_circular_speed_log_slope(
        radius,
        np.asarray([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]) * KPC_M,
        np.asarray([0.1, 0.3, 1.0, 3.0, 10.0, 30.0]),
    )
    fractional = fractional_linear_point_source_scaling(1.5)

    synthetic = protocol["synthetic_recovery"]
    synthetic_radius = np.geomspace(
        synthetic["radius_over_range_min"],
        synthetic["radius_over_range_max"],
        synthetic["points"],
    ) * synthetic["range_kpc"] * KPC_M
    synthetic_mass = synthetic["mass_solar"] * M_SUN_KG
    synthetic_unit = point_mass_yukawa_acceleration_m_s2(
        synthetic_radius,
        synthetic_mass,
        synthetic["range_kpc"] * KPC_M,
    )
    recovery = fit_effective_yukawa_from_extras(
        synthetic_radius,
        synthetic_mass,
        synthetic["dynamics_amplitude"] * synthetic_unit,
        synthetic["lensing_to_dynamics_ratio"]
        * synthetic["dynamics_amplitude"]
        * synthetic_unit,
        initial_dynamics_amplitude=0.2,
        initial_range_m=3.0 * KPC_M,
        initial_lensing_ratio=0.4,
    )
    synthetic_pass = bool(
        recovery.success
        and recovery.maximum_absolute_log_residual
        <= synthetic["maximum_log_residual"]
        and recovery.jacobian_condition_number
        <= synthetic["maximum_jacobian_condition_number"]
    )

    rows = candidate_decisions(
        protocol,
        canonical_maximum_speed_slope=float(np.max(canonical_slope)),
        spectral_maximum_speed_slope=float(np.max(spectral_slope)),
        fractional=fractional,
    )
    frame = pd.DataFrame(rows)
    counts = frame["decision"].value_counts().sort_index().to_dict()

    special = {}
    for name, alpha, beta in [
        ("pure_conformal", 1.0, 0.0),
        ("pure_disformal", 0.0, 1.0),
        ("no_slip", -0.5, -1.0),
        ("dust_source_blind", 1.0, 1.0),
    ]:
        value = reciprocal_dust_couplings(alpha, beta)
        special[name] = {
            "alpha": value.alpha,
            "beta": value.beta,
            "source_d": value.source_d,
            "dynamics_amplitude": value.dynamics_amplitude,
            "lensing_amplitude": value.lensing_amplitude,
            "lensing_to_dynamics_ratio": (
                None
                if not np.isfinite(value.lensing_to_dynamics_ratio)
                else value.lensing_to_dynamics_ratio
            ),
        }

    report = {
        "report_version": "NBM0-A1-action-space-0.1",
        "status": "completed reciprocal canonical action and declared ten-family structural scan",
        "protocol": str(args.protocol.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(args.protocol),
        "astronomical_fit_performed": False,
        "canonical_action_result": {
            "independent_source_strength_removed": "kappa_X",
            "identifiable_effective_parameters": ["A_dyn", "L_X", "q"],
            "reciprocal_dust_source_pass": True,
            "universal_metric_lensing_mapping_pass": True,
            "cosmological_background_subtraction_requires_background_solution": True,
            "full_vector_mode_health_pass": False,
            "full_vector_mode_health_disposition": "not derived; Maxwell-aether control is not promoted",
            "Solar_System_screening_pass": False,
            "Solar_System_screening_disposition": "not specified",
            "galaxy_outer_radial_shape_pass": False,
            "overall_decision": "reject canonical linear NBM0 as a galaxy-cluster unifier; retain it as a reciprocal null model"
        },
        "special_metric_limits": special,
        "radial_shape_theorems": {
            "canonical_Yukawa_speed_slope_minimum": float(np.min(canonical_slope)),
            "canonical_Yukawa_speed_slope_maximum": float(np.max(canonical_slope)),
            "canonical_required_flat_interval": [-0.1, 0.1],
            "canonical_pass": bool(np.any(np.abs(canonical_slope) <= 0.1)),
            "positive_spectral_speed_slope_minimum": float(np.min(spectral_slope)),
            "positive_spectral_speed_slope_maximum": float(np.max(spectral_slope)),
            "positive_spectral_pass": bool(np.any(np.abs(spectral_slope) <= 0.1)),
            "analytic_reason": "For A_i>=0, E(r)=1+sum A_i(1+r/L_i)exp(-r/L_i) has dE/dr<=0, so d ln(v)/d ln(r)=-1/2+(1/2)d ln(E)/d ln(r)<=-1/2."
        },
        "fractional_linear_control": fractional,
        "synthetic_identifiability": {
            "pass": synthetic_pass,
            "injected": {
                "dynamics_amplitude": synthetic["dynamics_amplitude"],
                "range_kpc": synthetic["range_kpc"],
                "lensing_to_dynamics_ratio": synthetic[
                    "lensing_to_dynamics_ratio"
                ],
            },
            "recovered": {
                "dynamics_amplitude": recovery.dynamics_amplitude,
                "range_kpc": recovery.range_m / KPC_M,
                "lensing_to_dynamics_ratio": recovery.lensing_to_dynamics_ratio,
                "maximum_absolute_log_residual": recovery.maximum_absolute_log_residual,
                "jacobian_condition_number": recovery.jacobian_condition_number,
            },
            "interpretation": "Ideal overlapping dynamics and lensing across the transition identify A_dyn, L_X, and q; the real same-system data gate remains separate and currently fails."
        },
        "candidate_counts": counts,
        "candidate_decisions": rows,
        "surviving_derivation_targets": ["A8", "A9"],
        "next_stage": [
            "A8: localize one causal nonlinear form factor with auxiliary fields and derive its complete quadratic spectrum before choosing constants.",
            "A9: derive the minimal positive-energy basin-phase stress tensor and test whether an underdensity can repel while its accumulated boundary energy lenses attractively.",
            "Reject either branch immediately if it recreates a local MOND/AQUAL equation, needs negative kinetic energy, introduces a class switch, or cannot screen in the Solar System."
        ]
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "candidate_catalog.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
