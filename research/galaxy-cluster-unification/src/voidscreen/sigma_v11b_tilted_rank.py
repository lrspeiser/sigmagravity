"""Finite tilted-flow kinetic falsification for Sigma v11B."""

from __future__ import annotations

import numpy as np


def tilted_flow_lagrangian(
    velocity_perturbation: float,
    *,
    aether_tilt: float,
    shear_speed_squared: float,
    bulk_weight: float,
) -> float:
    w = float(velocity_perturbation)
    v = float(aether_tilt)
    shear = float(shear_speed_squared)
    bulk = float(bulk_weight)
    values = np.asarray([w, v, shear, bulk])
    if (
        np.any(~np.isfinite(values))
        or not 0.0 < abs(v) < 1.0
        or shear <= 0.0
        or bulk < 0.0
    ):
        raise ValueError("tilted-flow inputs are outside their finite domain")
    gamma = 1.0 / np.sqrt(1.0 - v**2)
    a = gamma * v
    strain = 2.0 * a * w + a**2 * w**2
    trace_stiffness = 2.0 / 3.0 + bulk
    return 0.5 * gamma**2 * w**2 - 0.25 * shear * trace_stiffness * strain**2


def tilted_flow_hessian(
    velocity_perturbation: float,
    *,
    aether_tilt: float,
    shear_speed_squared: float,
    bulk_weight: float,
) -> float:
    w = float(velocity_perturbation)
    v = float(aether_tilt)
    shear = float(shear_speed_squared)
    bulk = float(bulk_weight)
    tilted_flow_lagrangian(
        w, aether_tilt=v, shear_speed_squared=shear, bulk_weight=bulk
    )
    gamma = 1.0 / np.sqrt(1.0 - v**2)
    a = gamma * v
    stiffness = shear * (2.0 / 3.0 + bulk)
    return gamma**2 - stiffness * (
        2.0 * a**2 + 6.0 * a**3 * w + 3.0 * a**4 * w**2
    )


def material_coordinate_velocity(
    velocity_perturbation: float, *, aether_tilt: float
) -> float:
    w = float(velocity_perturbation)
    v = float(aether_tilt)
    if not np.isfinite(w) or not np.isfinite(v) or abs(v) >= 1.0:
        raise ValueError("velocity inputs must be finite and subluminal")
    gamma = 1.0 / np.sqrt(1.0 - v**2)
    return v - w / gamma


def positive_rank_surface(
    *, aether_tilt: float, shear_speed_squared: float, bulk_weight: float
) -> float:
    v = float(aether_tilt)
    shear = float(shear_speed_squared)
    bulk = float(bulk_weight)
    tilted_flow_lagrangian(
        0.0, aether_tilt=v, shear_speed_squared=shear, bulk_weight=bulk
    )
    gamma = 1.0 / np.sqrt(1.0 - v**2)
    a = gamma * v
    stiffness = shear * (2.0 / 3.0 + bulk)
    coefficients = np.array(
        [
            -3.0 * stiffness * a**4,
            -6.0 * stiffness * a**3,
            gamma**2 - 2.0 * stiffness * a**2,
        ]
    )
    roots = np.roots(coefficients)
    positive = roots[(np.abs(roots.imag) < 1.0e-12) & (roots.real > 0.0)].real
    if positive.size != 1:
        raise RuntimeError("expected one positive finite rank surface")
    return float(positive[0])


def audit_v11b_tilted_rank(
    *,
    aether_tilt: float,
    shear_speed_squared: float,
    bulk_weight: float,
) -> dict[str, object]:
    coefficients = {
        "aether_tilt": float(aether_tilt),
        "shear_speed_squared": float(shear_speed_squared),
        "bulk_weight": float(bulk_weight),
    }
    critical = positive_rank_surface(**coefficients)
    ratios = np.array([0.99, 1.0, 1.01])
    hessians = np.array(
        [tilted_flow_hessian(ratio * critical, **coefficients) for ratio in ratios]
    )
    material_velocities = np.array(
        [
            material_coordinate_velocity(
                ratio * critical, aether_tilt=float(aether_tilt)
            )
            for ratio in ratios
        ]
    )
    lagrangian = tilted_flow_lagrangian(critical, **coefficients)
    gates = {
        "critical_configuration_finite": bool(
            np.all(np.isfinite([critical, lagrangian, *hessians]))
        ),
        "below_surface_positive": hessians[0] > 0.0,
        "critical_surface_zero": abs(hessians[1]) < 1.0e-12,
        "above_surface_negative": hessians[2] < 0.0,
        "crossing_occurs_with_timelike_material_flow": bool(
            np.all(np.abs(material_velocities) < 1.0)
        ),
        "globally_positive_material_velocity_hessian": bool(np.min(hessians) > 0.0),
    }
    return {
        "background": {
            "metric": "Minkowski",
            "aether": "A^mu=gamma(1,v,0,0)",
            "reference_map": "X^1=gamma(x-vt), X^2=y, X^3=z",
            "perturbation": "X^1 -> X^1+w t",
        },
        "analytic_identity": {
            "strain": "E_11=2 gamma v w+(gamma v)^2 w^2",
            "lagrangian": "L=gamma^2 w^2/2-s(2/3+b)E_11^2/4",
            "hessian": "H=gamma^2-s(2/3+b)[2a^2+6a^3w+3a^4w^2], a=gamma v",
            "large_velocity": "H -> -3s(2/3+b)a^4 w^2",
        },
        "critical_configuration": {
            "velocity_perturbation": critical,
            "material_coordinate_velocity": material_coordinate_velocity(
                critical, aether_tilt=float(aether_tilt)
            ),
            "lagrangian": lagrangian,
        },
        "crossing": [
            {
                "velocity_over_critical": float(ratio),
                "hessian": float(hessian),
                "material_velocity": float(material_velocity),
            }
            for ratio, hessian, material_velocity in zip(
                ratios, hessians, material_velocities, strict=True
            )
        ],
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_nonlinear_rank_gates_pass": bool(all(gates.values())),
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
