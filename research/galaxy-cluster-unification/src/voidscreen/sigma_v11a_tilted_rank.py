"""Tilted-slice nonlinear kinetic falsification for Sigma v11A.

When the aether is tilted relative to a coordinate foliation, the projected
AeST gradient ``S`` contains the coordinate velocity of ``phi``.  V11A's
bounded alignment then makes the memory-gradient energy a nonlinear function
of that velocity.  In its concave region an arbitrarily large but finite
spatial memory gradient drives the scalar velocity Hessian through zero.
"""

from __future__ import annotations

import numpy as np


def tilted_alignment_kinetic_lagrangian(
    scalar_velocity: float,
    memory_spatial_gradient: float,
    *,
    aether_tilt: float,
    acceleration_scale: float,
    memory_speed_squared: float,
    anisotropy_fraction: float,
    base_scalar_velocity_hessian: float,
) -> float:
    """Return the velocity-dependent local decoupling Lagrangian."""

    velocity = float(scalar_velocity)
    gradient = float(memory_spatial_gradient)
    tilt = float(aether_tilt)
    scale = float(acceleration_scale)
    speed = float(memory_speed_squared)
    fraction = float(anisotropy_fraction)
    base = float(base_scalar_velocity_hessian)
    values = np.asarray([velocity, gradient, tilt, scale, speed, fraction, base])
    if (
        np.any(~np.isfinite(values))
        or abs(tilt) >= 1.0
        or scale <= 0.0
        or speed <= 0.0
        or fraction <= 0.0
        or base <= 0.0
    ):
        raise ValueError("tilted-rank inputs are outside their finite domains")
    gamma_squared = 1.0 / (1.0 - tilt**2)
    projected_ratio_squared = (
        gamma_squared * tilt**2 * velocity**2 / scale**2
    )
    alignment = projected_ratio_squared / (1.0 + projected_ratio_squared)
    projected_memory_gradient_squared = gamma_squared * gradient**2
    return 0.5 * base * velocity**2 + 0.5 * (
        speed * fraction * projected_memory_gradient_squared * alignment
    )


def tilted_scalar_velocity_hessian(
    scalar_velocity: float,
    memory_spatial_gradient: float,
    *,
    aether_tilt: float,
    acceleration_scale: float,
    memory_speed_squared: float,
    anisotropy_fraction: float,
    base_scalar_velocity_hessian: float,
) -> float:
    """Return the exact ``d^2 L/d(dot(phi))^2`` in the local subblock."""

    velocity = float(scalar_velocity)
    gradient = float(memory_spatial_gradient)
    tilt = float(aether_tilt)
    scale = float(acceleration_scale)
    speed = float(memory_speed_squared)
    fraction = float(anisotropy_fraction)
    base = float(base_scalar_velocity_hessian)
    tilted_alignment_kinetic_lagrangian(
        velocity,
        gradient,
        aether_tilt=tilt,
        acceleration_scale=scale,
        memory_speed_squared=speed,
        anisotropy_fraction=fraction,
        base_scalar_velocity_hessian=base,
    )
    gamma_squared = 1.0 / (1.0 - tilt**2)
    coefficient = gamma_squared * tilt**2 / scale**2
    argument = coefficient * velocity**2
    projected_memory_gradient_squared = gamma_squared * gradient**2
    alignment_second_derivative = (
        2.0 * coefficient * (1.0 - 3.0 * argument) / (1.0 + argument) ** 3
    )
    return base + 0.5 * (
        speed
        * fraction
        * projected_memory_gradient_squared
        * alignment_second_derivative
    )


def critical_memory_gradient(
    *,
    aether_tilt: float,
    acceleration_scale: float,
    memory_speed_squared: float,
    anisotropy_fraction: float,
    base_scalar_velocity_hessian: float,
) -> dict[str, float]:
    """Return a finite exact Hessian-zero point at projected ratio squared one."""

    tilt = float(aether_tilt)
    scale = float(acceleration_scale)
    speed = float(memory_speed_squared)
    fraction = float(anisotropy_fraction)
    base = float(base_scalar_velocity_hessian)
    probe = tilted_alignment_kinetic_lagrangian(
        0.0,
        0.0,
        aether_tilt=tilt,
        acceleration_scale=scale,
        memory_speed_squared=speed,
        anisotropy_fraction=fraction,
        base_scalar_velocity_hessian=base,
    )
    if probe != 0.0 or tilt == 0.0:
        raise ValueError("a nonzero aether tilt is required")
    gamma_squared = 1.0 / (1.0 - tilt**2)
    coefficient = gamma_squared * tilt**2 / scale**2
    scalar_velocity = np.reciprocal(np.sqrt(coefficient))
    gradient_squared = 4.0 * base / (
        speed * fraction * gamma_squared * coefficient
    )
    return {
        "scalar_velocity": float(scalar_velocity),
        "memory_spatial_gradient": float(np.sqrt(gradient_squared)),
        "projected_ratio_squared": 1.0,
    }


def finite_difference_scalar_hessian(
    scalar_velocity: float,
    memory_spatial_gradient: float,
    *,
    step: float,
    **coefficients: float,
) -> float:
    """Return a five-point finite-difference second derivative."""

    velocity = float(scalar_velocity)
    delta = float(step)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("step must be finite and positive")

    def lagrangian(offset: float) -> float:
        return tilted_alignment_kinetic_lagrangian(
            velocity + offset,
            memory_spatial_gradient,
            **coefficients,
        )

    return (
        -lagrangian(2.0 * delta)
        + 16.0 * lagrangian(delta)
        - 30.0 * lagrangian(0.0)
        + 16.0 * lagrangian(-delta)
        - lagrangian(-2.0 * delta)
    ) / (12.0 * delta**2)


def audit_v11a_tilted_rank(
    *,
    aether_tilt: float,
    acceleration_scale: float,
    memory_speed_squared: float,
    anisotropy_fraction: float,
    base_scalar_velocity_hessian: float,
    finite_difference_step: float,
) -> dict[str, object]:
    """Audit the finite nonlinear kinetic zero and ghost beyond it."""

    coefficients = {
        "aether_tilt": float(aether_tilt),
        "acceleration_scale": float(acceleration_scale),
        "memory_speed_squared": float(memory_speed_squared),
        "anisotropy_fraction": float(anisotropy_fraction),
        "base_scalar_velocity_hessian": float(base_scalar_velocity_hessian),
    }
    critical = critical_memory_gradient(**coefficients)
    velocity = critical["scalar_velocity"]
    gradient = critical["memory_spatial_gradient"]
    ratios = np.array([0.99, 1.0, 1.01])
    hessians = np.array(
        [
            tilted_scalar_velocity_hessian(
                velocity,
                ratio * gradient,
                **coefficients,
            )
            for ratio in ratios
        ]
    )
    numerical = finite_difference_scalar_hessian(
        velocity,
        gradient,
        step=float(finite_difference_step),
        **coefficients,
    )
    analytic = tilted_scalar_velocity_hessian(
        velocity,
        gradient,
        **coefficients,
    )
    finite_difference_scale = max(1.0, abs(numerical), abs(analytic))
    finite_difference_error = abs(numerical - analytic) / finite_difference_scale
    lagrangian_at_zero = tilted_alignment_kinetic_lagrangian(
        velocity,
        gradient,
        **coefficients,
    )
    gates = {
        "analytic_hessian_matches_finite_difference": finite_difference_error
        < 1.0e-6,
        "finite_critical_configuration": bool(
            np.all(np.isfinite([velocity, gradient, lagrangian_at_zero]))
        ),
        "below_surface_positive": hessians[0] > 0.0,
        "critical_surface_zero": abs(hessians[1]) < 1.0e-10,
        "above_surface_negative": hessians[2] < 0.0,
        "globally_positive_scalar_velocity_hessian": bool(np.min(hessians) > 0.0),
    }
    return {
        "local_background": {
            "metric": "Minkowski",
            "aether": "A^mu=gamma(1,v,0,0)",
            "phi_derivative": "partial_mu phi=(dot(phi),0,0,0)",
            "chi_derivative": "partial_mu chi=(0,D_x chi,0,0)",
        },
        "analytic_identity": {
            "projected_scalar_magnitude": "S:S=gamma^2 v^2 dot(phi)^2",
            "projected_memory_gradient": "Dchi:Dchi=gamma^2 (D_x chi)^2",
            "alignment": "z=c dot(phi)^2/[1+c dot(phi)^2], c=gamma^2 v^2/a_Sigma^2",
            "alignment_second_derivative": "z''=2c[1-3c dot(phi)^2]/[1+c dot(phi)^2]^3",
            "consequence": "At c dot(phi)^2=1, z''=-c/2; finite D_x chi can cancel and then reverse any finite positive base scalar Hessian.",
        },
        "critical_configuration": {
            **critical,
            "lagrangian": lagrangian_at_zero,
            "analytic_scalar_velocity_hessian": analytic,
            "finite_difference_scalar_velocity_hessian": numerical,
            "finite_difference_relative_error": finite_difference_error,
        },
        "crossing": [
            {
                "gradient_over_critical": float(ratio),
                "scalar_velocity_hessian": float(hessian),
            }
            for ratio, hessian in zip(ratios, hessians, strict=True)
        ],
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_nonlinear_rank_gates_pass": bool(all(gates.values())),
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }
