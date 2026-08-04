"""Hamiltonian-first convex carrier selected for Sigma v13B.

The local preferred-frame Hamiltonian density is

``H = a_sigma^2 F_epsilon(z)/2``

with ``z=(Pi^2+|grad sigma|^2)/a_sigma^2`` and

``F_epsilon(z)=epsilon z+(1-epsilon)[z-2 sqrt(z)+2 log(1+sqrt(z))]``.

Its static flux is the simple AQUAL interpolation

``mu(t)=epsilon+(1-epsilon)t/(1+t)``, ``t=sqrt(z)``.

Unlike a naive Lorentz-scalar AQUAL completion, the Hamiltonian is strictly
convex in the canonical momentum and every spatial-gradient component.  Its
full Hessian has eigenvalues between ``epsilon`` and one.  The corresponding
first-order scalar characteristics are therefore real and lie inside the
preferred-frame unit cone on arbitrary momentum/gradient backgrounds.

This module selects a reduced carrier only.  It does not supply the covariant
field that defines the preferred foliation or a physical metric equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class ConvexCarrierParameters:
    """Universal reduced-carrier parameters in units with ``c=1``."""

    acceleration_scale: float = 1.0
    epsilon: float = 1.0e-6

    def validated(self) -> ConvexCarrierParameters:
        values = np.asarray([self.acceleration_scale, self.epsilon], dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("convex-carrier parameters must be finite")
        if self.acceleration_scale <= 0.0:
            raise ValueError("acceleration_scale must be positive")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie strictly between zero and one")
        return self


DEFAULT_CONVEX_CARRIER_PARAMETERS = ConvexCarrierParameters()


def _base_shape_from_radius(radius_ratio) -> np.ndarray:
    """Evaluate ``t^2-2t+2log(1+t)`` without small-``t`` cancellation."""

    value = np.asarray(radius_ratio, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("radius_ratio must be finite and nonnegative")
    output = np.empty_like(value)
    small = value < 1.0e-3
    if np.any(small):
        x = value[small]
        series = np.zeros_like(x)
        for power in range(3, 15):
            series += 2.0 * (-1.0) ** (power + 1) * x**power / power
        output[small] = series
    if np.any(~small):
        x = value[~small]
        output[~small] = x**2 - 2.0 * x + 2.0 * np.log1p(x)
    return output


def carrier_shape(
    radius_ratio,
    *,
    epsilon: float,
) -> np.ndarray:
    """Return the dimensionless convex Hamiltonian shape ``F_epsilon``."""

    floor = float(epsilon)
    if not np.isfinite(floor) or not 0.0 < floor < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    radius = np.asarray(radius_ratio, dtype=float)
    return floor * radius**2 + (1.0 - floor) * _base_shape_from_radius(radius)


def carrier_response_mu(radius_ratio, *, epsilon: float) -> np.ndarray:
    """Return the static AQUAL flux coefficient and transverse Hessian value."""

    floor = float(epsilon)
    radius = np.asarray(radius_ratio, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("radius_ratio must be finite and nonnegative")
    if not np.isfinite(floor) or not 0.0 < floor < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    return floor + (1.0 - floor) * radius / (1.0 + radius)


def carrier_radial_curvature(radius_ratio, *, epsilon: float) -> np.ndarray:
    """Return the radial eigenvalue of the complete phase-space Hessian."""

    floor = float(epsilon)
    radius = np.asarray(radius_ratio, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("radius_ratio must be finite and nonnegative")
    if not np.isfinite(floor) or not 0.0 < floor < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    return floor + (1.0 - floor) * (1.0 - 1.0 / (1.0 + radius) ** 2)


def carrier_hamiltonian_density(
    momentum: float,
    spatial_gradient,
    *,
    parameters: ConvexCarrierParameters = DEFAULT_CONVEX_CARRIER_PARAMETERS,
) -> float:
    """Return the positive local Hamiltonian density."""

    params = parameters.validated()
    pi = float(momentum)
    gradient = np.asarray(spatial_gradient, dtype=float)
    if gradient.ndim != 1 or gradient.size == 0:
        raise ValueError("spatial_gradient must be a nonempty vector")
    if not np.isfinite(pi) or np.any(~np.isfinite(gradient)):
        raise ValueError("carrier phase-space values must be finite")
    radius = np.sqrt(pi**2 + float(gradient @ gradient))
    ratio = radius / params.acceleration_scale
    return float(
        0.5
        * params.acceleration_scale**2
        * carrier_shape(ratio, epsilon=params.epsilon)
    )


def carrier_phase_space_flux(
    momentum: float,
    spatial_gradient,
    *,
    parameters: ConvexCarrierParameters = DEFAULT_CONVEX_CARRIER_PARAMETERS,
) -> np.ndarray:
    """Return ``dH/d(Pi,grad sigma)=mu(Pi,grad sigma)``."""

    params = parameters.validated()
    gradient = np.asarray(spatial_gradient, dtype=float)
    phase = np.concatenate(([float(momentum)], gradient))
    if gradient.ndim != 1 or gradient.size == 0 or np.any(~np.isfinite(phase)):
        raise ValueError("carrier phase-space values must form a finite vector")
    ratio = float(np.linalg.norm(phase) / params.acceleration_scale)
    mu = float(carrier_response_mu(ratio, epsilon=params.epsilon))
    return mu * phase


def carrier_phase_space_hessian(
    momentum: float,
    spatial_gradient,
    *,
    parameters: ConvexCarrierParameters = DEFAULT_CONVEX_CARRIER_PARAMETERS,
) -> np.ndarray:
    """Return the analytic Hessian of the convex Hamiltonian."""

    params = parameters.validated()
    gradient = np.asarray(spatial_gradient, dtype=float)
    phase = np.concatenate(([float(momentum)], gradient))
    if gradient.ndim != 1 or gradient.size == 0 or np.any(~np.isfinite(phase)):
        raise ValueError("carrier phase-space values must form a finite vector")
    norm = float(np.linalg.norm(phase))
    ratio = norm / params.acceleration_scale
    transverse = float(carrier_response_mu(ratio, epsilon=params.epsilon))
    radial = float(carrier_radial_curvature(ratio, epsilon=params.epsilon))
    hessian = transverse * np.eye(phase.size)
    if norm > 0.0:
        unit = phase / norm
        hessian += (radial - transverse) * np.outer(unit, unit)
    return hessian


def carrier_characteristic_speeds(
    momentum: float,
    spatial_gradient,
    propagation_direction,
    *,
    parameters: ConvexCarrierParameters = DEFAULT_CONVEX_CARRIER_PARAMETERS,
) -> dict[str, float | bool]:
    """Return both scalar characteristic speeds on an arbitrary background.

    With ``A=H_PiPi``, ``b=n.H_Pi,s`` and ``C=n.H_ss.n``, Hamilton's
    equations give ``(c+b)^2=A C``.  The relevant 2x2 block is a principal
    submatrix of the positive Hamiltonian Hessian.  Its largest eigenvalue is
    an analytic upper bound on both absolute characteristic speeds.
    """

    gradient = np.asarray(spatial_gradient, dtype=float)
    direction = np.asarray(propagation_direction, dtype=float)
    if direction.shape != gradient.shape or np.any(~np.isfinite(direction)):
        raise ValueError("propagation_direction must match the finite gradient")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0:
        raise ValueError("propagation_direction must be nonzero")
    direction = direction / direction_norm
    hessian = carrier_phase_space_hessian(
        momentum,
        gradient,
        parameters=parameters,
    )
    temporal = float(hessian[0, 0])
    mixing = float(direction @ hessian[1:, 0])
    spatial = float(direction @ hessian[1:, 1:] @ direction)
    discriminant = temporal * spatial
    root = np.sqrt(max(0.0, discriminant))
    speeds = (-mixing - root, -mixing + root)
    submatrix = np.asarray([[temporal, mixing], [mixing, spatial]])
    largest_submatrix_eigenvalue = float(np.max(np.linalg.eigvalsh(submatrix)))
    maximum_absolute_speed = max(abs(value) for value in speeds)
    return {
        "temporal_hessian": temporal,
        "momentum_gradient_mixing": mixing,
        "directional_spatial_hessian": spatial,
        "discriminant": discriminant,
        "negative_characteristic_speed": float(speeds[0]),
        "positive_characteristic_speed": float(speeds[1]),
        "maximum_absolute_characteristic_speed": maximum_absolute_speed,
        "largest_relevant_hessian_eigenvalue": largest_submatrix_eigenvalue,
        "hyperbolic": bool(discriminant > 0.0),
        "causal_in_preferred_unit_cone": bool(
            maximum_absolute_speed <= 1.0 + 1.0e-12
        ),
    }


def carrier_legendre_state(
    time_derivative: float,
    spatial_gradient,
    *,
    parameters: ConvexCarrierParameters = DEFAULT_CONVEX_CARRIER_PARAMETERS,
) -> dict[str, float]:
    """Invert ``dot sigma=dH/dPi`` and return the local Lagrangian state."""

    params = parameters.validated()
    velocity = float(time_derivative)
    gradient = np.asarray(spatial_gradient, dtype=float)
    if gradient.ndim != 1 or gradient.size == 0:
        raise ValueError("spatial_gradient must be a nonempty vector")
    if not np.isfinite(velocity) or np.any(~np.isfinite(gradient)):
        raise ValueError("Legendre data must be finite")
    if velocity == 0.0:
        momentum = 0.0
    else:
        sign = np.sign(velocity)
        target = abs(velocity)

        def residual(positive_momentum: float) -> float:
            flux = carrier_phase_space_flux(
                positive_momentum,
                gradient,
                parameters=params,
            )
            return float(flux[0] - target)

        lower = target
        upper = target / params.epsilon
        momentum = float(sign * optimize.brentq(residual, lower, upper))
    flux = carrier_phase_space_flux(momentum, gradient, parameters=params)
    hamiltonian = carrier_hamiltonian_density(momentum, gradient, parameters=params)
    lagrangian = momentum * velocity - hamiltonian
    hessian = carrier_phase_space_hessian(momentum, gradient, parameters=params)
    return {
        "time_derivative": velocity,
        "momentum": momentum,
        "momentum_map_residual": float(flux[0] - velocity),
        "hamiltonian_density": hamiltonian,
        "lagrangian_density": lagrangian,
        "legendre_reconstruction_residual": float(
            momentum * velocity - lagrangian - hamiltonian
        ),
        "hamiltonian_momentum_curvature": float(hessian[0, 0]),
        "lagrangian_time_kinetic_curvature": float(1.0 / hessian[0, 0]),
    }


def numerical_flux_jacobian(
    momentum: float,
    spatial_gradient,
    *,
    parameters: ConvexCarrierParameters = DEFAULT_CONVEX_CARRIER_PARAMETERS,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Differentiate the phase-space flux independently by centered differences."""

    params = parameters.validated()
    gradient = np.asarray(spatial_gradient, dtype=float)
    phase = np.concatenate(([float(momentum)], gradient))
    step_fraction = float(relative_step)
    if not np.isfinite(step_fraction) or step_fraction <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    step = step_fraction * max(params.acceleration_scale, float(np.linalg.norm(phase)))
    output = np.empty((phase.size, phase.size), dtype=float)
    for column in range(phase.size):
        upper = phase.copy()
        lower = phase.copy()
        upper[column] += step
        lower[column] -= step
        upper_flux = carrier_phase_space_flux(
            float(upper[0]),
            upper[1:],
            parameters=params,
        )
        lower_flux = carrier_phase_space_flux(
            float(lower[0]),
            lower[1:],
            parameters=params,
        )
        output[:, column] = (upper_flux - lower_flux) / (2.0 * step)
    return output
