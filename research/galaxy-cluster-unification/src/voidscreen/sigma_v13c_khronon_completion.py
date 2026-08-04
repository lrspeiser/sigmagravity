"""Minimal one-metric khronon embedding of the v13B carrier.

The covariant ingredients are a normalized khronon normal ``u_mu``, its
expansion ``Theta=nabla_mu u^mu``, and its acceleration
``a_mu=u^nu nabla_nu u_mu``.  An auxiliary momentum ``p`` gives the local
first-order carrier

``L_C=p Theta-H_13B(p,a)``.

Subtracting the canonical reference ``L_0=(Theta^2-a^2)/2`` makes the static
GR plus modifier energy equal to the v13B AQUAL energy.  The same subtraction
also fixes the temporal completion: at a static acceleration background its
extra trace-kinetic curvature is

``delta=(1/mu)-1=(1-epsilon)/(epsilon+a/a_sigma)``.

In adapted ADM variables this changes

``K_ij K^ij-K^2 -> K_ij K^ij-lambda_eff K^2``

with ``lambda_eff=1+c_trace-delta/2``.  Eliminating the scalar shift gives
the exact kinetic coefficient ``2(1-3 lambda_eff)/(1-lambda_eff)``.  With no
counterterm, every sufficiently high but finite acceleration lies in the
ghost interval ``1/3 < lambda_eff < 1``.  A finite constant counterterm either
still crosses that interval or fails the GR high-field limit.

This module audits that minimal embedding.  It does not reject the standalone
convexity theorem for v13B and it does not assert that every conceivable
covariant placement of the carrier has this ADM trace structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.sigma_v13b_convex_carrier import (
    carrier_response_mu,
    carrier_shape,
)


@dataclass(frozen=True)
class KhrononCompletionParameters:
    """Parameters of the minimal same-weight trace completion."""

    epsilon: float = 1.0e-6
    completion_weight: float = 1.0
    trace_counterterm: float = 0.0

    def validated(self) -> KhrononCompletionParameters:
        values = np.asarray(
            [self.epsilon, self.completion_weight, self.trace_counterterm],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("khronon completion parameters must be finite")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie strictly between zero and one")
        if self.completion_weight <= 0.0:
            raise ValueError("completion_weight must be positive")
        return self


DEFAULT_KHRONON_COMPLETION_PARAMETERS = KhrononCompletionParameters()


def static_susceptibility(
    acceleration_ratio,
    *,
    epsilon: float,
) -> np.ndarray:
    """Return ``chi=mu-1`` for the published khronon ``f(a)`` convention."""

    ratio = np.asarray(acceleration_ratio, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("acceleration_ratio must be finite and nonnegative")
    return carrier_response_mu(ratio, epsilon=epsilon) - 1.0


def dimensionless_khronon_static_function(
    acceleration_ratio,
    *,
    epsilon: float,
) -> np.ndarray:
    """Return ``f/a_sigma^2=F_epsilon(x)-x^2``.

    It obeys ``(df/dx)/(2x)=mu-1`` away from the origin and has the regular
    limiting derivative zero at the origin.
    """

    ratio = np.asarray(acceleration_ratio, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("acceleration_ratio must be finite and nonnegative")
    return carrier_shape(ratio, epsilon=epsilon) - ratio**2


def temporal_excess_curvature(
    acceleration_ratio,
    *,
    parameters: KhrononCompletionParameters = (
        DEFAULT_KHRONON_COMPLETION_PARAMETERS
    ),
) -> np.ndarray:
    """Return the exact additional ``d^2 L/dTheta^2`` over its reference.

    The stable closed form avoids subtracting two nearly equal numbers when
    ``a/a_sigma`` is very large.
    """

    params = parameters.validated()
    ratio = np.asarray(acceleration_ratio, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("acceleration_ratio must be finite and nonnegative")
    return (
        params.completion_weight
        * (1.0 - params.epsilon)
        / (params.epsilon + ratio)
    )


def effective_adm_lambda(
    acceleration_ratio,
    *,
    parameters: KhrononCompletionParameters = (
        DEFAULT_KHRONON_COMPLETION_PARAMETERS
    ),
) -> np.ndarray:
    """Return ``lambda`` in ``K_ij K^ij-lambda K^2``."""

    params = parameters.validated()
    excess = temporal_excess_curvature(
        acceleration_ratio,
        parameters=params,
    )
    return 1.0 + params.trace_counterterm - 0.5 * excess


def scalar_shift_reduced_kinetic_coefficient(adm_lambda) -> np.ndarray:
    """Eliminate the scalar shift from the ADM trace kinetic block.

    For ``K_ij=dot(zeta) delta_ij+k_i k_j B`` the unreduced coefficients are

    ``A=3-9 lambda``, ``B_cross=2-6 lambda``, ``C=1-lambda``.

    Eliminating ``q=k^2 B`` yields
    ``A-B_cross^2/(4C)=2(1-3lambda)/(1-lambda)``.
    ``lambda=1`` is the GR constraint point and is returned as NaN rather than
    misclassified as a propagating scalar.
    """

    value = np.asarray(adm_lambda, dtype=float)
    if np.any(~np.isfinite(value)):
        raise ValueError("adm_lambda must be finite")
    denominator = 1.0 - value
    with np.errstate(divide="ignore", invalid="ignore"):
        coefficient = 2.0 * (1.0 - 3.0 * value) / denominator
    return np.where(np.abs(denominator) <= 1.0e-14, np.nan, coefficient)


def scalar_shift_block(adm_lambda: float) -> dict[str, float | bool]:
    """Return direct and Schur-complement forms of the scalar kinetic block."""

    lam = float(adm_lambda)
    if not np.isfinite(lam):
        raise ValueError("adm_lambda must be finite")
    dot_squared = 3.0 - 9.0 * lam
    dot_shift = 2.0 - 6.0 * lam
    shift_squared = 1.0 - lam
    if abs(shift_squared) <= 1.0e-14:
        reduced = np.nan
        stationary_shift_per_dot = np.nan
    else:
        stationary_shift_per_dot = -dot_shift / (2.0 * shift_squared)
        reduced = dot_squared - dot_shift**2 / (4.0 * shift_squared)
    analytic = float(scalar_shift_reduced_kinetic_coefficient(lam))
    ghost_interval = 1.0 / 3.0 < lam < 1.0
    return {
        "adm_lambda": lam,
        "dot_squared_coefficient": dot_squared,
        "dot_shift_coefficient": dot_shift,
        "shift_squared_coefficient": shift_squared,
        "stationary_shift_per_dot": stationary_shift_per_dot,
        "direct_schur_kinetic_coefficient": reduced,
        "analytic_reduced_kinetic_coefficient": analytic,
        "schur_identity_residual": (
            np.nan if np.isnan(reduced) else float(reduced - analytic)
        ),
        "in_standard_ghost_interval": ghost_interval,
        "positive_reduced_scalar_kinetic": bool(
            np.isfinite(reduced) and reduced > 0.0
        ),
        "gr_constraint_point": bool(abs(lam - 1.0) <= 1.0e-14),
    }


def khronon_completion_row(
    acceleration_ratio: float,
    *,
    parameters: KhrononCompletionParameters = (
        DEFAULT_KHRONON_COMPLETION_PARAMETERS
    ),
) -> dict[str, float | bool]:
    """Return all local static and scalar-kinetic diagnostics."""

    params = parameters.validated()
    ratio = float(acceleration_ratio)
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("acceleration_ratio must be finite and nonnegative")
    mu = float(carrier_response_mu(ratio, epsilon=params.epsilon))
    excess = float(temporal_excess_curvature(ratio, parameters=params))
    lam = float(effective_adm_lambda(ratio, parameters=params))
    block = scalar_shift_block(lam)
    return {
        "acceleration_over_a_sigma": ratio,
        "static_mu": mu,
        "static_fractional_extra_force": 1.0 / mu - 1.0,
        "temporal_excess_curvature": excess,
        **block,
    }


def traceless_tensor_modifier_contraction(
    tensor_polarization,
    *,
    trace_curvature: float,
) -> float:
    """Contract the pure-trace completion Hessian with a tensor polarization."""

    polarization = np.asarray(tensor_polarization, dtype=float)
    curvature = float(trace_curvature)
    if (
        polarization.shape != (3, 3)
        or np.any(~np.isfinite(polarization))
        or not np.isfinite(curvature)
    ):
        raise ValueError("require a finite 3x3 polarization and curvature")
    if not np.allclose(polarization, polarization.T, atol=1.0e-12):
        raise ValueError("tensor_polarization must be symmetric")
    trace = float(np.trace(polarization))
    return curvature * trace**2
