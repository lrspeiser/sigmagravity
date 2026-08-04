"""Covariant action-gate tests for the Sigma v14 gauge-tidal carrier.

The flat rank-two scalar gauge transformation is

``delta A_mn = partial_m partial_n alpha``.

For the naive covariant field strength ``F_mn|r=2 nabla_[m A_n]r`` its gauge
variation is ``R_mnr{}^s nabla_s alpha``.  A partially-massless curvature
correction cancels the constant-curvature part but leaves the Weyl contraction.
The four-derivative conformal/Bach completion is covariant, but its quadratic
propagator has opposite residues.  These identities implement the three
theory-only v14 carrier screens without opening observational data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PropagatorResidues:
    """Residues in a two-pole fourth-order propagator."""

    massless: float
    massive: float


def _validated_metric(metric) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(metric, dtype=float)
    if value.shape != (4, 4) or np.any(~np.isfinite(value)):
        raise ValueError("metric must be a finite 4x4 array")
    if not np.allclose(value, value.T, atol=1.0e-12):
        raise ValueError("metric must be symmetric")
    inverse = np.linalg.inv(value)
    return value, inverse


def constant_curvature_riemann(metric, *, curvature: float) -> np.ndarray:
    """Return ``R_mnrs=H2(g_mr g_ns-g_ms g_nr)``."""

    value, _ = _validated_metric(metric)
    scale = float(curvature)
    if not np.isfinite(scale):
        raise ValueError("curvature must be finite")
    return scale * (
        np.einsum("mr,ns->mnrs", value, value)
        - np.einsum("ms,nr->mnrs", value, value)
    )


def electric_weyl_tensor(electric_tidal) -> np.ndarray:
    """Build a purely electric Weyl tensor in an orthonormal frame.

    ``electric_tidal`` must be a symmetric trace-free 3x3 matrix.  The metric
    convention is ``diag(-1,+1,+1,+1)``.
    """

    electric = np.asarray(electric_tidal, dtype=float)
    if electric.shape != (3, 3) or np.any(~np.isfinite(electric)):
        raise ValueError("electric_tidal must be a finite 3x3 array")
    if not np.allclose(electric, electric.T, atol=1.0e-12):
        raise ValueError("electric_tidal must be symmetric")
    if abs(float(np.trace(electric))) > 1.0e-12:
        raise ValueError("electric_tidal must be trace free")
    tensor = np.zeros((4, 4, 4, 4), dtype=float)
    for i in range(3):
        for j in range(3):
            ii = i + 1
            jj = j + 1
            component = electric[i, j]
            tensor[0, ii, 0, jj] = component
            tensor[ii, 0, 0, jj] = -component
            tensor[0, ii, jj, 0] = -component
            tensor[ii, 0, jj, 0] = component
    identity = np.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ell in range(3):
                    tensor[i + 1, j + 1, k + 1, ell + 1] = (
                        identity[i, k] * electric[j, ell]
                        + identity[j, ell] * electric[i, k]
                        - identity[i, ell] * electric[j, k]
                        - identity[j, k] * electric[i, ell]
                    )
    return tensor


def riemann_symmetry_residuals(riemann, *, metric) -> dict[str, float]:
    """Return algebraic Riemann/Weyl symmetry and trace residuals."""

    tensor = np.asarray(riemann, dtype=float)
    _, inverse = _validated_metric(metric)
    if tensor.shape != (4, 4, 4, 4) or np.any(~np.isfinite(tensor)):
        raise ValueError("riemann must be a finite 4x4x4x4 array")
    trace = np.einsum("mr,mnrs->ns", inverse, tensor)
    return {
        "first_pair_antisymmetry": float(
            np.max(np.abs(tensor + tensor.swapaxes(0, 1)))
        ),
        "second_pair_antisymmetry": float(
            np.max(np.abs(tensor + tensor.swapaxes(2, 3)))
        ),
        "pair_exchange_symmetry": float(
            np.max(np.abs(tensor - tensor.transpose(2, 3, 0, 1)))
        ),
        "single_trace": float(np.max(np.abs(trace))),
    }


def minimal_covariant_gauge_residual(riemann, gauge_gradient) -> np.ndarray:
    """Return ``R_mnrs q^s`` for the naive covariant field strength."""

    tensor = np.asarray(riemann, dtype=float)
    gradient = np.asarray(gauge_gradient, dtype=float)
    if (
        tensor.shape != (4, 4, 4, 4)
        or gradient.shape != (4,)
        or np.any(~np.isfinite(tensor))
        or np.any(~np.isfinite(gradient))
    ):
        raise ValueError("require finite Riemann tensor and four-gradient")
    return np.einsum("mnrs,s->mnr", tensor, gradient)


def partially_massless_gauge_residual(
    riemann,
    gauge_gradient,
    *,
    metric,
    curvature_counterterm: float,
) -> np.ndarray:
    """Return the residual after ``delta A_mn=(nabla_mn+H2 g_mn)alpha``."""

    value, _ = _validated_metric(metric)
    gradient = np.asarray(gauge_gradient, dtype=float)
    if gradient.shape != (4,) or np.any(~np.isfinite(gradient)):
        raise ValueError("gauge_gradient must be a finite four-vector")
    scale = float(curvature_counterterm)
    if not np.isfinite(scale):
        raise ValueError("curvature_counterterm must be finite")
    covector = value @ gradient
    residual = minimal_covariant_gauge_residual(riemann, gradient)
    correction = scale * (
        np.einsum("nr,m->mnr", value, covector)
        - np.einsum("mr,n->mnr", value, covector)
    )
    return residual + correction


def tracefree_stress_double_divergence(
    *,
    wave_covector_squared: float,
    stress_trace: float,
    spacetime_dimension: int = 4,
) -> float:
    """Gauge variation coefficient for a conserved trace-free stress source."""

    if spacetime_dimension <= 1:
        raise ValueError("spacetime_dimension must exceed one")
    values = np.asarray([wave_covector_squared, stress_trace], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("source diagnostics must be finite")
    return -float(wave_covector_squared) * float(stress_trace) / spacetime_dimension


def curved_improvement_divergence(ricci, scalar_gradient) -> np.ndarray:
    """Return the curvature obstruction ``R_ns nabla^s S``.

    This is the divergence of the flat conserved improvement
    ``J_mn=(nabla_m nabla_n-g_mn box)S`` on a curved background, up to the
    sign convention chosen for the Riemann tensor.
    """

    tensor = np.asarray(ricci, dtype=float)
    gradient = np.asarray(scalar_gradient, dtype=float)
    if (
        tensor.shape != (4, 4)
        or gradient.shape != (4,)
        or np.any(~np.isfinite(tensor))
        or np.any(~np.isfinite(gradient))
    ):
        raise ValueError("require finite Ricci tensor and four-gradient")
    return tensor @ gradient


def fourth_order_propagator_residues(*, massive_pole_squared: float) -> PropagatorResidues:
    """Return residues of ``1/[k^2(k^2+m^2)]``.

    The partial fraction is ``(1/m^2)[1/k^2-1/(k^2+m^2)]``.  Opposite signs
    are the quadratic negative-energy obstruction of a local fourth-order
    completion such as the conformal/Bach lane.
    """

    mass_squared = float(massive_pole_squared)
    if not np.isfinite(mass_squared) or mass_squared <= 0.0:
        raise ValueError("massive_pole_squared must be finite and positive")
    residue = 1.0 / mass_squared
    return PropagatorResidues(massless=residue, massive=-residue)
