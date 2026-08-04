"""Weak-field physical-metric projection for the Sigma v7C scalar control.

The frozen v7C equation evolves only the helicity-zero field ``pi``.  In the
decoupling-limit field redefinition, its leading physical-metric contribution
is conformal,

    delta h_{mu nu} = -eta_{mu nu} pi.

For the weak-field convention

    ds^2 = -(1 + 2 Psi) dt^2 + (1 - 2 Phi) dx^i dx^i,

this gives ``delta Psi=-pi/2`` and ``delta Phi=+pi/2``.  The scalar therefore
cancels identically from the Weyl potential ``(Psi+Phi)/2``.  This module also
keeps the disformal and helicity-2 possibilities explicit: they can affect
null rays, but they were not closed by the frozen scalar-only v7C equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class PotentialProjection:
    """Scalar contributions to the two metric potentials and their mean."""

    psi: Array
    phi: Array
    weyl: Array


def _finite_array(values: Array | float, *, name: str) -> Array:
    result = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def conformal_helicity0_projection(scalar: Array | float) -> PotentialProjection:
    """Project ``delta h_mn=-eta_mn*pi`` into ``Psi``, ``Phi``, and Weyl.

    With signature ``(-,+,+,+)``, ``delta h_00=+pi`` while
    ``delta h_ij=-pi delta_ij``.  Comparing with the weak-field metric gives
    the opposite potential shifts returned here.
    """

    field = _finite_array(scalar, name="scalar")
    psi = -0.5 * field
    phi = 0.5 * field
    return PotentialProjection(psi=psi, phi=phi, weyl=0.5 * (psi + phi))


def massive_spin2_helicity_decomposition(
    scalar_residue: float = 1.0 / 3.0,
) -> dict[str, dict[str, float] | float]:
    """Return the vDVZ weak-field decomposition in GR-normalized units.

    The helicity-2 response contributes equally to ``Psi`` and ``Phi``.  The
    helicity-zero response contributes with opposite signs.  With the
    Fierz--Pauli residue ``1/3``, the total is ``Psi=4/3``, ``Phi=2/3`` and
    ``Weyl=1`` before calibrating Newton's constant to the massive force.
    """

    residue = float(scalar_residue)
    if not np.isfinite(residue) or residue < 0.0:
        raise ValueError("scalar_residue must be finite and non-negative")
    helicity2 = {"psi": 1.0, "phi": 1.0, "weyl": 1.0}
    helicity0 = {"psi": residue, "phi": -residue, "weyl": 0.0}
    total_psi = helicity2["psi"] + helicity0["psi"]
    total_phi = helicity2["phi"] + helicity0["phi"]
    total_weyl = 0.5 * (total_psi + total_phi)
    return {
        "helicity2": helicity2,
        "helicity0": helicity0,
        "total": {"psi": total_psi, "phi": total_phi, "weyl": total_weyl},
        "ppn_gamma": total_phi / total_psi,
        "cavendish_normalized_weyl": total_weyl / total_psi,
    }


def static_disformal_metric(
    gradient: Array,
    coefficient: float,
) -> Array:
    """Return ``delta h_mn=-D partial_m pi partial_n pi`` for static ``pi``.

    The last dimension of ``gradient`` must contain the three spatial
    derivatives.  The returned array has shape ``gradient.shape[:-1]+(4,4)``.
    A static scalar has no time derivative, hence the entire time row and
    column vanish while the spatial block is anisotropic.
    """

    spatial_gradient = _finite_array(gradient, name="gradient")
    if spatial_gradient.ndim == 0 or spatial_gradient.shape[-1] != 3:
        raise ValueError("gradient must have a final dimension of length three")
    strength = float(coefficient)
    if not np.isfinite(strength):
        raise ValueError("coefficient must be finite")
    metric = np.zeros(spatial_gradient.shape[:-1] + (4, 4), dtype=float)
    metric[..., 1:, 1:] = -strength * np.einsum(
        "...i,...j->...ij", spatial_gradient, spatial_gradient
    )
    return metric


def null_metric_contraction(metric: Array, spatial_direction: Array) -> Array:
    """Return ``delta h_mn k^m k^n`` for ``k=(1,n)`` and unit ``n``."""

    perturbation = _finite_array(metric, name="metric")
    direction = _finite_array(spatial_direction, name="spatial_direction")
    if perturbation.shape[-2:] != (4, 4):
        raise ValueError("metric must have final dimensions (4, 4)")
    if direction.shape[-1:] != (3,):
        raise ValueError("spatial_direction must have a final dimension of length three")
    norm = np.linalg.norm(direction, axis=-1)
    if not np.allclose(norm, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("spatial_direction must be unit normalized")
    null_vector = np.concatenate((np.ones(direction.shape[:-1] + (1,)), direction), axis=-1)
    return np.einsum("...i,...ij,...j->...", null_vector, perturbation, null_vector)


def audit_v7c_metric_projection(
    *,
    scalar_samples: Array,
    conformal_cancellation_tolerance: float,
    minimum_nonzero_null_response: float,
    disformal_mapping_frozen: bool,
    complete_scalar_equation_frozen: bool,
    coupled_tensor_equation_frozen: bool,
) -> dict[str, object]:
    """Audit whether the frozen v7C scalar closes a nonzero lensing metric."""

    tolerance = float(conformal_cancellation_tolerance)
    minimum_response = float(minimum_nonzero_null_response)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("conformal_cancellation_tolerance must be non-negative")
    if not np.isfinite(minimum_response) or minimum_response <= 0.0:
        raise ValueError("minimum_nonzero_null_response must be positive")

    projection = conformal_helicity0_projection(scalar_samples)
    maximum_conformal_weyl = float(np.max(np.abs(projection.weyl)))
    decomposition = massive_spin2_helicity_decomposition()

    gradient = np.array([1.0, 0.0, 0.0])
    disformal = static_disformal_metric(gradient, 1.0)
    aligned_response = float(
        null_metric_contraction(disformal, np.array([1.0, 0.0, 0.0]))
    )
    orthogonal_response = float(
        null_metric_contraction(disformal, np.array([0.0, 1.0, 0.0]))
    )
    frozen_nonzero_lensing_term = bool(
        disformal_mapping_frozen
        and complete_scalar_equation_frozen
        and abs(aligned_response) >= minimum_response
    ) or bool(coupled_tensor_equation_frozen)

    gates = {
        "conformal_projection_identity": maximum_conformal_weyl <= tolerance,
        "vdvz_helicity_decomposition": np.isclose(
            float(decomposition["total"]["psi"]), 4.0 / 3.0
        )
        and np.isclose(float(decomposition["total"]["phi"]), 2.0 / 3.0)
        and np.isclose(float(decomposition["total"]["weyl"]), 1.0),
        "action_derived_nonzero_weyl_or_null_response": frozen_nonzero_lensing_term,
        "complete_scalar_metric_mapping": bool(
            disformal_mapping_frozen and complete_scalar_equation_frozen
        ),
        "coupled_tensor_closure_if_used": bool(coupled_tensor_equation_frozen),
    }
    return {
        "maximum_absolute_conformal_weyl_response": maximum_conformal_weyl,
        "massive_spin2_decomposition": decomposition,
        "static_disformal_diagnostic": {
            "time_time_component": float(disformal[0, 0]),
            "aligned_null_contraction": aligned_response,
            "orthogonal_null_contraction": orthogonal_response,
        },
        "frozen_content": {
            "disformal_mapping": bool(disformal_mapping_frozen),
            "complete_scalar_equation": bool(complete_scalar_equation_frozen),
            "coupled_tensor_equation": bool(coupled_tensor_equation_frozen),
        },
        "gates": {name: bool(value) for name, value in gates.items()},
    }
